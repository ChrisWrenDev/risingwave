// Copyright 2025 RisingWave Labs
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use faiss::{ConcurrentIndex, Index, MetricType};
use rand::rngs::StdRng;
use rand::{Rng as _, SeedableRng};
use risingwave_storage::vector::distance::InnerProductDistance;
use risingwave_storage::vector::hnsw::{HnswGraph, VectorAccessor, VectorStore, nearest};
use risingwave_storage::vector::{NearestBuilder, VectorInner, VectorItem, VectorRef};

/// Match the original big test
const D: usize = 128;
const N: usize = 20_000;
const Q: usize = 5_000;
const K: usize = 10;
const M: usize = 40;

const SEED: u64 = 233;
const EF_SWEEP: &[usize] = &[16]; // extend to &[16,30,100] if you like

// ---------- utilities (public APIs only) ----------

fn vref(x: &[VectorItem]) -> VectorRef<'_> {
    VectorInner::from_slice(x)
}

fn gen_dataset(n: usize, d: usize, seed: u64) -> (Vec<Vec<VectorItem>>, Vec<Vec<u8>>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut xs = Vec::with_capacity(n);
    for _ in 0..n {
        let mut v = Vec::with_capacity(d);
        for _ in 0..d {
            // use `random` (Rust 2024: `gen` is a keyword)
            v.push(rng.random::<f32>());
        }
        xs.push(v);
    }
    let infos = (0..n).map(|i| (i as u32).to_le_bytes().to_vec()).collect();
    (xs, infos)
}

fn gen_queries(q: usize, d: usize, seed: u64) -> Vec<Vec<VectorItem>> {
    let mut rng = StdRng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15);
    (0..q)
        .map(|_| {
            let mut v = Vec::with_capacity(d);
            for _ in 0..d {
                v.push(rng.random::<f32>());
            }
            v
        })
        .collect()
}

fn exact_topk_for_queries(
    xs: &[Vec<VectorItem>],
    infos: &[Vec<u8>],
    queries: &[Vec<VectorItem>],
) -> Vec<Vec<Vec<u8>>> {
    queries
        .iter()
        .map(|q| {
            let mut nb = NearestBuilder::<'_, _, InnerProductDistance>::new(vref(q), K);
            nb.add(
                xs.iter()
                    .zip(infos.iter())
                    .map(|(x, i)| (vref(x), i.as_slice())),
                |_, _, info| info.to_vec(),
            );
            nb.finish()
        })
        .collect()
}

fn recall(actual: &[Vec<u8>], expected: &[Vec<u8>]) -> f32 {
    use std::collections::HashSet;
    let e: HashSet<_> = expected.iter().map(|b| b.as_slice()).collect();
    let a: HashSet<_> = actual.iter().map(|b| b.as_slice()).collect();
    (a.intersection(&e).count() as f32) / (e.len() as f32)
}

// ---------- bench-local VectorStore + FAISS-backed HNSW graph ----------

struct BenchStore {
    d: usize,
    xs_flat: Vec<VectorItem>, // flattened N*D
    infos: Vec<Vec<u8>>,
}
impl BenchStore {
    fn new(xs: &[Vec<VectorItem>], infos: &[Vec<u8>]) -> Self {
        let d = xs.first().map(|v| v.len()).unwrap_or(0);
        let xs_flat = xs.iter().flat_map(|v| v.iter().copied()).collect();
        Self {
            d,
            xs_flat,
            infos: infos.to_vec(),
        }
    }

    fn vref_at(&self, i: usize) -> VectorRef<'_> {
        let start = i * self.d;
        let end = start + self.d;
        vref(&self.xs_flat[start..end])
    }
}
struct BenchAcc<'a> {
    s: &'a BenchStore,
    i: usize,
}
impl VectorAccessor for BenchAcc<'_> {
    fn vec_ref(&self) -> VectorRef<'_> {
        self.s.vref_at(self.i)
    }

    fn info(&self) -> &[u8] {
        &self.s.infos[self.i]
    }
}
impl VectorStore for BenchStore {
    type Accessor<'a>
        = BenchAcc<'a>
    where
        Self: 'a;

    async fn get_vector(
        &self,
        idx: usize,
    ) -> risingwave_storage::hummock::HummockResult<Self::Accessor<'_>> {
        Ok(BenchAcc { s: self, i: idx })
    }
}

// Graph adapter that exposes FAISS’ HNSW as our public HnswGraph
struct FaissGraph {
    entry: usize,
    levels: Vec<usize>,
    neigh: Vec<Vec<Vec<usize>>>, // [node][level] -> neighbors
}
impl HnswGraph for FaissGraph {
    fn entrypoint(&self) -> usize {
        self.entry
    }

    fn len(&self) -> usize {
        self.levels.len()
    }

    fn node_level(&self, idx: usize) -> usize {
        self.levels[idx]
    }

    fn node_neighbours(
        &self,
        idx: usize,
        level: usize,
    ) -> impl Iterator<Item = (usize, risingwave_storage::vector::VectorDistance)> + '_ {
        // distance is recomputed by `nearest()`, so we can return 0.0 here
        self.neigh[idx][level].iter().copied().map(|n| (n, 0.0))
    }
}

fn build_faiss_graph(xs: &[Vec<VectorItem>]) -> FaissGraph {
    let mut faiss_idx =
        faiss::index::hnsw::HnswFlatIndex::new(D as _, M as _, MetricType::InnerProduct).unwrap();
    let flat: Vec<f32> = xs.iter().flat_map(|v| v.iter().copied()).collect();
    faiss_idx.add(&flat).unwrap();

    let h = faiss_idx.hnsw();
    let (entry, _max_lv) = h.entry_point().unwrap();
    let levels_raw = h.levels_raw();
    let levels: Vec<usize> = levels_raw.iter().map(|&l| l as usize).collect();

    let mut neigh = Vec::with_capacity(levels.len());
    for (node, &lv_cnt) in levels.iter().enumerate() {
        let mut lv = Vec::with_capacity(lv_cnt);
        for l in 0..lv_cnt {
            let nbrs = h.neighbors_raw(node, l);
            lv.push(nbrs.iter().map(|&i| i as usize).collect());
        }
        neigh.push(lv);
    }

    FaissGraph {
        entry,
        levels,
        neigh,
    }
}

// ---------- Benchmarks ----------

fn bench_our_search_on_faiss_graph(c: &mut Criterion) {
    let (xs, infos) = gen_dataset(N, D, SEED);
    let queries = gen_queries(Q, D, SEED);

    let store = BenchStore::new(&xs, &infos);
    let graph = build_faiss_graph(&xs);

    // Precompute exact baseline once for recall check
    let expected = exact_topk_for_queries(&xs, &infos, &queries);

    let mut group = c.benchmark_group("our_search_on_faiss_graph_per_query");
    group.sample_size(20); // bump locally for stability
    group.throughput(Throughput::Elements(1)); // one query per iter

    for &ef in EF_SWEEP {
        group.bench_function(format!("nearest_ef{ef}_top{K}_per_query"), |b| {
            // simple round-robin over the query set
            let mut idx = 0usize;
            b.iter(|| {
                let i = idx % Q;
                idx += 1;
                let q = &queries[i];

                let (actual, _stats) =
                    futures::executor::block_on(nearest::<_, InnerProductDistance>(
                        &store,
                        &graph,
                        vref(q),
                        |_, _, info| info.to_vec(),
                        ef,
                        K,
                    ))
                    .unwrap();

                // Optionally check recall against precomputed expected[i]
                let _rec = recall(&actual, &expected[i]);

                black_box(actual);
            });
        });
    }
    group.finish();
}

fn bench_faiss_query(c: &mut Criterion) {
    let (xs, _infos) = gen_dataset(N, D, SEED);
    let queries = gen_queries(Q, D, SEED);

    let expected = exact_topk_for_queries(&xs, &_infos, &queries);

    // Build FAISS index
    let mut faiss_idx =
        faiss::index::hnsw::HnswFlatIndex::new(D as _, M as _, MetricType::InnerProduct).unwrap();
    let flat: Vec<f32> = xs.iter().flat_map(|v| v.iter().copied()).collect();
    faiss_idx.add(&flat).unwrap();

    let mut group = c.benchmark_group("faiss_query_per_query");
    group.sample_size(20); // bump locally for stability
    group.throughput(Throughput::Elements(1));

    group.bench_function("faiss_assign_top10_per_query", |b| {
        let mut idx = 0usize;
        b.iter(|| {
            let i = idx % Q;
            idx += 1;
            let q = &queries[i];

            let res = faiss_idx.assign(&q[..], K as _).unwrap();
            let actual: Vec<Vec<u8>> = res
                .labels
                .into_iter()
                .filter_map(|l| l.get().map(|idx| (idx as u32).to_le_bytes().to_vec()))
                .collect();

            let _rec = recall(&actual, &expected[i]);

            black_box(actual);
        });
    });
    group.finish();
}

criterion_group!(query, bench_our_search_on_faiss_graph, bench_faiss_query);
criterion_main!(query);
