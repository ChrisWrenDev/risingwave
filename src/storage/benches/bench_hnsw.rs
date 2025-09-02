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

use std::collections::HashSet;
use std::time::{Duration, Instant};

use bytes::Bytes;
use criterion::measurement::Measurement;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, Throughput, black_box, criterion_main};
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use risingwave_common::types::VectorVal;
use risingwave_common::vector::distance::InnerProductDistance;
use risingwave_storage::hummock::HummockResult;
use risingwave_storage::vector::hnsw::{
    HnswBuilder, HnswBuilderOptions, HnswGraphBuilder, VectorStoreImpl,
};
use risingwave_storage::vector::{NearestBuilder, VectorRef};
use tokio::runtime::Builder;

// -------------------------------
// CONFIGURATION - CI (SCALED)
// -------------------------------
#[cfg(not(feature = "hnsw-deep-bench"))]
mod cfgs {
    pub const VECTOR_LEN: usize = 128;
    pub const INPUT_COUNT: usize = 10_000; // smaller for CI
    pub const QUERY_COUNT: usize = 1_000; // smaller for CI
    pub const TOP_N: usize = 10;
    // You can expand this like &[16, 32, 64, 128] to probe the curve more deeply.
    pub const EF_SEARCH_LIST: &[usize] = &[8, 16, 32];
    pub const SEED: u64 = 233;
}

// -------------------------------
// CONFIGURATION - LOCAL (FULL)
// -------------------------------
#[cfg(feature = "hnsw-deep-bench")]
mod cfgs {
    pub const VECTOR_LEN: usize = 128;
    pub const INPUT_COUNT: usize = 20_000;
    pub const QUERY_COUNT: usize = 5_000;
    pub const TOP_N: usize = 10;
    // You can expand this like &[16, 32, 64, 128] to probe the curve more deeply.
    pub const EF_SEARCH_LIST: &[usize] = &[8, 16, 32];
    pub const SEED: u64 = 233;
}

use cfgs::*;

/// Repeat query passes (like the removed test): 1x in debug, 60x in release.
fn repeat_query_passes() -> usize {
    if cfg!(debug_assertions) { 1 } else { 60 }
}

/// Compute recall@k between actual top-k infos and ground-truth infos.
#[inline]
fn recall_k_ids(actual: &[usize], expected_set: &HashSet<usize>) -> f32 {
    if expected_set.is_empty() {
        return 1.0;
    }
    let hits = actual.iter().filter(|id| expected_set.contains(id)).count();
    hits as f32 / expected_set.len() as f32
}

fn gen_info(i: usize) -> Bytes {
    Bytes::copy_from_slice((i as u64).to_le_bytes().as_slice())
}

fn gen_vector_with(rng: &mut StdRng, d: usize) -> risingwave_common::types::VectorVal {
    VectorVal::from_iter((0..d).map(|_| rng.random::<f32>().try_into().unwrap()))
}

/// Generate input dataset (vectors + info payload).
fn make_input(n: usize, dim: usize) -> Vec<(VectorVal, Bytes)> {
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push((gen_vector_with(&mut rng, dim), gen_info(i)));
    }
    out
}

/// Pre-generate queries.
fn make_queries(n: usize, dim: usize) -> Vec<VectorVal> {
    let mut rng = StdRng::seed_from_u64(SEED ^ 0x5EED5EED); // different stream
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(gen_vector_with(&mut rng, dim));
    }
    out
}

/// Brute-force exact top-k (ground truth) with ``NearestBuilder`` on the full input.
fn ground_truth_ids_for_queries(
    queries: &[VectorVal],
    input: &[(VectorVal, Bytes)],
    k: usize,
) -> (Vec<Vec<usize>>, Vec<HashSet<usize>>) {
    let gt_lists: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| {
            let mut nb = NearestBuilder::<usize, InnerProductDistance>::new(q.to_ref(), k);
            nb.add(
                input.iter().map(|(v, info)| {
                    (VectorRef::from_slice_unchecked(v.as_slice()), info.as_ref())
                }),
                |_v, _d, info| info_to_usize(info),
            );
            nb.finish()
        })
        .collect();
    let gt_sets = gt_lists
        .iter()
        .map(|lst| lst.iter().cloned().collect())
        .collect();
    (gt_lists, gt_sets)
}

fn tune_heavy<G: Measurement>(g: &mut BenchmarkGroup<'_, G>) {
    g.sample_size(10); // default is 100
    g.measurement_time(Duration::from_secs(12));
    g.warm_up_time(Duration::from_secs(3));
}

#[inline]
fn info_to_usize(info: &[u8]) -> usize {
    let bytes: [u8; std::mem::size_of::<usize>()] = info[..std::mem::size_of::<usize>()]
        .try_into()
        .expect("info too short");
    usize::from_le_bytes(bytes)
}

/// Build an HNSW index (Rust implementation).
async fn build_hnsw(
    dim: usize,
    input: &[(VectorVal, Bytes)],
) -> HummockResult<HnswBuilder<VectorStoreImpl, HnswGraphBuilder, InnerProductDistance, StdRng>> {
    let opts = HnswBuilderOptions {
        m: 40,
        ef_construction: 40,
        max_level: 10,
    };
    let mut hnsw =
        HnswBuilder::<_, _, InnerProductDistance, _>::new(dim, StdRng::seed_from_u64(SEED), opts);
    // First insert sets up the graph; rest do normal work.
    for (v, info) in input {
        hnsw.insert(VectorRef::from_slice_unchecked(v.as_slice()), info.as_ref())
            .await?;
    }
    Ok(hnsw)
}

// -------------------------------
// BENCHES - CI (SCALED)
// -------------------------------
fn bench_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_build");
    tune_heavy(&mut group);

    // Vary m and ef_construction; keep max_level moderate.
    let configs: &[(usize, usize, usize)] = &[
        (8, 32, 8),
        (16, 64, 8),
        (32, 64, 8),
        (40, 40, 10), // matches removed test
    ];

    let rt = Builder::new_current_thread().enable_all().build().unwrap();
    let input = make_input(INPUT_COUNT, VECTOR_LEN);

    for &(m, efc, max_level) in configs {
        group.throughput(Throughput::Bytes(
            (INPUT_COUNT * VECTOR_LEN * std::mem::size_of::<f32>()) as u64,
        ));
        group.throughput(Throughput::Elements(INPUT_COUNT as u64));
        group.bench_function(
            BenchmarkId::from_parameter(format!("m={m}_efc={efc}_L={max_level}")),
            |b| {
                b.to_async(&rt).iter(|| async {
                    // Build and accumulate per-insert stats.
                    let opts = HnswBuilderOptions {
                        m,
                        ef_construction: efc,
                        max_level,
                    };
                    let mut hnsw = HnswBuilder::<_, _, InnerProductDistance, _>::new(
                        VECTOR_LEN,
                        StdRng::seed_from_u64(SEED),
                        opts,
                    );
                    let mut dist_sum = 0usize;
                    let mut hops_sum = 0usize;
                    let mut inserts = 0usize;
                    for (v, info) in &input {
                        let stats = hnsw
                            .insert(VectorRef::from_slice_unchecked(v.as_slice()), info.as_ref())
                            .await
                            .unwrap();
                        // first insert returns default stats; still fine
                        dist_sum += stats.distances_computed();
                        hops_sum += stats.nhops();
                        inserts += 1;
                    }
                    // Store averages so optimizer can't drop the work.
                    black_box((
                        dist_sum as f64 / inserts as f64,
                        hops_sum as f64 / inserts as f64,
                    ));
                });
            },
        );
    }

    group.finish();
}

/// --- Bench 2: Query latency + recall@k (matches old test quality check) ---
fn bench_search_latency_and_recall(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search");
    tune_heavy(&mut group);

    let rt = Builder::new_current_thread().enable_all().build().unwrap();

    // Prepare dataset, index, queries, and ground-truth once.
    let input = make_input(INPUT_COUNT, VECTOR_LEN);
    let queries = make_queries(QUERY_COUNT, VECTOR_LEN);
    let (_expected_lists, expected_sets) = ground_truth_ids_for_queries(&queries, &input, TOP_N);
    let hnsw = rt.block_on(build_hnsw(VECTOR_LEN, &input)).unwrap();

    // Measure across ef_search settings.
    for &ef_search in EF_SEARCH_LIST {
        group.throughput(Throughput::Elements(
            queries.len() as u64 * repeat_query_passes() as u64,
        ));
        group.bench_function(
            BenchmarkId::from_parameter(format!(
                "n={}_d={}_ef={}_k={}",
                INPUT_COUNT, VECTOR_LEN, ef_search, TOP_N
            )),
            |b| {
                b.to_async(&rt).iter(|| async {
                    // Repeat the full query set several times (like the old test).
                    let mut total_recall = 0.0f64;
                    let mut total_queries = 0usize;
                    let mut dist_sum = 0usize;
                    let mut hops_sum = 0usize;
                    for _pass in 0..repeat_query_passes() {
                        for (i, q) in queries.iter().enumerate() {
                            let (actual, stats) = hnsw
                                .search::<usize>(
                                    VectorRef::from_slice_unchecked(black_box(q.as_slice())),
                                    |_v, _d, info| info_to_usize(info),
                                    ef_search,
                                    TOP_N,
                                )
                                .await
                                .unwrap();
                            // Accumulate recall (prevent optimizer dropping work).
                            let r = recall_k_ids(&actual, &expected_sets[i]) as f64;
                            total_recall += r;
                            total_queries += 1;
                            dist_sum += stats.distances_computed();
                            hops_sum += stats.nhops();
                        }
                    }
                    black_box((
                        total_recall / (total_queries as f64),
                        dist_sum as f64 / (total_queries as f64),
                        hops_sum as f64 / (total_queries as f64),
                    ));
                });
            },
        );
    }

    group.finish();
}

fn bench_determinism(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_determinism");
    let rt = Builder::new_current_thread().enable_all().build().unwrap();

    let input = make_input(10_000, VECTOR_LEN);
    let queries = make_queries(512, VECTOR_LEN);
    let a = rt.block_on(build_hnsw(VECTOR_LEN, &input)).unwrap();
    let bld = rt.block_on(build_hnsw(VECTOR_LEN, &input)).unwrap();

    group.bench_function("same_seed_same_results", |b| {
        b.to_async(&rt).iter(|| async {
            let mut same = 0usize;
            for q in &queries {
                let (ka, _) = a
                    .search::<usize>(
                        VectorRef::from_slice_unchecked(q.as_slice()),
                        |_v, _d, info| info_to_usize(info),
                        32,
                        10,
                    )
                    .await
                    .unwrap();
                let (kb, _) = bld
                    .search::<usize>(
                        VectorRef::from_slice_unchecked(q.as_slice()),
                        |_v, _d, info| info_to_usize(info),
                        32,
                        10,
                    )
                    .await
                    .unwrap();
                if ka == kb {
                    same += 1;
                }
            }
            let pct = same as f64 / queries.len() as f64;
            assert!(
                pct >= 0.999_999,
                "expected identical results; got {:.6}",
                pct
            );
            black_box(pct);
        });
    });

    group.finish();
}

// -------------------------------
// BENCHES - LOCAL (FULL)
// -------------------------------
#[cfg(feature = "hnsw-deep-bench")]
fn bench_search_latency_quantiles(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search_quantiles");
    tune_heavy(&mut group);

    let rt = Builder::new_current_thread().enable_all().build().unwrap();

    let input = make_input(INPUT_COUNT, VECTOR_LEN);
    let queries = make_queries(QUERY_COUNT, VECTOR_LEN);
    let hnsw = rt.block_on(build_hnsw(VECTOR_LEN, &input)).unwrap();

    // Report throughput in queries/sec for the batch
    group.throughput(Throughput::Elements(queries.len() as u64));

    let configs: &[(usize, usize)] = &[(16, TOP_N), (32, TOP_N), (64, TOP_N)];
    for &(ef, k) in configs {
        group.bench_function(BenchmarkId::from_parameter(format!("ef={ef}_k={k}")), |b| {
            b.to_async(&rt).iter(|| async {
                let mut per_query_ns: Vec<u128> = Vec::with_capacity(queries.len());
                let mut dist_sum = 0usize;
                let mut hops_sum = 0usize;

                for q in &queries {
                    let t0 = Instant::now();
                    let (_hits, stats) = hnsw
                        .search::<usize>(
                            VectorRef::from_slice_unchecked(black_box(q.as_slice())),
                            |_v, _d, info| info_to_usize(info),
                            ef,
                            k,
                        )
                        .await
                        .unwrap();
                    let dt = t0.elapsed();
                    per_query_ns.push(dt.as_nanos());
                    dist_sum += stats.distances_computed();
                    hops_sum += stats.nhops();
                }

                // compute p50/p90/p99
                per_query_ns.sort_unstable();
                let p =
                    |q: f64| per_query_ns[((per_query_ns.len() as f64 - 1.0) * q).round() as usize];
                let p50 = p(0.50);
                let p90 = p(0.90);
                let p99 = p(0.99);
                let avg_d = dist_sum as f64 / queries.len() as f64;
                let avg_h = hops_sum as f64 / queries.len() as f64;

                black_box((p50, p90, p99, avg_d, avg_h));
            });
        });
    }

    group.finish();
}

#[cfg(feature = "hnsw-deep-bench")]
fn bench_scaling_n(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_scaling_n");
    tune_heavy(&mut group);

    let rt = Builder::new_current_thread().enable_all().build().unwrap();

    // (n, max_level) choose L=0 for single-level; >0 for multi-level
    let cases = &[(1_000usize, 0usize), (10_000, 8), (100_000, 8)];

    for &(n, max_level) in cases {
        group.bench_function(
            BenchmarkId::from_parameter(format!("n={n}_L={max_level}_m=16_efc=64")),
            |b| {
                b.to_async(&rt).iter(|| async {
                    let input = make_input(n, VECTOR_LEN);
                    let opts = HnswBuilderOptions {
                        m: 16,
                        ef_construction: 64,
                        max_level,
                    };
                    let mut hnsw = HnswBuilder::<_, _, InnerProductDistance, _>::new(
                        VECTOR_LEN,
                        StdRng::seed_from_u64(999),
                        opts,
                    );
                    for (v, info) in &input {
                        let _ = hnsw
                            .insert(VectorRef::from_slice_unchecked(v.as_slice()), info.as_ref())
                            .await
                            .unwrap();
                    }
                    // time a small query batch to get scaling feel
                    let queries = make_queries(256, VECTOR_LEN);
                    for q in &queries {
                        let _ = hnsw
                            .search::<usize>(
                                VectorRef::from_slice_unchecked(q.as_slice()),
                                |_v, _d, info| info_to_usize(info),
                                32,
                                10,
                            )
                            .await
                            .unwrap();
                    }
                    black_box(());
                });
            },
        );
    }
    group.finish();
}

#[cfg(feature = "hnsw-deep-bench")]
fn bench_dimensionality(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_dimensionality");
    tune_heavy(&mut group);

    let rt = Builder::new_current_thread().enable_all().build().unwrap();

    for &dim in &[32usize, 128, 384, 768] {
        group.bench_function(BenchmarkId::from_parameter(format!("dim={dim}")), |b| {
            b.to_async(&rt).iter(|| async {
                let input_n = INPUT_COUNT.min(10_000);
                let input = make_input(input_n, dim);
                let hnsw = build_hnsw(dim, &input).await.unwrap();

                let queries = make_queries(1_000, dim);
                for q in &queries {
                    let _ = hnsw
                        .search::<usize>(
                            VectorRef::from_slice_unchecked(q.as_slice()),
                            |_v, _d, info| info_to_usize(info),
                            32,
                            10,
                        )
                        .await
                        .unwrap();
                }
                black_box(());
            });
        });
    }
    group.finish();
}

#[cfg(feature = "hnsw-deep-bench")]
fn bench_graph_shape(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_graph_shape");
    tune_heavy(&mut group);

    let rt = Builder::new_current_thread().enable_all().build().unwrap();

    group.bench_function("shape", |b| {
        b.to_async(&rt).iter(|| async {
            let input = make_input(20_000, VECTOR_LEN);
            let hnsw = build_hnsw(VECTOR_LEN, &input).await.unwrap();

            let shape = hnsw.graph_shape(64).expect("graph not built");
            black_box((
                shape.level_histogram,
                shape.total_edges,
                shape.avg_outdegree,
            ));
        });
    });

    group.finish();
}

#[cfg(feature = "hnsw-deep-bench")]
fn bench_warm_vs_cold(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_warm_vs_cold");
    let rt = Builder::new_current_thread().enable_all().build().unwrap();
    let input = make_input(20_000, VECTOR_LEN);
    let hnsw = rt.block_on(build_hnsw(VECTOR_LEN, &input)).unwrap();

    // Warm: reuse queries
    let warm_q = make_queries(5_000, VECTOR_LEN);
    group.bench_function("warm", |b| {
        b.to_async(&rt).iter(|| async {
            for q in &warm_q {
                let _ = hnsw
                    .search::<usize>(
                        VectorRef::from_slice_unchecked(q.as_slice()),
                        |_v, _d, info| info_to_usize(info),
                        32,
                        10,
                    )
                    .await
                    .unwrap();
            }
            black_box(());
        });
    });

    // Cold: generate new queries inside the measurement
    group.bench_function("cold", |b| {
        b.to_async(&rt).iter(|| async {
            let cold_q = make_queries(5_000, VECTOR_LEN);
            for q in &cold_q {
                let _ = hnsw
                    .search::<usize>(
                        VectorRef::from_slice_unchecked(q.as_slice()),
                        |_v, _d, info| info_to_usize(info),
                        32,
                        10,
                    )
                    .await
                    .unwrap();
            }
            black_box(());
        });
    });

    group.finish();
}

#[cfg(feature = "hnsw-deep-bench")]
fn bench_faiss_parity(c: &mut Criterion) {
    use faiss::{Index, MetricType};

    let mut group = c.benchmark_group("faiss_parity");
    let _rt = Builder::new_current_thread().enable_all().build().unwrap();

    // Prepare dataset and queries once.
    let input = make_input(INPUT_COUNT, VECTOR_LEN);
    let queries = make_queries(QUERY_COUNT, VECTOR_LEN);
    let (_expected_lists, expected_sets) = ground_truth_ids_for_queries(&queries, &input, TOP_N);

    // Build FAISS HNSW (m=40, IP metric), adding vectors individually.
    // This mirrors what the removed test did when it *didn't* rely on the internal contiguous payload.
    let mut faiss_hnsw = faiss::index::hnsw::HnswFlatIndex::new(
        VECTOR_LEN as u32,
        40, // m
        MetricType::InnerProduct,
    )
    .unwrap();

    // Add vectors one by one (VectorVal::as_raw_slice() yields &[f32]).
    for (v, _) in &input {
        faiss_hnsw.add(v.as_raw_slice()).unwrap();
    }

    group.throughput(Throughput::Elements(
        (queries.len() * repeat_query_passes()) as u64,
    ));
    group.bench_function(
        BenchmarkId::from_parameter(format!(
            "n={}_d={}_faiss_k={}",
            INPUT_COUNT, VECTOR_LEN, TOP_N
        )),
        |b| {
            b.iter(|| {
                let mut total_recall = 0.0f64;
                let mut total_queries = 0usize;
                for _pass in 0..repeat_query_passes() {
                    for (i, q) in queries.iter().enumerate() {
                        let res = faiss_hnsw.assign(q.as_raw_slice(), TOP_N).unwrap();
                        let actual: Vec<usize> = res
                            .labels
                            .into_iter()
                            .filter_map(|lbl| lbl.get().map(|j| j as usize))
                            .collect();
                        let r = recall_k_ids(&actual, &expected_sets[i]) as f64;
                        total_recall += r;
                        total_queries += 1;
                    }
                }
                black_box(total_recall / (total_queries as f64))
            });
        },
    );

    group.finish();
}

#[cfg(not(feature = "hnsw-deep-bench"))]
criterion::criterion_group!(
    benches,
    bench_build,
    bench_search_latency_and_recall,
    bench_determinism,
);
#[cfg(feature = "hnsw-deep-bench")]
criterion::criterion_group!(
    benches,
    bench_build,
    bench_search_latency_and_recall,
    bench_determinism,
    bench_search_latency_quantiles,
    bench_scaling_n,
    bench_dimensionality,
    bench_graph_shape,
    bench_warm_vs_cold,
    bench_faiss_parity,
);

criterion_main!(benches);
