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

use std::cmp::min;
use std::marker::PhantomData;

use faiss::index::hnsw::Hnsw;
use rand::Rng;
use rand::distr::uniform::{UniformFloat, UniformSampler};

use crate::hummock::HummockResult;
use crate::vector::utils::{BoundedNearest, MinDistanceHeap};
use crate::vector::{
    MeasureDistance, MeasureDistanceBuilder, OnNearestItem, VectorDistance, VectorItem, VectorRef,
};

#[derive(Copy, Clone)]
pub struct HnswBuilderOptions {
    pub m: usize,
    pub ef_construction: usize,
    pub max_level: usize,
}

impl HnswBuilderOptions {
    fn level_m(&self, level: usize) -> usize {
        // borrowed from pg_vector
        // double the number of connections in ground level
        if level == 0 { 2 * self.m } else { self.m }
    }

    fn m_l(&self) -> f32 {
        assert!(self.m >= 2, "m must be >= 2");
        1.0 / (self.m as f32).ln()
    }
}

fn gen_level(options: &HnswBuilderOptions, rng: &mut impl Rng) -> usize {
    let level = (-UniformFloat::<f32>::sample_single(0.0, 1.0, rng)
        .unwrap()
        .ln()
        * options.m_l())
    .floor() as usize;
    min(level, options.max_level)
}

pub(crate) fn new_node(options: &HnswBuilderOptions, rng: &mut impl Rng) -> VectorHnswNode {
    let level = gen_level(options, rng);
    let mut level_neighbours = Vec::with_capacity(level + 1);
    level_neighbours.extend((0..=level).map(|level| BoundedNearest::new(options.level_m(level))));
    VectorHnswNode { level_neighbours }
}

pub(crate) struct VectorHnswNode {
    level_neighbours: Vec<BoundedNearest<usize>>,
}

impl VectorHnswNode {
    fn level(&self) -> usize {
        self.level_neighbours.len()
    }
}

struct VectorStoreImpl {
    dimension: usize,
    vector_payload: Vec<VectorItem>,
    info_payload: Vec<u8>,
    info_offsets: Vec<usize>,
}

impl VectorStoreImpl {
    fn new(dimension: usize) -> Self {
        Self {
            dimension,
            vector_payload: vec![],
            info_payload: Default::default(),
            info_offsets: vec![],
        }
    }

    fn len(&self) -> usize {
        self.info_offsets.len()
    }

    fn vec_ref(&self, idx: usize) -> VectorRef<'_> {
        assert!(idx < self.info_offsets.len());
        let start = idx * self.dimension;
        let end = start + self.dimension;
        VectorRef::from_slice_unchecked(&self.vector_payload[start..end])
    }

    fn info(&self, idx: usize) -> &[u8] {
        let start = self.info_offsets[idx];
        let end = if idx < self.info_offsets.len() - 1 {
            self.info_offsets[idx + 1]
        } else {
            self.info_payload.len()
        };
        &self.info_payload[start..end]
    }

    fn add(&mut self, vec: VectorRef<'_>, info: &[u8]) {
        assert_eq!(vec.dimension(), self.dimension);

        self.vector_payload.extend_from_slice(vec.as_slice());
        let offset = self.info_payload.len();
        self.info_payload.extend_from_slice(info);
        self.info_offsets.push(offset);
    }
}

pub trait VectorAccessor {
    fn vec_ref(&self) -> VectorRef<'_>;

    fn info(&self) -> &[u8];
}

pub trait VectorStore: 'static {
    type Accessor<'a>: VectorAccessor + 'a
    where
        Self: 'a;
    async fn get_vector(&self, idx: usize) -> HummockResult<Self::Accessor<'_>>;
}

pub struct VectorStoreImplAccessor<'a> {
    vector_store_impl: &'a VectorStoreImpl,
    idx: usize,
}

impl VectorAccessor for VectorStoreImplAccessor<'_> {
    fn vec_ref(&self) -> VectorRef<'_> {
        self.vector_store_impl.vec_ref(self.idx)
    }

    fn info(&self) -> &[u8] {
        self.vector_store_impl.info(self.idx)
    }
}

impl VectorStore for VectorStoreImpl {
    type Accessor<'a> = VectorStoreImplAccessor<'a>;

    async fn get_vector(&self, idx: usize) -> HummockResult<Self::Accessor<'_>> {
        Ok(VectorStoreImplAccessor {
            vector_store_impl: self,
            idx,
        })
    }
}

#[expect(clippy::len_without_is_empty)]
pub trait HnswGraph {
    fn entrypoint(&self) -> usize;
    fn len(&self) -> usize;
    fn node_level(&self, idx: usize) -> usize;
    fn node_neighbours(
        &self,
        idx: usize,
        level: usize,
    ) -> impl Iterator<Item = (usize, VectorDistance)> + '_;
}

pub struct HnswGraphBuilder {
    /// entrypoint of the graph: Some(`entrypoint_vector_idx`)
    entrypoint: usize,
    nodes: Vec<VectorHnswNode>,
}

impl HnswGraphBuilder {
    pub(crate) fn first(node: VectorHnswNode) -> Self {
        Self {
            entrypoint: 0,
            nodes: vec![node],
        }
    }
}

impl HnswGraph for HnswGraphBuilder {
    fn entrypoint(&self) -> usize {
        self.entrypoint
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn node_level(&self, idx: usize) -> usize {
        self.nodes[idx].level()
    }

    fn node_neighbours(
        &self,
        idx: usize,
        level: usize,
    ) -> impl Iterator<Item = (usize, VectorDistance)> + '_ {
        (&self.nodes[idx].level_neighbours[level])
            .into_iter()
            .map(|(distance, &neighbour_index)| (neighbour_index, distance))
    }
}

pub struct HnswBuilder<V: VectorStore, G: HnswGraph, M: MeasureDistanceBuilder, R: Rng> {
    options: HnswBuilderOptions,

    // payload
    vector_store: V,
    graph: Option<G>,

    // utils
    rng: R,
    _measure: PhantomData<M>,
}

#[derive(Default, Debug)]
pub struct HnswStats {
    distances_computed: usize,
    nhops: usize,
}

struct VecSet {
    // TODO: optimize with bitmap
    payload: Vec<bool>,
}

impl VecSet {
    fn new(size: usize) -> Self {
        Self {
            payload: vec![false; size],
        }
    }

    fn set(&mut self, idx: usize) {
        self.payload[idx] = true;
    }

    fn is_set(&self, idx: usize) -> bool {
        self.payload[idx]
    }

    fn reset(&mut self) {
        self.payload.fill(false);
    }
}

impl<M: MeasureDistanceBuilder, R: Rng> HnswBuilder<VectorStoreImpl, HnswGraphBuilder, M, R> {
    pub fn new(dimension: usize, rng: R, options: HnswBuilderOptions) -> Self {
        Self {
            options,
            graph: None,
            vector_store: VectorStoreImpl::new(dimension),
            rng,
            _measure: Default::default(),
        }
    }

    pub fn with_faiss_hnsw(self, faiss_hnsw: Hnsw<'_>) -> Self {
        assert_eq!(self.vector_store.len(), faiss_hnsw.levels_raw().len());
        let (entry_point, _max_level) = faiss_hnsw.entry_point().unwrap();
        let levels = faiss_hnsw.levels_raw();
        let Some(graph) = &self.graph else {
            assert_eq!(levels.len(), 0);
            return Self::new(self.vector_store.dimension, self.rng, self.options);
        };
        assert_eq!(levels.len(), graph.nodes.len());
        let mut nodes = Vec::with_capacity(graph.nodes.len());
        for (node, level_count) in levels.iter().enumerate() {
            let level_count = *level_count as usize;
            let mut level_neighbors = Vec::with_capacity(level_count);
            for level in 0..level_count {
                let neighbors = faiss_hnsw.neighbors_raw(node, level);
                let mut nearest_neighbors = BoundedNearest::new(neighbors.len());
                for &neighbor in neighbors {
                    nearest_neighbors.insert(
                        M::distance(
                            self.vector_store.vec_ref(node),
                            self.vector_store.vec_ref(neighbor as _),
                        ),
                        || neighbor as _,
                    );
                }
                level_neighbors.push(nearest_neighbors);
            }
            nodes.push(VectorHnswNode {
                level_neighbours: level_neighbors,
            });
        }
        Self {
            options: self.options,
            graph: Some(HnswGraphBuilder {
                entrypoint: entry_point,
                nodes,
            }),
            vector_store: self.vector_store,
            rng: self.rng,
            _measure: Default::default(),
        }
    }

    pub fn print_graph(&self) {
        let Some(graph) = &self.graph else {
            println!("empty graph");
            return;
        };
        println!(
            "entrypoint {} in level {}",
            graph.entrypoint,
            graph.nodes[graph.entrypoint].level()
        );
        for (i, node) in graph.nodes.iter().enumerate() {
            println!("node {} has {} levels", i, node.level());
            for level in 0..node.level() {
                print!("level {}: ", level);
                for (_, &neighbor) in &node.level_neighbours[level] {
                    print!("{} ", neighbor);
                }
                println!()
            }
        }
    }

    pub async fn insert(&mut self, vec: VectorRef<'_>, info: &[u8]) -> HummockResult<HnswStats> {
        let node = new_node(&self.options, &mut self.rng);
        let stat = if let Some(graph) = &mut self.graph {
            insert_graph::<M>(
                &self.vector_store,
                graph,
                node,
                vec,
                self.options.ef_construction,
            )
            .await?
        } else {
            self.graph = Some(HnswGraphBuilder::first(node));
            HnswStats::default()
        };
        self.vector_store.add(vec, info);
        Ok(stat)
    }
}

pub(crate) async fn insert_graph<M: MeasureDistanceBuilder>(
    vector_store: &impl VectorStore,
    graph: &mut HnswGraphBuilder,
    mut node: VectorHnswNode,
    vec: VectorRef<'_>,
    ef_construction: usize,
) -> HummockResult<HnswStats> {
    {
        let mut stats = HnswStats::default();
        let entrypoint_index = graph.entrypoint();
        let measure = M::new(vec);
        let mut entrypoints = BoundedNearest::new(1);
        entrypoints.insert(
            measure.measure(vector_store.get_vector(entrypoint_index).await?.vec_ref()),
            || (entrypoint_index, ()),
        );
        let mut visited = VecSet::new(graph.nodes.len());
        let entrypoint_level = graph.nodes[entrypoint_index].level();
        {
            let mut curr_level = entrypoint_level;
            while curr_level > node.level() + 1 {
                curr_level -= 1;
                visited.reset();
                entrypoints = search_layer(
                    vector_store,
                    &*graph,
                    &measure,
                    |_, _, _| (),
                    entrypoints,
                    curr_level,
                    1,
                    &mut stats,
                    &mut visited,
                )
                .await?;
            }
        }
        {
            let mut curr_level = min(entrypoint_level, node.level());
            while curr_level > 0 {
                curr_level -= 1;
                visited.reset();
                entrypoints = search_layer(
                    vector_store,
                    &*graph,
                    &measure,
                    |_, _, _| (),
                    entrypoints,
                    curr_level,
                    ef_construction,
                    &mut stats,
                    &mut visited,
                )
                .await?;
                let level_neighbour = &mut node.level_neighbours[curr_level];
                for (neighbour_distance, &(neighbour_index, _)) in &entrypoints {
                    level_neighbour.insert(neighbour_distance, || neighbour_index);
                }
            }
        }
        let vector_index = graph.nodes.len();
        for (level_index, level) in node.level_neighbours.iter().enumerate() {
            for (neighbour_distance, &neighbour_index) in level {
                graph.nodes[neighbour_index].level_neighbours[level_index]
                    .insert(neighbour_distance, || vector_index);
            }
        }
        if graph.nodes[entrypoint_index].level() < node.level() {
            graph.entrypoint = vector_index;
        }
        graph.nodes.push(node);
        Ok(stats)
    }
}

pub async fn nearest<O: Send, M: MeasureDistanceBuilder>(
    vector_store: &impl VectorStore,
    graph: &impl HnswGraph,
    vec: VectorRef<'_>,
    on_nearest_fn: impl OnNearestItem<O>,
    ef_search: usize,
    top_n: usize,
) -> HummockResult<(Vec<O>, HnswStats)> {
    {
        let entrypoint_index = graph.entrypoint();
        let measure = M::new(vec);
        let mut entrypoints = BoundedNearest::new(1);
        let mut stats = HnswStats::default();
        let entrypoint_vector = vector_store.get_vector(entrypoint_index).await?;
        let entrypoint_distance = measure.measure(entrypoint_vector.vec_ref());
        entrypoints.insert(entrypoint_distance, || {
            (
                entrypoint_index,
                on_nearest_fn(
                    entrypoint_vector.vec_ref(),
                    entrypoint_distance,
                    entrypoint_vector.info(),
                ),
            )
        });
        stats.distances_computed += 1;

        let entrypoint_level = graph.node_level(entrypoint_index);
        let mut visited = VecSet::new(graph.len());
        visited.set(entrypoint_index);
        {
            let mut curr_level = entrypoint_level;
            while curr_level > 1 {
                curr_level -= 1;
                entrypoints = search_layer(
                    vector_store,
                    graph,
                    &measure,
                    &on_nearest_fn,
                    entrypoints,
                    curr_level,
                    1,
                    &mut stats,
                    &mut visited,
                )
                .await?;
            }
        }
        entrypoints = search_layer(
            vector_store,
            graph,
            &measure,
            &on_nearest_fn,
            entrypoints,
            0,
            ef_search,
            &mut stats,
            &mut visited,
        )
        .await?;
        Ok((
            entrypoints.collect_with(|(_, output)| output, Some(top_n)),
            stats,
        ))
    }
}

async fn search_layer<O: Send>(
    vector_store: &impl VectorStore,
    graph: &impl HnswGraph,
    measure: &impl MeasureDistance,
    on_nearest_fn: impl OnNearestItem<O>,
    entrypoints: BoundedNearest<(usize, O)>,
    level_index: usize,
    ef: usize,
    stats: &mut HnswStats,
    visited: &mut VecSet,
) -> HummockResult<BoundedNearest<(usize, O)>> {
    {
        // If ef == 0, there's nothing to explore. Return an empty result set.
        if ef == 0 {
            return Ok(BoundedNearest::new(0));
        }

        let mut candidates = MinDistanceHeap::with_capacity(ef);
        for (distance, &(idx, _)) in &entrypoints {
            visited.set(idx);
            candidates.push(distance, idx);
        }
        let mut nearest = entrypoints;
        nearest.resize(ef);

        while let Some((c_distance, c_index)) = candidates.pop() {
            let (f_distance, _) = nearest.furthest().expect("non-empty");
            if c_distance > f_distance {
                // early break here when even the nearest node in `candidates` is further than the
                // furthest node in the `nearest` set, because no node in `candidates` can be added to `nearest`
                break;
            }
            stats.nhops += 1;
            for (neighbour_index, _) in graph.node_neighbours(c_index, level_index) {
                if visited.is_set(neighbour_index) {
                    continue;
                }
                visited.set(neighbour_index);
                let vector = vector_store.get_vector(neighbour_index).await?;
                let info = vector.info();
                let distance = measure.measure(vector.vec_ref());
                stats.distances_computed += 1;
                let mut added = false;
                let added = &mut added;
                nearest.insert(distance, || {
                    *added = true;
                    (
                        neighbour_index,
                        on_nearest_fn(vector.vec_ref(), distance, info),
                    )
                });
                if *added {
                    candidates.push(distance, neighbour_index);
                }
            }
        }

        Ok(nearest)
    }
}

impl<V, G, M, R> HnswBuilder<V, G, M, R>
where
    V: VectorStore,
    G: HnswGraph,
    M: MeasureDistanceBuilder,
    R: rand::Rng,
{
    pub async fn search<O: Send>(
        &self,
        q: VectorRef<'_>,
        on_nearest: impl OnNearestItem<O>,
        ef_search: usize,
        top_n: usize,
    ) -> HummockResult<(Vec<O>, HnswStats)> {
        let g = self
            .graph
            .as_ref()
            .expect("HNSW graph is empty; insert at least one vector before searching");
        nearest::<O, M>(&self.vector_store, g, q, on_nearest, ef_search, top_n).await
    }
}

#[cfg(test)]
mod tests {
    use faiss::{Index, MetricType};
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use risingwave_common::types::VectorVal;
    use risingwave_common::vector::distance::InnerProductDistance;

    use super::*;
    use crate::vector::test_utils::{gen_info, gen_vector};

    /// Minimal L2 distance for tests using only public traits.
    struct TestL2;

    struct TestL2Measure<'a> {
        q: VectorRef<'a>,
    }

    impl MeasureDistanceBuilder for TestL2 {
        type Measure<'a> = TestL2Measure<'a>;

        fn new<'a>(q: VectorRef<'a>) -> Self::Measure<'a> {
            TestL2Measure { q }
        }

        fn distance(a: VectorRef<'_>, b: VectorRef<'_>) -> VectorDistance {
            // Sum of squared diffs over the public slice API.
            a.as_slice()
                .iter()
                .zip(b.as_slice().iter())
                .map(|(&x, &y)| {
                    // VectorItem <-> f32 conversion (mirrors test_utils usage).
                    let xf: f32 = x.try_into().unwrap();
                    let yf: f32 = y.try_into().unwrap();
                    let d = (xf as f64) - (yf as f64);
                    d * d
                })
                .sum::<f64>()
        }
    }

    impl<'a> MeasureDistance for TestL2Measure<'a> {
        fn measure(&self, v: VectorRef<'_>) -> VectorDistance {
            TestL2::distance(self.q, v)
        }
    }

    fn decode_info_usize(info: &[u8]) -> usize {
        let mut a = [0u8; std::mem::size_of::<usize>()];
        let n = a.len();
        a.copy_from_slice(&info[..n]);
        usize::from_le_bytes(info[..std::mem::size_of::<usize>()].try_into().unwrap())
    }

    fn opts(m: usize, efc: usize, max_level: usize) -> HnswBuilderOptions {
        HnswBuilderOptions {
            m,
            ef_construction: efc,
            max_level,
        }
    }

    async fn build_index(
        dim: usize,
        n: usize,
        seed: u64,
        options: HnswBuilderOptions,
    ) -> HummockResult<(
        HnswBuilder<VectorStoreImpl, HnswGraphBuilder, TestL2, StdRng>,
        Vec<VectorVal>,
    )> {
        let mut hnsw: HnswBuilder<_, _, TestL2, _> =
            HnswBuilder::new(dim, StdRng::seed_from_u64(seed), options);
        let mut vecs = Vec::with_capacity(n);
        for i in 0..n {
            let v = gen_vector(dim);
            // For determinism of dataset, we don't shuffle; insert in order.
            // info is little-endian usize(i) via test_utils.
            if i == 0 {
                // first insert yields default stats
                let stats = hnsw
                    .insert(VectorRef::from_slice_unchecked(v.as_slice()), &gen_info(i))
                    .await?;
                assert_eq!(stats.distances_computed, 0);
                assert_eq!(stats.nhops, 0);
            } else {
                let _ = hnsw
                    .insert(VectorRef::from_slice_unchecked(v.as_slice()), &gen_info(i))
                    .await?;
            }
            vecs.push(v);
        }
        Ok((hnsw, vecs))
    }

    // Helper: brute-force KNN using the inserted vectors we already own.
    fn brute_force_knn(q: VectorRef<'_>, vecs: &[VectorVal], k: usize) -> Vec<usize> {
        let mut s: Vec<(VectorDistance, usize)> = vecs
            .iter()
            .enumerate()
            .map(|(i, v)| {
                (
                    TestL2::distance(q, VectorRef::from_slice_unchecked(v.as_slice())),
                    i,
                )
            })
            .collect();
        s.sort_by(|a, b| a.0.total_cmp(&b.0));
        s.truncate(k);
        s.into_iter().map(|(_, i)| i).collect()
    }

    // Exactness when the graph is dense, single-level.
    #[cfg(not(madsim))]
    #[tokio::test]
    async fn hnsw_search_matches_bruteforce_when_dense_single_level() -> HummockResult<()> {
        let dim = 8;
        let n = 32;
        // Force a dense ground layer by making capacity huge and no upper levels.
        let options = opts(n * 2, n * 2, 0);
        let (hnsw, vecs) = build_index(dim, n, 1234, options).await?;
        let q = gen_vector(dim);
        let k = 10;

        let (hits, _stats) = hnsw
            .search::<usize>(
                VectorRef::from_slice_unchecked(q.as_slice()),
                |_v, _d, info| decode_info_usize(info),
                n,
                k,
            )
            .await?;
        let truth = brute_force_knn(VectorRef::from_slice_unchecked(q.as_slice()), &vecs, k);
        assert_eq!(hits, truth);
        Ok(())
    }

    // With dense single-level graph, ef_search < top_n returns the true best `ef_search`.
    #[cfg(not(madsim))]
    #[tokio::test]
    async fn hnsw_search_best_m_equals_bruteforce_when_ef_lt_topn_dense() -> HummockResult<()> {
        let dim = 8;
        let n = 32;
        let options = opts(n * 2, n * 2, 0);
        let (hnsw, vecs) = build_index(dim, n, 4321, options).await?;
        let q = gen_vector(dim);
        let ef = 5;
        let top_n = 10;

        let (hits, _stats) = hnsw
            .search::<usize>(
                VectorRef::from_slice_unchecked(q.as_slice()),
                |_v, _d, info| decode_info_usize(info),
                ef,
                top_n,
            )
            .await?;
        let truth = brute_force_knn(VectorRef::from_slice_unchecked(q.as_slice()), &vecs, ef);
        assert_eq!(hits, truth);
        Ok(())
    }

    // Multi-level navigation still finds the true top-1.
    #[cfg(not(madsim))]
    #[tokio::test]
    async fn hnsw_search_top1_equals_bruteforce_multi_level() -> HummockResult<()> {
        let dim = 8;
        let n = 64;
        // Allow multiple levels; big ef to make search thorough.
        let options = opts(8, 64, 8);
        let (hnsw, vecs) = build_index(dim, n, 2468, options).await?;
        let q = gen_vector(dim);

        let (out, _stats) = hnsw
            .search::<usize>(
                VectorRef::from_slice_unchecked(q.as_slice()),
                |_v, _d, info| decode_info_usize(info),
                64,
                1,
            )
            .await?;
        let truth = brute_force_knn(VectorRef::from_slice_unchecked(q.as_slice()), &vecs, 1);
        assert_eq!(out, truth);
        Ok(())
    }

    #[cfg(not(madsim))]
    #[tokio::test]
    async fn hnsw_search_singleton_returns_only_item() -> HummockResult<()> {
        let dim = 8;
        let (hnsw, vecs) = build_index(dim, 1, 7, opts(8, 16, 8)).await?;
        let q = VectorRef::from_slice_unchecked(vecs[0].as_slice());
        let (out, stats) = hnsw
            .search::<usize>(q, |_v, _d, info| decode_info_usize(info), 8, 5)
            .await?;
        assert_eq!(out, vec![0]);
        assert_eq!(stats.distances_computed, 1);
        // search_layer increments nhops once per popped candidate, even if no neighbors exist.
        assert_eq!(stats.nhops, 1);
        Ok(())
    }

    #[cfg(not(madsim))]
    #[tokio::test]
    async fn hnsw_search_topn_zero_is_empty() -> HummockResult<()> {
        let dim = 8;
        let (hnsw, vecs) = build_index(dim, 10, 42, opts(8, 32, 8)).await?;
        let q = VectorRef::from_slice_unchecked(vecs[0].as_slice());
        let (out, _stats) = hnsw
            .search::<usize>(q, |_v, _d, info| decode_info_usize(info), 10, 0)
            .await?;
        assert!(out.is_empty());
        Ok(())
    }

    #[cfg(not(madsim))]
    #[tokio::test]
    async fn hnsw_search_len_is_min_of_topn_and_ef() -> HummockResult<()> {
        let dim = 8;
        let n = 32;
        let (hnsw, vecs) = build_index(dim, n, 11, opts(8, 32, 8)).await?;
        let q = VectorRef::from_slice_unchecked(vecs[0].as_slice());
        let ef = 3usize;
        let top_n = 10usize;
        let (out, _stats) = hnsw
            .search::<usize>(q, |_v, _d, info| decode_info_usize(info), ef, top_n)
            .await?;
        assert_eq!(out.len(), ef);
        Ok(())
    }

    #[cfg(not(madsim))]
    #[tokio::test]
    async fn hnsw_search_topn_over_dataset_caps_at_dataset_size() -> HummockResult<()> {
        let dim = 6;
        let n = 12;
        let (hnsw, vecs) = build_index(dim, n, 99, opts(8, 32, 8)).await?;
        let q = VectorRef::from_slice_unchecked(vecs[0].as_slice());
        // ef large enough; request more than dataset.
        let (out, _stats) = hnsw
            .search::<usize>(q, |_v, _d, info| decode_info_usize(info), n * 2, n * 5)
            .await?;
        assert_eq!(out.len(), n);
        Ok(())
    }

    #[cfg(not(madsim))]
    #[tokio::test]
    async fn hnsw_search_forwards_info_payload() -> HummockResult<()> {
        let dim = 5;
        let n = 20;
        let (hnsw, vecs) = build_index(dim, n, 2025, opts(8, 64, 8)).await?;
        // Query exactly the ith vector to make its info the obvious top-1.
        let i = 7usize;
        let q = VectorRef::from_slice_unchecked(vecs[i].as_slice());
        let (out, _stats) = hnsw
            .search::<usize>(q, |_v, _d, info| decode_info_usize(info), 32, 1)
            .await?;
        assert_eq!(out, vec![i]);
        Ok(())
    }

    #[cfg(not(madsim))]
    #[tokio::test]
    async fn hnsw_search_increasing_ef_does_not_worsen_best_distance() -> HummockResult<()> {
        let dim = 8;
        let n = 50;
        let (hnsw, _vecs) = build_index(dim, n, 31415, opts(8, 64, 8)).await?;
        let q = gen_vector(dim); // query not in the index
        // Return the best distance itself.
        let (small, _s_stats) = hnsw
            .search::<f64>(
                VectorRef::from_slice_unchecked(q.as_slice()),
                |_v, d, _info| d,
                4,
                1,
            )
            .await?;
        let (large, _l_stats) = hnsw
            .search::<f64>(
                VectorRef::from_slice_unchecked(q.as_slice()),
                |_v, d, _info| d,
                64,
                1,
            )
            .await?;
        assert!(large[0] <= small[0] + 1e-6);
        Ok(())
    }

    #[cfg(not(madsim))]
    #[tokio::test]
    async fn hnsw_search_results_are_sorted_by_distance_non_decreasing() -> HummockResult<()> {
        let dim = 8;
        let n = 40;
        let (hnsw, _vecs) = build_index(dim, n, 2718, opts(8, 64, 8)).await?;
        let q = gen_vector(dim);
        let (dists, _stats) = hnsw
            .search::<f64>(
                VectorRef::from_slice_unchecked(q.as_slice()),
                |_v, d, _info| d,
                n,
                n,
            )
            .await?;
        // verify non-decreasing distances
        for w in dists.windows(2) {
            assert!(w[0] <= w[1] + 1e-6);
        }
        Ok(())
    }

    #[cfg(not(madsim))]
    #[tokio::test]
    async fn hnsw_search_handles_ties_returns_both_zero_distance_hits() -> HummockResult<()> {
        let dim = 6;
        // Build with two identical vectors first to maximize chance they're both in the final top.
        let options = opts(8, 64, 8);
        let mut hnsw: HnswBuilder<_, _, TestL2, _> =
            HnswBuilder::new(dim, StdRng::seed_from_u64(123), options);

        // identical base vector
        let v0 = gen_vector(dim);
        let v1 = v0.clone();
        let v2 = gen_vector(dim);

        let _ = hnsw
            .insert(VectorRef::from_slice_unchecked(v0.as_slice()), &gen_info(0))
            .await?;
        let _ = hnsw
            .insert(VectorRef::from_slice_unchecked(v1.as_slice()), &gen_info(1))
            .await?;
        let _ = hnsw
            .insert(VectorRef::from_slice_unchecked(v2.as_slice()), &gen_info(2))
            .await?;

        // Query exactly the shared vector.
        let (dists, _stats) = hnsw
            .search::<f64>(
                VectorRef::from_slice_unchecked(v0.as_slice()),
                |_v, d, _info| d,
                16,
                2,
            )
            .await?;
        assert_eq!(dists.len(), 2);
        assert!((dists[0] - 0.0).abs() < 1e-8);
        assert!((dists[1] - 0.0).abs() < 1e-8);
        Ok(())
    }

    #[cfg(not(madsim))]
    #[tokio::test]
    async fn hnsw_search_is_deterministic_with_seeded_build() -> HummockResult<()> {
        let dim = 8;
        let n = 30;
        let seed = 7777;
        let options = opts(8, 64, 8);

        let (hnsw_a, vecs_a) = build_index(dim, n, seed, options).await?;
        let q = gen_vector(dim);

        let (out_a, _sa) = hnsw_a
            .search::<usize>(
                VectorRef::from_slice_unchecked(q.as_slice()),
                |_v, _d, info| decode_info_usize(info),
                32,
                10,
            )
            .await?;

        // Rebuild with the same seed and identical data order.
        let mut hnsw_b: HnswBuilder<VectorStoreImpl, HnswGraphBuilder, TestL2, StdRng> =
            HnswBuilder::new(dim, StdRng::seed_from_u64(seed), options);
        for i in 0..n {
            let _ = hnsw_b
                .insert(
                    VectorRef::from_slice_unchecked(vecs_a[i].as_slice()),
                    &gen_info(i),
                )
                .await?;
        }
        let (out_b, _sb) = hnsw_b
            .search::<usize>(
                VectorRef::from_slice_unchecked(q.as_slice()),
                |_v, _d, info| decode_info_usize(info),
                32,
                10,
            )
            .await?;

        assert_eq!(out_a, out_b);
        Ok(())
    }

    #[cfg(not(madsim))]
    #[tokio::test]
    async fn hnsw_search_stats_are_sane() -> HummockResult<()> {
        let dim = 8;
        let n = 50;
        let (hnsw, _vecs) = build_index(dim, n, 13579, opts(8, 64, 8)).await?;
        let q = gen_vector(dim);
        let (out, stats) = hnsw
            .search::<usize>(
                VectorRef::from_slice_unchecked(q.as_slice()),
                |_v, _d, info| decode_info_usize(info),
                20,
                10,
            )
            .await?;
        assert!(stats.distances_computed >= out.len());
        assert!(stats.distances_computed <= n);
        // Very loose sanity bound; behavior-focused (not internals).
        assert!(stats.nhops <= n * 2);
        Ok(())
    }

    #[cfg(not(madsim))]
    #[tokio::test]
    async fn hnsw_search_ef_zero_returns_empty() -> HummockResult<()> {
        let (hnsw, vecs) = build_index(8, 10, 42, opts(8, 32, 8)).await?;
        let q = VectorRef::from_slice_unchecked(vecs[0].as_slice());
        let (out, _stats) = hnsw
            .search::<usize>(q, |_v, _d, info| decode_info_usize(info), 0, 5)
            .await?;
        assert!(out.is_empty());
        Ok(())
    }

    #[cfg(not(madsim))]
    #[tokio::test]
    #[should_panic(expected = "HNSW graph is empty")]
    async fn hnsw_search_panics_on_empty_graph() {
        let hnsw: HnswBuilder<_, _, TestL2, _> =
            HnswBuilder::new(8, StdRng::seed_from_u64(1), opts(8, 32, 8));
        let q = gen_vector(8);
        let _ = hnsw
            .search::<usize>(
                VectorRef::from_slice_unchecked(q.as_slice()),
                |_v, _d, info| decode_info_usize(info),
                8,
                1,
            )
            .await;
    }

    #[cfg(not(madsim))]
    #[tokio::test]
    async fn hnsw_faiss_interop_smoke() {
        const DIM: usize = 16;
        const N: usize = 200;
        const M: usize = 8;

        // Build our HNSW
        let opts = HnswBuilderOptions {
            m: M,
            ef_construction: 32,
            max_level: 4,
        };
        let mut ours: HnswBuilder<VectorStoreImpl, HnswGraphBuilder, InnerProductDistance, _> =
            HnswBuilder::new(DIM, StdRng::seed_from_u64(123), opts);

        let mut vecs = Vec::new();
        for i in 0..N {
            let v = gen_vector(DIM);
            ours.insert(VectorRef::from_slice_unchecked(v.as_slice()), &gen_info(i))
                .await
                .unwrap();
            vecs.push(v);
        }

        // Build FAISS HNSW on the same data
        let mut faiss_hnsw =
            faiss::index::hnsw::HnswFlatIndex::new(DIM as u32, M as u32, MetricType::InnerProduct)
                .unwrap();
        for v in &vecs {
            faiss_hnsw.add(v.as_raw_slice()).unwrap();
        }

        // Quick parity: query a few random vectors and compare top-1 id equality
        let q = &vecs[7];
        let (hits, _stats) = ours
            .search::<usize>(
                VectorRef::from_slice_unchecked(q.as_slice()),
                |_v, _d, info| decode_info_usize(info),
                16,
                1,
            )
            .await
            .unwrap();

        let fa = faiss_hnsw.assign(q.as_raw_slice(), 1).unwrap();
        let fa_top1 = fa.labels[0].get().unwrap_or_default() as usize;

        // We expect both to return the same self-id for exact match queries
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0], fa_top1);
    }
}
