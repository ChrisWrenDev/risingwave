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
    MeasureDistance, MeasureDistanceBuilder, OnNearestItem, VectorDistance, VectorInner,
    VectorItem, VectorRef,
};

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
    let mut level_neighbours = Vec::with_capacity(level);
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
    vector_len: usize,
    vector_payload: Vec<VectorItem>,
    info_payload: Vec<u8>,
    info_offsets: Vec<usize>,
}

impl VectorStoreImpl {
    fn new(vector_len: usize) -> Self {
        Self {
            vector_len,
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
        let start = idx * self.vector_len;
        let end = start + self.vector_len;
        VectorInner(&self.vector_payload[start..end])
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
        assert_eq!(vec.0.len(), self.vector_len);

        self.vector_payload.extend_from_slice(vec.0);
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
    pub fn new(vector_len: usize, rng: R, options: HnswBuilderOptions) -> Self {
        Self {
            options,
            graph: None,
            vector_store: VectorStoreImpl::new(vector_len),
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
            return Self::new(self.vector_store.vector_len, self.rng, self.options);
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
        visited.reset();
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

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use bytes::Bytes;
    use itertools::Itertools;
    use rand::SeedableRng;
    use rand::prelude::StdRng;

    use crate::vector::NearestBuilder;
    use crate::vector::MeasureDistanceBuilder;
    use crate::vector::distance::InnerProductDistance;
    use crate::vector::hnsw::{HnswBuilder, HnswBuilderOptions, HnswGraph, nearest};
    use crate::vector::test_utils::{gen_info, gen_vector};
    // Access internal types from the parent module
    use super::{BoundedNearest, VectorHnswNode};

    pub const SEED: u64 = 233;

// Core correctness vs a brute-force baseline
// Small-set search ≈ exact NN (robust threshold)
#[tokio::test]
async fn hnsw_small_matches_bruteforce() {
    const VECTOR_LEN: usize = 16;
    const INPUT_COUNT: usize = 200;
    const QUERY_COUNT: usize = 20;
    const TOP_N: usize = 5;

    // dataset
    let mut data = Vec::with_capacity(INPUT_COUNT);
    for i in 0..INPUT_COUNT {
        data.push((gen_vector(VECTOR_LEN), gen_info(i)));
    }

    // build HNSW (slightly larger ef_construction for stability)
    let mut hnsw = HnswBuilder::<_, _, InnerProductDistance, _>::new(
        VECTOR_LEN,
        StdRng::seed_from_u64(SEED),
        HnswBuilderOptions { m: 8, ef_construction: 32, max_level: 5 },
    );
    for (v, info) in &data {
        hnsw.insert(v.to_ref(), info).await.unwrap();
    }

    // queries
    let mut queries = Vec::with_capacity(QUERY_COUNT);
    for _ in 0..QUERY_COUNT {
        queries.push(gen_vector(VECTOR_LEN));
    }

    for q in &queries {
        // exact baseline
        let mut nb = NearestBuilder::<'_, _, InnerProductDistance>::new(q.to_ref(), TOP_N);
        nb.add(
            data.iter().map(|(v, info)| (v.to_ref(), info.as_ref())),
            |_, _, info| Bytes::copy_from_slice(info),
        );
        let expected = nb.finish();
        let expected_set: HashSet<_> =
            expected.iter().map(|b| b.as_ref().to_vec()).collect();

        // ANN with a bit higher ef_search for better recall but still fast
        let (actual, stats) = nearest::<_, InnerProductDistance>(
            &hnsw.vector_store,
            hnsw.graph.as_ref().expect("graph built"),
            q.to_ref(),
            |_, _, info| Bytes::copy_from_slice(info),
            /* ef_search */ 32,
            TOP_N,
        )
        .await
        .unwrap();

        assert!(stats.distances_computed > 0, "distances_computed should be > 0");
        assert!(stats.nhops > 0, "nhops should be > 0");

        let actual_set: HashSet<_> =
            actual.iter().map(|b| b.as_ref().to_vec()).collect();
        let inter = expected_set.intersection(&actual_set).count();

        // Allow up to 2 misses to avoid brittleness on random data
        assert!(
            inter >= TOP_N.saturating_sub(2),
            "low recall: {inter}/{TOP_N}"
        );
    }
}

// Deterministic answers with fixed seed
// Same seed ⇒ same graph & results
#[tokio::test]
async fn hnsw_deterministic_with_fixed_seed() {
    const VECTOR_LEN: usize = 16;
    const INPUT_COUNT: usize = 200;
    const QUERY_COUNT: usize = 20;
    const TOP_N: usize = 5;

    // fixed dataset & queries
    let input = (0..INPUT_COUNT)
        .map(|i| (gen_vector(VECTOR_LEN), gen_info(i)))
        .collect_vec();
    let queries = (0..QUERY_COUNT)
        .map(|_| gen_vector(VECTOR_LEN))
        .collect_vec();

    // build #1
    let mut h1 = HnswBuilder::<_, _, InnerProductDistance, _>::new(
        VECTOR_LEN,
        StdRng::seed_from_u64(SEED),
        HnswBuilderOptions {
            m: 8,
            ef_construction: 16,
            max_level: 5,
        },
    );
    for (v, info) in &input {
        h1.insert(v.to_ref(), info).await.unwrap();
    }

    // build #2 with the same seed and data
    let mut h2 = HnswBuilder::<_, _, InnerProductDistance, _>::new(
        VECTOR_LEN,
        StdRng::seed_from_u64(SEED),
        HnswBuilderOptions {
            m: 8,
            ef_construction: 16,
            max_level: 5,
        },
    );
    for (v, info) in &input {
        h2.insert(v.to_ref(), info).await.unwrap();
    }

    // for each query, the returned info bytes must be identical (same order & contents)
    for q in &queries {
        let (a1, _s1) = nearest::<_, InnerProductDistance>(
            &h1.vector_store,
            h1.graph.as_ref().expect("graph built"),
            q.to_ref(),
            |_, _, info| Bytes::copy_from_slice(info),
            16,
            TOP_N,
        )
        .await
        .unwrap();

        let (a2, _s2) = nearest::<_, InnerProductDistance>(
            &h2.vector_store,
            h2.graph.as_ref().expect("graph built"),
            q.to_ref(),
            |_, _, info| Bytes::copy_from_slice(info),
            16,
            TOP_N,
        )
        .await
        .unwrap();

        assert_eq!(
            a1, a2,
            "non-deterministic results for query; a1={:?}, a2={:?}",
            a1.iter().map(|b| b.as_ref()).collect_vec(),
            a2.iter().map(|b| b.as_ref()).collect_vec()
        );
    }
}

// Graph construction invariants — entrypoint is at the highest level
#[tokio::test]
async fn hnsw_entrypoint_is_highest_level() {

    const VECTOR_LEN: usize = 16;
    const INPUT_COUNT: usize = 200;

    let mut hnsw = HnswBuilder::<_, _, InnerProductDistance, _>::new(
        VECTOR_LEN,
        StdRng::seed_from_u64(SEED),
        HnswBuilderOptions { m: 8, ef_construction: 16, max_level: 6 },
    );

    for i in 0..INPUT_COUNT {
        let v = gen_vector(VECTOR_LEN);
        let info = gen_info(i);
        hnsw.insert(v.to_ref(), &info).await.unwrap();
    }

    let g = hnsw.graph.as_ref().expect("graph built");
    let top = (0..g.len()).map(|i| g.node_level(i)).max().unwrap();
    assert_eq!(g.node_level(g.entrypoint()), top, "entrypoint not at top level");
}

// Graph construction invariants — degree bounds per level are respected
#[tokio::test]
async fn hnsw_degree_bounds_per_level() {
    const VECTOR_LEN: usize = 16;
    const INPUT_COUNT: usize = 200;
    const M: usize = 8;

    let mut hnsw = HnswBuilder::<_, _, InnerProductDistance, _>::new(
        VECTOR_LEN,
        StdRng::seed_from_u64(SEED),
        HnswBuilderOptions { m: M, ef_construction: 16, max_level: 6 },
    );

    for i in 0..INPUT_COUNT {
        let v = gen_vector(VECTOR_LEN);
        let info = gen_info(i);
        hnsw.insert(v.to_ref(), &info).await.unwrap();
    }

    let g = hnsw.graph.as_ref().expect("graph built");

    for node_idx in 0..g.len() {
        let levels = g.node_level(node_idx);
        for level in 0..levels {
            // ground level: up to 2*m, upper levels: up to m
            let max_deg = if level == 0 { 2 * M } else { M };
            let deg = g.node_neighbours(node_idx, level).count();
            assert!(
                deg <= max_deg,
                "node {node_idx} level {level} has degree {deg} > {max_deg}"
            );
        }
    }
}

// Graph construction invariants — back-link admissibility at level 0
//
// This test assumes the ordering used by our MeasureDistance implementation:
// "smaller distance is better / closer". For the current InnerProductDistance,
// this is true because it returns a *distance* (e.g., -dot or another
// monotone transform) where lower values mean closer.
//
// If the metric is ever changed to return a similarity where "higher is better",
// or if the distance is redefined so that larger numbers are better, then the
// inequality checks in this test must be inverted accordingly.
//
// In particular, when v omits a backlink to u at level 0, we require that
// u is not strictly better than v's worst admitted neighbor; i.e.:
//   d(u, v) >= worst_neighbor_distance(v)
// under the "lower is better" convention.
#[tokio::test]
async fn hnsw_backlink_admissibility_level0() {
    const VECTOR_LEN: usize = 16;
    const N: usize = 24;
    const M: usize = 8; // m used to compute capacity (2*m at level 0)

    let mut hnsw = HnswBuilder::<_, _, InnerProductDistance, _>::new(
        VECTOR_LEN,
        StdRng::seed_from_u64(SEED),
        HnswBuilderOptions { m: M, ef_construction: 16, max_level: 6 },
    );

    for i in 0..N {
        let v = gen_vector(VECTOR_LEN);
        let info = gen_info(i);
        hnsw.insert(v.to_ref(), &info).await.unwrap();

        let g = hnsw.graph.as_ref().expect("graph built");
        let u = g.len() - 1; // newly inserted node

        // All level-0 neighbors of u
        let u_neigh0: HashSet<_> = g.node_neighbours(u, 0).map(|(idx, _)| idx).collect();

        for &v_idx in &u_neigh0 {
            // Does v have u back?
            let v_has_u = g.node_neighbours(v_idx, 0).any(|(idx, _)| idx == u);
            if v_has_u {
                continue; // symmetric link present; good
            }

            // Otherwise, admissibility check:
            // 1) v's level-0 neighbor set must be at capacity
            let cap = 2 * M; // ground-level capacity
            let v_deg = g.node_neighbours(v_idx, 0).count();
            assert!(
                v_deg == cap,
                "v({v_idx}) missing backlink to u({u}) but degree {v_deg} < cap {cap}"
            );

            // 2) d(u,v) must be >= worst distance currently in v's N0
            let d_uv = InnerProductDistance::distance(
                hnsw.vector_store.vec_ref(u),
                hnsw.vector_store.vec_ref(v_idx),
            );

            // "Worst" here means the furthest neighbor under the current distance order.
            // Because smaller is better, the worst is the MAX distance among v's neighbors.
            let mut worst = f32::NEG_INFINITY;
            for (_w_idx, d_vw) in g.node_neighbours(v_idx, 0) {
                if d_vw > worst {
                    worst = d_vw;
                }
            }

            // Admissibility: if v didn't admit u, then u must not be strictly better than v's worst admitted neighbor. 
            // With "lower is better": d_uv >= worst (+ tiny eps to avoid float noise).
            let eps = 1e-6;
            assert!(
                d_uv + eps >= worst,
                "v({v_idx}) missing backlink to u({u}) but d(u,v)={d_uv} < worst={worst} among v's neighbors"
            );
        }
    }
}

// Graph construction invariants — entrypoint updates when a higher-level node appears (public API, robust)
#[tokio::test]
async fn hnsw_entrypoint_updates_on_higher_level_insert() {
    const D: usize = 16;
    const M: usize = 2;          // small m => higher levels are relatively common
    const MAX_LEVEL: usize = 10; // give room so we can see several increases without capping out
    const INSERTS: usize = 200;  // small and fast; plenty to see ≥1 level increase with M=2

    let mut hnsw = HnswBuilder::<_, _, InnerProductDistance, _>::new(
        D,
        StdRng::seed_from_u64(SEED),
        HnswBuilderOptions {
            m: M,
            ef_construction: 16,
            max_level: MAX_LEVEL,
        },
    );

    // Track the highest level we've observed so far.
    let mut top_seen: Option<usize> = None;
    let mut increases_observed = 0usize;

    for i in 0..INSERTS {
        let v = gen_vector(D);
        let info = gen_info(1_000_000usize + i);
        hnsw.insert(v.to_ref(), &info).await.unwrap();

        let g = hnsw.graph.as_ref().expect("graph built");
        let new_top = (0..g.len()).map(|idx| g.node_level(idx)).max().unwrap();
        match top_seen {
            None => {
                // First insertion establishes the initial top
                top_seen = Some(new_top);
            }
            Some(prev_top) if new_top > prev_top => {
                // Top level increased; the last inserted node must own that level
                let last = g.len() - 1;
                assert_eq!(
                    g.node_level(last),
                    new_top,
                    "new top level {new_top} not owned by last inserted node {last}"
                );
                assert_eq!(
                    g.entrypoint(),
                    last,
                    "entrypoint was not updated to the highest-level newly inserted node"
                );
                top_seen = Some(new_top);
                increases_observed += 1;
            }
            _ => {
                // No increase this round; nothing to assert.
            }
        }
    }

    assert!(
        increases_observed > 0,
        "no top-level increases observed over {INSERTS} inserts (M={M}, MAX_LEVEL={MAX_LEVEL}); \
         consider increasing INSERTS if this ever flakes"
    );
}

// Search-procedure behaviour — ef_search bounds work is respected
#[tokio::test]
async fn hnsw_ef_search_bounds_work() {
    const VECTOR_LEN: usize = 16;
    const INPUT_COUNT: usize = 500;
    const QUERY_COUNT: usize = 20;
    const TOP_N: usize = 5;

    // dataset
    let input = (0..INPUT_COUNT)
        .map(|i| (gen_vector(VECTOR_LEN), gen_info(i)))
        .collect_vec();

    // build HNSW
    let mut hnsw = HnswBuilder::<_, _, InnerProductDistance, _>::new(
        VECTOR_LEN,
        StdRng::seed_from_u64(SEED),
        HnswBuilderOptions { m: 8, ef_construction: 16, max_level: 6 },
    );
    for (v, info) in &input {
        hnsw.insert(v.to_ref(), info).await.unwrap();
    }
    let g = hnsw.graph.as_ref().expect("graph built");

    // queries
    let queries = (0..QUERY_COUNT).map(|_| gen_vector(VECTOR_LEN)).collect_vec();

    // Compare ef_search=8 vs ef_search=32
    let mut total_dist_8 = 0usize;
    let mut total_dist_32 = 0usize;
    let mut total_hops_8 = 0usize;
    let mut total_hops_32 = 0usize;

    // Optional: also check recall monotonicity against brute force
    let mut recall_ok = true;

    for q in &queries {
        // exact top-K
        let mut nb = NearestBuilder::<'_, _, InnerProductDistance>::new(q.to_ref(), TOP_N);
        nb.add(
            input.iter().map(|(v, info)| (v.to_ref(), info.as_ref())),
            |_, _, info| Bytes::copy_from_slice(info),
        );
        let expected = nb.finish();
        let expected_set: HashSet<_> =
            expected.iter().map(|b| b.as_ref().to_vec()).collect();

        // ef=8
        let (a8, s8) = nearest::<_, InnerProductDistance>(
            &hnsw.vector_store,
            g,
            q.to_ref(),
            |_, _, info| Bytes::copy_from_slice(info),
            8,
            TOP_N,
        )
        .await
        .unwrap();

        // ef=32
        let (a32, s32) = nearest::<_, InnerProductDistance>(
            &hnsw.vector_store,
            g,
            q.to_ref(),
            |_, _, info| Bytes::copy_from_slice(info),
            32,
            TOP_N,
        )
        .await
        .unwrap();

        total_dist_8 += s8.distances_computed;
        total_dist_32 += s32.distances_computed;
        total_hops_8 += s8.nhops;
        total_hops_32 += s32.nhops;

        // recall monotonicity: ef=8 should not beat ef=32
        let a8_set: HashSet<_> =
            a8.iter().map(|b| b.as_ref().to_vec()).collect();
        let a32_set: HashSet<_> =
            a32.iter().map(|b| b.as_ref().to_vec()).collect();
        let r8 = expected_set.intersection(&a8_set).count();
        let r32 = expected_set.intersection(&a32_set).count();
        if r8 > r32 {
            recall_ok = false;
        }
    }

    // With smaller ef_search we expect *less or equal* work
    assert!(
        total_dist_8 <= total_dist_32,
        "distances_computed should not increase when lowering ef_search: {} > {}",
        total_dist_8,
        total_dist_32
    );
    assert!(
        total_hops_8 <= total_hops_32,
        "nhops should not increase when lowering ef_search: {} > {}",
        total_hops_8,
        total_hops_32
    );
    assert!(recall_ok, "ef=8 unexpectedly had higher recall than ef=32 for some queries");
}

// Search-procedure behaviour — early-break pruning is exercised (robust variant)
#[tokio::test]
async fn hnsw_early_break_pruning_triggers() {
    const D: usize = 16;
    const INPUT_COUNT: usize = 200;
    const TOP_N: usize = 5;

    let mut hnsw = HnswBuilder::<_, _, InnerProductDistance, _>::new(
        D,
        StdRng::seed_from_u64(SEED),
        HnswBuilderOptions { m: 8, ef_construction: 16, max_level: 6 },
    );

    for i in 0..INPUT_COUNT {
        let v = gen_vector(D);
        let info = gen_info(i);
        hnsw.insert(v.to_ref(), &info).await.unwrap();
    }

    let g = hnsw.graph.as_ref().unwrap();

    // Query with the entrypoint vector
    let ep_idx = g.entrypoint();
    let ep_vec = hnsw.vector_store.vec_ref(ep_idx);
    let (_a_ep, s_ep) = nearest::<_, InnerProductDistance>(
        &hnsw.vector_store,
        g,
        ep_vec,
        |_, _, info| Bytes::copy_from_slice(info),
        16,
        TOP_N,
    )
    .await
    .unwrap();

    // Try several random queries and expect at least one to explore more (strictly more hops)
    let mut any_strictly_more = false;
    for _ in 0..10 {
        let q = gen_vector(D);
        let (_a_rand, s_rand) = nearest::<_, InnerProductDistance>(
            &hnsw.vector_store,
            g,
            q.to_ref(),
            |_, _, _info| Bytes::new(),
            16,
            TOP_N,
        )
        .await
        .unwrap();

        if s_rand.nhops > s_ep.nhops {
            any_strictly_more = true;
            break;
        }
    }

    assert!(
        any_strictly_more,
        "expected at least one random query to require strictly more hops than entrypoint query"
    );
}

// Search-procedure behaviour — visited set prevents revisits (no node expanded twice per layer)
#[tokio::test]
async fn hnsw_visited_set_prevents_revisits() {
    const VECTOR_LEN: usize = 16;
    const N: usize = 5;    // tiny graph
    const TOP_N: usize = 3;

    // Dense-ish level-0 with small N and m=8
    let mut hnsw = HnswBuilder::<_, _, InnerProductDistance, _>::new(
        VECTOR_LEN,
        StdRng::seed_from_u64(SEED),
        HnswBuilderOptions { m: 8, ef_construction: 16, max_level: 4 },
    );
    for i in 0..N {
        let v = gen_vector(VECTOR_LEN);
        let info = gen_info(i);
        hnsw.insert(v.to_ref(), &info).await.unwrap();
    }
    let g = hnsw.graph.as_ref().unwrap();

    // Run with relatively large ef_search to traverse more
    let q = gen_vector(VECTOR_LEN);
    let (_a_large, s_large) = nearest::<_, InnerProductDistance>(
        &hnsw.vector_store,
        g,
        q.to_ref(),
        |_, _, _info| (),
        16,
        TOP_N,
    )
    .await
    .unwrap();

    // Safe upper bound: visited resets per layer; nhops should be ≤ nodes × (#levels visited)
    // (#levels visited) ≤ max_node_level + 1 (levels are 0..=max)
    let max_level = (0..g.len()).map(|i| g.node_level(i)).max().unwrap_or(0);
    let upper_bound = g.len() * (max_level + 1);
    assert!(
        s_large.nhops <= upper_bound,
        "nhops {} exceeded conservative bound {}; visited set may not be preventing revisits",
        s_large.nhops,
        upper_bound
    );

    // Now lower ef_search; work (nhops) should not *increase* anomalously
    let (_a_small, s_small) = nearest::<_, InnerProductDistance>(
        &hnsw.vector_store,
        g,
        q.to_ref(),
        |_, _, _info| (),
        4,
        TOP_N,
    )
    .await
    .unwrap();

    assert!(
        s_small.nhops <= s_large.nhops,
        "lowering ef_search increased nhops: {} -> {}",
        s_large.nhops,
        s_small.nhops
    );
}

// Edge cases — single item and empty graph behaviour (+ top_n truncation on size=1)
#[tokio::test]
async fn hnsw_single_item_behaviour() {
    const D: usize = 16;

    // Start with an empty builder (graph created on first insert)
    let mut hnsw = HnswBuilder::<_, _, InnerProductDistance, _>::new(
        D,
        StdRng::seed_from_u64(SEED),
        HnswBuilderOptions { m: 8, ef_construction: 16, max_level: 4 },
    );

    // Insert exactly one vector
    let v = gen_vector(D);
    let info = gen_info(42);
    hnsw.insert(v.to_ref(), &info).await.unwrap();

    // Query with the same vector; ask for many results (> size)
    let (ans, stats) = nearest::<_, InnerProductDistance>(
        &hnsw.vector_store,
        hnsw.graph.as_ref().unwrap(),
        v.to_ref(),
        |_, _, info| Bytes::copy_from_slice(info),
        /* ef_search */ 16,
        /* top_n */ 10,
    )
    .await
    .unwrap();

    // Should return exactly the one item we inserted, no panic
    assert_eq!(ans.len(), 1, "top_n>size should truncate to available items");
    assert_eq!(ans[0].as_ref(), info.as_ref(), "returned item mismatch");
    assert!(stats.distances_computed > 0 && stats.nhops > 0);
}

// Edge cases — top_n > dataset size (graceful truncation & monotonicity)
#[tokio::test]
async fn hnsw_topn_greater_than_dataset_size() {
    const D: usize = 16;
    const N: usize = 20;
    const TOP_N: usize = 50; // request more than available

    // Build a small index of size N
    let mut hnsw = HnswBuilder::<_, _, InnerProductDistance, _>::new(
        D,
        StdRng::seed_from_u64(SEED),
        HnswBuilderOptions { m: 8, ef_construction: 16, max_level: 5 },
    );
    let mut data = Vec::with_capacity(N);
    for i in 0..N {
        let v = gen_vector(D);
        let info = gen_info(i);
        hnsw.insert(v.to_ref(), &info).await.unwrap();
        data.push((v, info));
    }

    // Query the first vector with two different ef_search values
    let q = data[0].0.to_ref();

    let (ans_small, _s_small) = nearest::<_, InnerProductDistance>(
        &hnsw.vector_store,
        hnsw.graph.as_ref().unwrap(),
        q,
        |_, _, info| Bytes::copy_from_slice(info),
        /* ef_search */ 8,
        TOP_N,
    )
    .await
    .unwrap();

    let (ans_large, _s_large) = nearest::<_, InnerProductDistance>(
        &hnsw.vector_store,
        hnsw.graph.as_ref().unwrap(),
        q,
        |_, _, info| Bytes::copy_from_slice(info),
        /* ef_search */ 256,
        TOP_N,
    )
    .await
    .unwrap();

    // Basic safety: never exceed what's available or requested
    assert!(ans_small.len() <= N && ans_small.len() <= TOP_N);
    assert!(ans_large.len() <= N && ans_large.len() <= TOP_N);

    // No duplicates
    let uniq_small: HashSet<_> =
        ans_small.iter().map(|b| b.as_ref().to_vec()).collect();
    assert_eq!(uniq_small.len(), ans_small.len(), "duplicates in small-ef results");

    let uniq_large: HashSet<_> =
        ans_large.iter().map(|b| b.as_ref().to_vec()).collect();
    assert_eq!(uniq_large.len(), ans_large.len(), "duplicates in large-ef results");

    // Monotonicity: allowing more exploration should not reduce the number of returned items
    assert!(
        ans_large.len() >= ans_small.len(),
        "larger ef_search should not reduce result count: {} -> {}",
        ans_small.len(),
        ans_large.len()
    );
}

// Edge cases — duplicate vectors / tie handling (stable, distinct entries)
#[tokio::test]
async fn hnsw_duplicate_vectors_tie_handling() {
    const D: usize = 16;
    const DUPS: usize = 10; // insert the same vector multiple times

    // Single vector to duplicate
    let base = gen_vector(D);

    // Distinct infos
    let infos: Vec<_> = (0..DUPS).map(gen_info).collect();

    // Build with duplicates
    let mut hnsw = HnswBuilder::<_, _, InnerProductDistance, _>::new(
        D,
        StdRng::seed_from_u64(SEED),
        HnswBuilderOptions { m: 8, ef_construction: 16, max_level: 4 },
    );
    for info in &infos {
        hnsw.insert(base.to_ref(), info).await.unwrap();
    }

    // Query with the same vector twice; verify identical ordering and distinct entries
    let (ans1, _s1) = nearest::<_, InnerProductDistance>(
        &hnsw.vector_store,
        hnsw.graph.as_ref().unwrap(),
        base.to_ref(),
        |_, _, info| Bytes::copy_from_slice(info),
        16,
        /* top_n */ DUPS,
    )
    .await
    .unwrap();

    let (ans2, _s2) = nearest::<_, InnerProductDistance>(
        &hnsw.vector_store,
        hnsw.graph.as_ref().unwrap(),
        base.to_ref(),
        |_, _, info| Bytes::copy_from_slice(info),
        16,
        /* top_n */ DUPS,
    )
    .await
    .unwrap();

    // Distinct entries returned; length equals number of duplicates
    let set1: HashSet<_> = ans1.iter().map(|b| b.as_ref().to_vec()).collect();
    assert_eq!(ans1.len(), DUPS, "should return all duplicate entries");
    assert_eq!(set1.len(), DUPS, "duplicate infos collapsed unexpectedly");

    // Deterministic ordering across runs
    assert_eq!(
        ans1, ans2,
        "tie ordering should be deterministic for identical distances"
    );
}

// Edge cases — pathological params still function (m=1, ef=1)
#[tokio::test]
async fn hnsw_pathological_params_functional() {
    const D: usize = 16;
    const N: usize = 100;
    const K: usize = 5;

    let mut hnsw = HnswBuilder::<_, _, InnerProductDistance, _>::new(
        D,
        StdRng::seed_from_u64(SEED),
        HnswBuilderOptions { m: 1, ef_construction: 1, max_level: 4 },
    );

    let mut data = Vec::with_capacity(N);
    for i in 0..N {
        let v = gen_vector(D);
        let info = gen_info(i);
        hnsw.insert(v.to_ref(), &info).await.unwrap();
        data.push((v, info));
    }

    // Random query
    let q = gen_vector(D);

    // Exact baseline
    let mut nb = NearestBuilder::<'_, _, InnerProductDistance>::new(q.to_ref(), K);
    nb.add(
        data.iter().map(|(v, info)| (v.to_ref(), info.as_ref())),
        |_, _, info| Bytes::copy_from_slice(info),
    );
    let expected = nb.finish();
    let expected_set: HashSet<_> =
        expected.iter().map(|b| b.as_ref().to_vec()).collect();

    // Very small ef_search
    let (a1, s1) = nearest::<_, InnerProductDistance>(
        &hnsw.vector_store,
        hnsw.graph.as_ref().unwrap(),
        q.to_ref(),
        |_, _, info| Bytes::copy_from_slice(info),
        /* ef_search */ 1,
        K,
    )
    .await
    .unwrap();

    // Larger ef_search
    let (a2, s2) = nearest::<_, InnerProductDistance>(
        &hnsw.vector_store,
        hnsw.graph.as_ref().unwrap(),
        q.to_ref(),
        |_, _, info| Bytes::copy_from_slice(info),
        /* ef_search */ 32,
        K,
    )
    .await
    .unwrap();

    assert!(s1.distances_computed > 0 && s1.nhops > 0);
    assert!(s2.distances_computed > 0 && s2.nhops > 0);

    let r1 = {
        let got: HashSet<_> = a1.iter().map(|b| b.as_ref().to_vec()).collect();
        expected_set.intersection(&got).count()
    };
    let r2 = {
        let got: HashSet<_> = a2.iter().map(|b| b.as_ref().to_vec()).collect();
        expected_set.intersection(&got).count()
    };

    // Larger ef shouldn't be worse; and should be nonzero recall on average
    assert!(r2 >= r1, "recall did not improve when increasing ef_search: {r1} -> {r2}");
    assert!(r2 > 0, "recall should be non-zero for ef_search=32 even with m=1");
}
}
