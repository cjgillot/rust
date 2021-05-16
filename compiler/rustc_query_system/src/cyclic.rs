use crate::dep_graph::{
    hash_result, DepContext, DepKind, DepNode, DepNodeIndex, TaskDeps, TaskDepsRef,
};
use crate::ich::StableHashingContext;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::stable_hasher::HashStable;
use rustc_data_structures::sync::Lock;
use smallvec::{smallvec, SmallVec};
use std::fmt::Debug;
use std::hash::Hash;

/// Description of a query which expects to encounter cycles and can provide a sensible answer in
/// those case.
pub trait CyclicQuery<Ctxt: DepContext> {
    const DEP_KIND: Ctxt::DepKind;
    type Key: Copy + Hash + Eq + Debug;
    type Value: Debug + for<'a> HashStable<StableHashingContext<'a>>;

    fn dep_node(tcx: Ctxt, key: Self::Key) -> DepNode<Ctxt::DepKind>;

    /// Compute the expected dependencies of this invocation.
    fn compute_dependencies(tcx: Ctxt, key: Self::Key, add_dep: impl FnMut(Self::Key));

    /// Compute the query result from its dependencies when there is no cycle.
    fn compute_direct<'a>(
        tcx: Ctxt,
        key: Self::Key,
        deps: impl Fn(&Self::Key) -> &'a Self::Value,
    ) -> Self::Value
    where
        Self::Value: 'a;

    /// Compute the results of a whole cycle.
    fn compute_cyclic<'a>(
        tcx: Ctxt,
        cycle: &[Self::Key],
        deps: impl Fn(&Self::Key) -> &'a Self::Value,
    ) -> Self::CyclicResult
    where
        Self::Value: 'a;
    type CyclicResult: Iterator<Item = Self::Value>;
}

pub struct CyclicQueryState<K, V> {
    cache: FxHashMap<K, (V, DepNodeIndex)>,
}

impl<K, V> CyclicQueryState<K, V>
where
    K: Copy + Eq + Hash + Debug,
    V: Clone,
{
    /// Force a cyclic query.
    fn compute<Ctxt, Q>(&mut self, tcx: Ctxt, key: Q::Key) -> (Q::Value, DepNodeIndex)
    where
        Ctxt: DepContext,
        Q: CyclicQuery<Ctxt, Key = K, Value = V>,
    {
        if let Some(ret) = self.cache.get(&key) {
            return ret.clone();
        }

        CycleStack::<'_, Ctxt, Q> {
            tcx,
            index: FxHashMap::default(),
            low_link: FxHashMap::default(),
            max_index: 0,
            stack: SmallVec::new(),
            on_stack: FxHashSet::default(),
            cache: &mut self.cache,
        }
        .compute_scc(key);

        // unwrap: The first caller is necessarily a SCC root.
        self.cache.get(&key).unwrap().clone()
    }

    /// Force a cyclic query.
    pub fn get<Ctxt, Q>(&mut self, tcx: Ctxt, key: Q::Key) -> Q::Value
    where
        Ctxt: DepContext,
        Q: CyclicQuery<Ctxt, Key = K, Value = V>,
    {
        let (v, dni) = self.compute::<_, Q>(tcx, key);
        tcx.dep_graph().read_index(dni);
        v
    }

    /// Force a cyclic query.
    pub fn force<Ctxt, Q>(&mut self, tcx: Ctxt, key: Q::Key) -> Q::Value
    where
        Ctxt: DepContext,
        Q: CyclicQuery<Ctxt, Key = K, Value = V>,
    {
        let (v, _) = self.compute::<_, Q>(tcx, key);
        v
    }
}

struct CycleStack<'a, Ctxt, Q>
where
    Ctxt: DepContext,
    Q: CyclicQuery<Ctxt>,
{
    tcx: Ctxt,

    index: FxHashMap<Q::Key, usize>,
    low_link: FxHashMap<Q::Key, usize>,
    max_index: usize,
    stack: SmallVec<[(Q::Key, DepNodeIndex); 4]>,
    on_stack: FxHashSet<Q::Key>,

    cache: &'a mut FxHashMap<Q::Key, (Q::Value, DepNodeIndex)>,
}

impl<Ctxt, Q> CycleStack<'_, Ctxt, Q>
where
    Ctxt: DepContext,
    Q: CyclicQuery<Ctxt>,
{
    /// Compute the SCC recursively using Tarjan's algorithm,
    /// and compute the query's result for each SCC root.
    fn compute_scc(&mut self, key: Q::Key) {
        self.index.insert(key, self.max_index);
        self.low_link.insert(key, self.max_index);
        self.max_index += 1;

        let mut pending = FxHashSet::default();
        let ((), dep_dni) = self.tcx.dep_graph().with_anon_task(self.tcx, Q::DEP_KIND, || {
            Q::compute_dependencies(self.tcx, key, |k| {
                pending.insert(k);
            })
        });
        self.stack.push((key, dep_dni));
        self.on_stack.insert(key);

        for k in pending.iter() {
            if self.cache.contains_key(k) {
                // This node has been computed independently, nothing to do.
            } else if !self.index.contains_key(k) {
                self.compute_scc(*k);
                *self.low_link.get_mut(&key).unwrap() =
                    std::cmp::min(self.low_link[&key], self.low_link[k]);
            } else if self.on_stack.contains(k) {
                *self.low_link.get_mut(&key).unwrap() =
                    std::cmp::min(self.low_link[&key], self.index[k]);
            } else {
                // This node is from a subgraph which has already been visited and
                // marked as part of another SCC.
            }
        }

        if self.low_link[&key] != self.index[&key] {
            // We are not a SCC root, nothing to do.
            return;
        }

        // We are the root of a SCC.
        let mut cycle: SmallVec<[Q::Key; 1]> = SmallVec::new();
        let mut cycle_deps: SmallVec<[DepNodeIndex; 1]> = SmallVec::new();
        while let Some((k, dni)) = self.stack.pop() {
            let is_self = k == key;

            self.on_stack.remove(&k);
            cycle.push(k);
            cycle_deps.push(dni);

            if is_self {
                break;
            }
        }
        debug_assert_eq!(cycle_deps.len(), cycle.len());

        // Explicitly mark dependencies when accessing the results.
        if cycle.len() == 1 {
            // This is not a cycle but an independent node.
            debug_assert!(cycle[0] == key);
            self.force_tree(key, cycle_deps[0], pending)
        } else {
            self.force_cycle(cycle, cycle_deps)
        }
    }

    fn force_tree(&mut self, key: Q::Key, dep_dep: DepNodeIndex, pending: FxHashSet<Q::Key>) {
        let tcx = self.tcx;
        let dep_graph = tcx.dep_graph();

        let ret = if dep_graph.is_fully_enabled() {
            let cache = &*self.cache;
            let task_deps = Lock::new(TaskDeps::default());
            let task_deps_ref = TaskDepsRef::Allow(&task_deps);
            let result = Ctxt::DepKind::with_deps(task_deps_ref, || {
                Q::compute_direct(tcx, key, |k| {
                    debug_assert!(pending.contains(k));
                    let (v, d) = &cache[k];
                    tcx.dep_graph().read_index(*d);
                    v
                })
            });
            let mut edges = task_deps.into_inner().reads;

            // Mark recursive dependencies.
            edges.push(dep_dep);

            let dep_node = Q::dep_node(tcx, key);
            dep_graph.register_node_from_edges(dep_node, tcx, result, edges, Some(hash_result))
        } else {
            let cache = &*self.cache;
            let ret = Q::compute_direct(tcx, key, |k| {
                debug_assert!(pending.contains(k));
                let (v, _) = &cache[k];
                v
            });
            (ret, dep_graph.next_virtual_depnode_index())
        };
        self.cache.insert(key, ret);
    }

    fn force_cycle(
        &mut self,
        cycle: SmallVec<[Q::Key; 1]>,
        cycle_deps: SmallVec<[DepNodeIndex; 1]>,
    ) {
        let tcx = self.tcx;
        let dep_graph = tcx.dep_graph();

        if dep_graph.is_fully_enabled() {
            // This is an actual cycle, create a common DepNode for the entire cycle,
            // to be read by each participant.
            let cache = &*self.cache;
            let (rets, dni) = tcx.dep_graph().with_anon_task(tcx, Q::DEP_KIND, || {
                // Mark recursive dependencies.
                for dni in cycle_deps {
                    tcx.dep_graph().read_index(dni);
                }
                Q::compute_cyclic(tcx, &cycle, |k| {
                    debug_assert!(cycle.contains(k));
                    let (v, d) = &cache[k];
                    tcx.dep_graph().read_index(*d);
                    v
                })
            });

            for (k, v) in std::iter::zip(cycle, rets) {
                let dep_node = Q::dep_node(tcx, k);
                // Launder the result through the dep-graph to make it forceable.
                let ret = dep_graph.register_node_from_edges(
                    dep_node,
                    tcx,
                    v,
                    smallvec![dni],
                    Some(hash_result),
                );
                self.cache.insert(k, ret);
            }
        } else {
            let cache = &*self.cache;
            let rets = Q::compute_cyclic(tcx, &cycle, |k| {
                debug_assert!(cycle.contains(k));
                let (v, _) = &cache[k];
                v
            });

            for (k, v) in std::iter::zip(cycle, rets) {
                let ret = (v, dep_graph.next_virtual_depnode_index());
                self.cache.insert(k, ret);
            }
        }
    }
}
