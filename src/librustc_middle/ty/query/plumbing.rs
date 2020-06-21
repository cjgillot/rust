//! The implementation of the query system itself. This defines the macros that
//! generate the actual methods on tcx which find and execute the provider,
//! manage the caches, and so forth.

use crate::dep_graph::DepGraph;
use crate::ty::query::Query;
use crate::ty::tls;
use crate::ty::TyCtxt;
use rustc_query_system::query::QueryContext;
use rustc_query_system::query::{QueryJobId, QueryJobInfo};

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::Lock;
use rustc_data_structures::thin_vec::ThinVec;
use rustc_errors::Diagnostic;
use rustc_span::def_id::DefId;

impl QueryContext for TyCtxt<'tcx> {
    type Query = Query<'tcx>;

    fn incremental_verify_ich(&self) -> bool {
        self.sess.opts.debugging_opts.incremental_verify_ich
    }
    fn verbose(&self) -> bool {
        self.sess.verbose()
    }

    fn def_path_str(&self, def_id: DefId) -> String {
        TyCtxt::def_path_str(*self, def_id)
    }

    fn dep_graph(&self) -> &DepGraph {
        &self.dep_graph
    }

    fn current_query_job(&self) -> Option<QueryJobId<Self::DepKind>> {
        tls::with_related_context(*self, |icx| icx.query)
    }

    fn try_collect_active_jobs(
        &self,
    ) -> Option<FxHashMap<QueryJobId<Self::DepKind>, QueryJobInfo<Self>>> {
        self.queries.try_collect_active_jobs()
    }

    /// Executes a job by changing the `ImplicitCtxt` to point to the
    /// new query job while it executes. It returns the diagnostics
    /// captured during execution and the actual result.
    #[inline(always)]
    fn start_query<R>(
        &self,
        token: QueryJobId<Self::DepKind>,
        diagnostics: Option<&Lock<ThinVec<Diagnostic>>>,
        compute: impl FnOnce(Self) -> R,
    ) -> R {
        TyCtxt::start_query(self, token, diagnostics, compute)
    }
}

macro_rules! handle_cycle_error {
    ([][$tcx: expr, $error:expr]) => {{
        $tcx.report_cycle($error).emit();
        Value::from_cycle_error($tcx)
    }};
    ([fatal_cycle $($rest:tt)*][$tcx:expr, $error:expr]) => {{
        $tcx.report_cycle($error).emit();
        $tcx.sess.abort_if_errors();
        unreachable!()
    }};
    ([cycle_delay_bug $($rest:tt)*][$tcx:expr, $error:expr]) => {{
        $tcx.report_cycle($error).delay_as_bug();
        Value::from_cycle_error($tcx)
    }};
    ([$other:ident $(($($other_args:tt)*))* $(, $($modifiers:tt)*)*][$($args:tt)*]) => {
        handle_cycle_error!([$($($modifiers)*)*][$($args)*])
    };
}

macro_rules! is_anon {
    ([]) => {{
        false
    }};
    ([anon $($rest:tt)*]) => {{
        true
    }};
    ([$other:ident $(($($other_args:tt)*))* $(, $($modifiers:tt)*)*]) => {
        is_anon!([$($($modifiers)*)*])
    };
}

macro_rules! is_eval_always {
    ([]) => {{
        false
    }};
    ([eval_always $($rest:tt)*]) => {{
        true
    }};
    ([$other:ident $(($($other_args:tt)*))* $(, $($modifiers:tt)*)*]) => {
        is_eval_always!([$($($modifiers)*)*])
    };
}

macro_rules! query_storage {
    ([][$K:ty, $V:ty]) => {
        <<$K as Key>::CacheSelector as CacheSelector<$K, $V>>::Cache
    };
    ([storage($ty:ty) $($rest:tt)*][$K:ty, $V:ty]) => {
        <$ty as CacheSelector<$K, $V>>::Cache
    };
    ([$other:ident $(($($other_args:tt)*))* $(, $($modifiers:tt)*)*][$($args:tt)*]) => {
        query_storage!([$($($modifiers)*)*][$($args)*])
    };
}

macro_rules! hash_result {
    ([][$hcx:expr, $result:expr]) => {{
        dep_graph::hash_result($hcx, &$result)
    }};
    ([no_hash $($rest:tt)*][$hcx:expr, $result:expr]) => {{
        None
    }};
    ([$other:ident $(($($other_args:tt)*))* $(, $($modifiers:tt)*)*][$($args:tt)*]) => {
        hash_result!([$($($modifiers)*)*][$($args)*])
    };
}

macro_rules! query_helper_param_ty {
    (DefId) => { impl IntoQueryParam<DefId> };
    ($K:ty) => { $K };
}

macro_rules! define_query_enum {
    (<$tcx:tt> $([$($modifiers:tt)*] fn $name:ident: $node:ident($K:ty) -> $V:ty,)*) => {

        use std::mem;
        use crate::{
            rustc_data_structures::stable_hasher::HashStable,
            rustc_data_structures::stable_hasher::StableHasher,
        };

        #[allow(nonstandard_style)]
        #[derive(Clone, Debug)]
        pub enum Query<$tcx> {
            $($name($K)),*
        }

        impl<$tcx> Query<$tcx> {
            pub fn name(&self) -> &'static str {
                match *self {
                    $(Query::$name(_) => stringify!($name),)*
                }
            }

            pub fn describe(&self, tcx: TyCtxt<$tcx>) -> Cow<'static, str> {
                let (r, name) = match *self {
                    $(Query::$name(key) => {
                        (queries::$name::describe(tcx, key), stringify!($name))
                    })*
                };
                if tcx.sess.verbose() {
                    format!("{} [{}]", r, name).into()
                } else {
                    r
                }
            }

            // FIXME(eddyb) Get more valid `Span`s on queries.
            pub fn default_span(&self, tcx: TyCtxt<$tcx>, span: Span) -> Span {
                if !span.is_dummy() {
                    return span;
                }
                // The `def_span` query is used to calculate `default_span`,
                // so exit to avoid infinite recursion.
                if let Query::def_span(..) = *self {
                    return span
                }
                match *self {
                    $(Query::$name(key) => key.default_span(tcx),)*
                }
            }
        }

        impl<'a, $tcx> HashStable<StableHashingContext<'a>> for Query<$tcx> {
            fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
                mem::discriminant(self).hash_stable(hcx, hasher);
                match *self {
                    $(Query::$name(key) => key.hash_stable(hcx, hasher),)*
                }
            }
        }
    }
}

macro_rules! define_queries {
    (<$tcx:tt> $([$($modifiers:tt)*] fn $name:ident: $node:ident($($K:tt)*) -> $V:ty,)*) => {

        use crate::ich::StableHashingContext;

        pub mod queries {
            use std::marker::PhantomData;

            $(#[allow(nonstandard_style)]
            pub struct $name<$tcx> {
                data: PhantomData<&$tcx ()>
            })*
        }

        $(impl<$tcx> QueryConfig<TyCtxt<$tcx>> for queries::$name<$tcx> {
            type Key = $($K)*;
            type Value = $V;
            type Stored = <
                query_storage!([$($modifiers)*][$($K)*, $V])
                as QueryStorage
            >::Stored;
            const NAME: &'static str = stringify!($name);
        }

        impl<$tcx> QueryAccessors<TyCtxt<$tcx>> for queries::$name<$tcx> {
            const ANON: bool = is_anon!([$($modifiers)*]);
            const EVAL_ALWAYS: bool = is_eval_always!([$($modifiers)*]);
            const DEP_KIND: dep_graph::DepKind = dep_graph::DepKind::$node;

            type Cache = query_storage!([$($modifiers)*][$($K)*, $V]);

            #[inline(always)]
            fn query_state<'a>(tcx: TyCtxt<$tcx>) -> &'a QueryState<TyCtxt<$tcx>, Self::Cache> {
                &tcx.queries.$name
            }

            #[inline]
            fn compute(tcx: TyCtxt<'tcx>, key: Self::Key) -> Self::Value {
                let provider = tcx.queries.providers.get(key.query_crate())
                    // HACK(eddyb) it's possible crates may be loaded after
                    // the query engine is created, and because crate loading
                    // is not yet integrated with the query engine, such crates
                    // would be missing appropriate entries in `providers`.
                    .unwrap_or(&tcx.queries.fallback_extern_providers)
                    .$name;
                provider(tcx, key)
            }

            fn hash_result(
                _hcx: &mut StableHashingContext<'_>,
                _result: &Self::Value
            ) -> Option<Fingerprint> {
                hash_result!([$($modifiers)*][_hcx, _result])
            }

            fn handle_cycle_error(
                tcx: TyCtxt<'tcx>,
                error: CycleError<Query<'tcx>>
            ) -> Self::Value {
                handle_cycle_error!([$($modifiers)*][tcx, error])
            }
        })*

        impl TyCtxtEnsure<$tcx> {
            $(#[inline(always)]
            pub fn $name(self, key: query_helper_param_ty!($($K)*)) {
                ensure_query::<queries::$name<'_>, _>(self.tcx, key.into_query_param())
            })*
        }

        impl TyCtxt<$tcx> {
            $(#[inline(always)]
            #[must_use]
            pub fn $name(self, key: query_helper_param_ty!($($K)*))
                -> <queries::$name<$tcx> as QueryConfig<TyCtxt<$tcx>>>::Stored
            {
                self.at(DUMMY_SP).$name(key.into_query_param())
            })*
        }

        impl TyCtxtAt<$tcx> {
            $(#[inline(always)]
            pub fn $name(self, key: query_helper_param_ty!($($K)*))
                -> <queries::$name<$tcx> as QueryConfig<TyCtxt<$tcx>>>::Stored
            {
                get_query::<queries::$name<'_>, _>(self.tcx, self.span, key.into_query_param())
            })*
        }
    }
}

macro_rules! define_queries_struct {
    (<$tcx:tt> $([$($modifiers:tt)*] fn $name:ident: $node:ident($K:ty) -> $V:ty,)*) => {
        pub struct Queries<$tcx> {
            /// This provides access to the incrimental comilation on-disk cache for query results.
            /// Do not access this directly. It is only meant to be used by
            /// `DepGraph::try_mark_green()` and the query infrastructure.
            pub(crate) on_disk_cache: OnDiskCache<'tcx>,

            providers: IndexVec<CrateNum, Providers<$tcx>>,
            fallback_extern_providers: Box<Providers<$tcx>>,

            $($name: QueryState<
                TyCtxt<$tcx>,
                <queries::$name<$tcx> as QueryAccessors<TyCtxt<'tcx>>>::Cache,
            >,)*
        }

        impl<$tcx> Queries<$tcx> {
            pub(crate) fn new(
                providers: IndexVec<CrateNum, Providers<$tcx>>,
                fallback_extern_providers: Providers<$tcx>,
                on_disk_cache: OnDiskCache<'tcx>,
            ) -> Self {
                Queries {
                    providers,
                    fallback_extern_providers: Box::new(fallback_extern_providers),
                    on_disk_cache,
                    $($name: Default::default()),*
                }
            }

            fn try_collect_active_jobs(
                &self
            ) -> Option<FxHashMap<QueryJobId<crate::dep_graph::DepKind>, QueryJobInfo<TyCtxt<'tcx>>>> {
                let mut jobs = FxHashMap::default();

                $(
                    self.$name.try_collect_active_jobs(
                        <queries::$name<'tcx> as QueryAccessors<TyCtxt<'tcx>>>::DEP_KIND,
                        Query::$name,
                        &mut jobs,
                    )?;
                )*

                Some(jobs)
            }

            fn alloc_self_profile_query_strings(
                &self,
                tcx: TyCtxt<$tcx>,
                string_cache: &mut QueryKeyStringCache,
            ) {
                use crate::ty::query::profiling_support::alloc_self_profile_query_strings_for_query_cache;

                $({
                    alloc_self_profile_query_strings_for_query_cache(
                        tcx,
                        stringify!($name),
                        &self.$name,
                        string_cache,
                    );
                })*
            }
        }
    };
}

macro_rules! define_provider_struct {
    (<$tcx:tt> $([$($modifiers:tt)*] fn $name:ident: $node:ident($K:ty) -> $R:ty,)*) => {
        pub struct Providers<$tcx> {
            $(pub $name: fn(TyCtxt<$tcx>, $K) -> $R,)*
        }

        impl<$tcx> Default for Providers<$tcx> {
            fn default() -> Self {
                $(fn $name<$tcx>(_: TyCtxt<$tcx>, key: $K) -> $R {
                    bug!("`tcx.{}({:?})` unsupported by its crate",
                         stringify!($name), key);
                })*
                Providers { $($name),* }
            }
        }

        impl<$tcx> Copy for Providers<$tcx> {}
        impl<$tcx> Clone for Providers<$tcx> {
            fn clone(&self) -> Self { *self }
        }
    };
}
