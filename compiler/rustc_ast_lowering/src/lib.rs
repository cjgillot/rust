//! Lowers the AST to the HIR.
//!
//! Since the AST and HIR are fairly similar, this is mostly a simple procedure,
//! much like a fold. Where lowering involves a bit more work things get more
//! interesting and there are some invariants you should know about. These mostly
//! concern spans and IDs.
//!
//! Spans are assigned to AST nodes during parsing and then are modified during
//! expansion to indicate the origin of a node and the process it went through
//! being expanded. IDs are assigned to AST nodes just before lowering.
//!
//! For the simpler lowering steps, IDs and spans should be preserved. Unlike
//! expansion we do not preserve the process of lowering in the spans, so spans
//! should not be modified here. When creating a new node (as opposed to
//! "folding" an existing one), create a new ID using `next_id()`.
//!
//! You must ensure that IDs are unique. That means that you should only use the
//! ID from an AST node in a single HIR node (you can assume that AST node-IDs
//! are unique). Every new node must have a unique ID. Avoid cloning HIR nodes.
//! If you do, you must then set the new node's ID to a fresh one.
//!
//! Spans are used for error messages and for tools to map semantics back to
//! source code. It is therefore not as important with spans as IDs to be strict
//! about use (you can't break the compiler by screwing up a span). Obviously, a
//! HIR node can only have a single span. But multiple nodes can have the same
//! span and spans don't need to be kept in order, etc. Where code is preserved
//! by lowering, it should have the same span as in the AST. Where HIR nodes are
//! new it is probably best to give a span for the whole AST node being lowered.
//! All nodes should have real spans; don't use dummy spans. Tools are likely to
//! get confused if the spans from leaf AST nodes occur in multiple places
//! in the HIR, especially for multiple identifiers.

#![feature(crate_visibility_modifier)]
#![feature(box_patterns)]
#![feature(iter_zip)]
#![feature(let_else)]
#![feature(never_type)]
#![recursion_limit = "256"]

use rustc_ast::token::{self, Token};
use rustc_ast::tokenstream::{CanSynthesizeMissingTokens, TokenStream, TokenTree};
use rustc_ast::visit;
use rustc_ast::{self as ast, *};
use rustc_ast_pretty::pprust;
use rustc_data_structures::captures::Captures;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::{FxHashSet, FxIndexMap};
use rustc_data_structures::sorted_map::SortedMap;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::sync::Lrc;
use rustc_errors::{struct_span_err, Applicability};
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Namespace, PartialRes, PerNS, Res};
use rustc_hir::def_id::{DefId, DefPathHash, LocalDefId, CRATE_DEF_ID};
use rustc_hir::definitions::{DefKey, DefPathData, Definitions};
use rustc_hir::intravisit;
use rustc_hir::{ConstArg, GenericArg, InferKind, ParamName};
use rustc_index::vec::{Idx, IndexVec};
use rustc_query_system::ich::StableHashingContext;
use rustc_session::lint::builtin::BARE_TRAIT_OBJECTS;
use rustc_session::lint::{BuiltinLintDiagnostics, LintBuffer};
use rustc_session::utils::{FlattenNonterminals, NtToTokenstream};
use rustc_session::Session;
use rustc_span::edition::Edition;
use rustc_span::hygiene::ExpnId;
use rustc_span::source_map::{respan, DesugaringKind};
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::{Span, DUMMY_SP};

use indexmap::map::Entry;
use smallvec::SmallVec;
use tracing::{debug, trace};

macro_rules! arena_vec {
    ($this:expr; $($x:expr),*) => ({
        let a = [$($x),*];
        $this.arena.alloc_from_iter(std::array::IntoIter::new(a))
    });
}

mod asm;
mod block;
mod expr;
mod index;
mod item;
mod pat;
mod path;

rustc_hir::arena_types!(rustc_arena::declare_arena);

struct LoweringContext<'a, 'hir: 'a> {
    /// Used to assign IDs to HIR nodes that do not directly correspond to AST nodes.
    sess: &'a Session,

    resolver: &'a mut dyn ResolverAstLowering,

    /// HACK(Centril): there is a cyclic dependency between the parser and lowering
    /// if we don't have this function pointer. To avoid that dependency so that
    /// `rustc_middle` is independent of the parser, we use dynamic dispatch here.
    nt_to_tokenstream: NtToTokenstream,

    /// Used to allocate HIR nodes.
    arena: &'hir Arena<'hir>,

    /// The items being lowered are collected here.
    owners: IndexVec<LocalDefId, Option<hir::OwnerInfo<'hir>>>,
    /// Bodies inside the owner being lowered.
    bodies: Vec<(hir::ItemLocalId, &'hir hir::Body<'hir>)>,
    /// Attributes inside the owner being lowered.
    attrs: SortedMap<hir::ItemLocalId, &'hir [Attribute]>,

    generator_kind: Option<hir::GeneratorKind>,

    /// When inside an `async` context, this is the `HirId` of the
    /// `task_context` local bound to the resume argument of the generator.
    task_context: Option<hir::HirId>,

    /// Used to get the current `fn`'s def span to point to when using `await`
    /// outside of an `async fn`.
    current_item: Option<Span>,

    catch_scope: Option<NodeId>,
    loop_scope: Option<NodeId>,
    is_in_loop_condition: bool,
    is_in_trait_impl: bool,
    is_in_dyn_type: bool,

    /// What to do when we encounter an "anonymous lifetime
    /// reference". The term "anonymous" is meant to encompass both
    /// `'_` lifetimes as well as fully elided cases where nothing is
    /// written at all (e.g., `&T` or `std::cell::Ref<T>`).
    anonymous_lifetime_mode: AnonymousLifetimeMode,

    /// Used to create lifetime definitions from in-band lifetime usages.
    /// e.g., `fn foo(x: &'x u8) -> &'x u8` to `fn foo<'x>(x: &'x u8) -> &'x u8`
    /// When a named lifetime is encountered in a function or impl header and
    /// has not been defined, it is added to this list.
    /// The results of this list are then added to the list of
    /// lifetime definitions in the corresponding impl or function generics.
    lifetimes_to_define: FxIndexMap<NodeId, (Span, ParamName)>,

    /// `true` if in-band lifetimes are being collected. This is used to
    /// indicate whether or not we're in a place where new lifetimes will result
    /// in in-band lifetime definitions, such a function or an impl header,
    /// including implicit lifetimes from `impl_header_lifetime_elision`.
    is_collecting_in_band_lifetimes: Option<LocalDefId>,

    captured_lifetimes: Option<(
        FxIndexMap<
            NodeId, // original parameter id
            (
                Span,              // Span
                NodeId,            // synthetized parameter id
                ParamName,         // parameter name
                hir::LifetimeName, // lifetime name
                LifetimeRes,       // original resolution
            ),
        >,
        FxHashSet<NodeId>,
    )>,

    current_hir_id_owner: LocalDefId,
    item_local_id_counter: hir::ItemLocalId,
    node_id_to_hir_id: IndexVec<NodeId, Option<hir::HirId>>,

    /// NodeIds that are lowered inside the current HIR owner.
    local_node_ids: Vec<NodeId>,

    allow_try_trait: Option<Lrc<[Symbol]>>,
    allow_gen_future: Option<Lrc<[Symbol]>>,
}

/// Resolution for a lifetime appearing in a type.
#[derive(Copy, Clone, Debug)]
pub enum LifetimeRes {
    /// Successfully linked the lifetime to a generic parameter.
    Param {
        /// Id of the generic parameter that introduced it.
        param: NodeId,
        /// Id of the introducing place. That can be:
        /// - an item's id, for the item's generic parameters;
        /// - a TraitRef's ref_id, identifying the `for<...>` binder;
        /// - a BareFn type's id;
        /// - a Path's id when this path has parenthesized generic args.
        binder: NodeId,
        /// Whether this parameter was introduced as in-band.
        in_band: bool,
        /// Whether this parameter was created for anonymous lifetime.
        fresh: Option<usize>,
    },
    /// This will should follow implicit lifetime resolution later.
    Anonymous {
        /// Id of the introducing place. See `Param`.
        binder: NodeId,
        /// Whether this lifetime was spelled or elided.
        elided: bool,
    },
    /// Explicit `'static` lifetime.
    Static,
    /// Resolution failure.
    Error,
}

pub trait ResolverAstLowering {
    fn def_key(&mut self, id: DefId) -> DefKey;

    fn def_span(&self, id: LocalDefId) -> Span;

    fn item_generics_num_lifetimes(&self, def: DefId) -> usize;

    fn legacy_const_generic_args(&mut self, expr: &Expr) -> Option<Vec<usize>>;

    /// Obtains resolution for a `NodeId` with a single resolution.
    fn get_partial_res(&self, id: NodeId) -> Option<PartialRes>;

    /// Obtains per-namespace resolutions for `use` statement with the given `NodeId`.
    fn get_import_res(&mut self, id: NodeId) -> PerNS<Option<Res<NodeId>>>;

    /// Obtains resolution for a label with the given `NodeId`.
    fn get_label_res(&mut self, id: NodeId) -> Option<NodeId>;

    /// Obtains resolution for a lifetime with the given `NodeId`.
    fn get_lifetime_res(&mut self, id: NodeId) -> Option<LifetimeRes>;

    /// We must keep the set of definitions up to date as we add nodes that weren't in the AST.
    /// This should only return `None` during testing.
    fn definitions(&mut self) -> &mut Definitions;

    fn create_stable_hashing_context(&self) -> StableHashingContext<'_>;

    fn lint_buffer(&mut self) -> &mut LintBuffer;

    fn next_node_id(&mut self) -> NodeId;

    fn take_trait_map(&mut self, node: NodeId) -> Option<Vec<hir::TraitCandidate>>;

    fn opt_local_def_id(&self, node: NodeId) -> Option<LocalDefId>;

    fn local_def_id(&self, node: NodeId) -> LocalDefId;

    fn def_path_hash(&self, def_id: DefId) -> DefPathHash;

    fn create_def(
        &mut self,
        parent: LocalDefId,
        node_id: ast::NodeId,
        data: DefPathData,
        expn_id: ExpnId,
        span: Span,
    ) -> LocalDefId;
}

/// Context of `impl Trait` in code, which determines whether it is allowed in an HIR subtree,
/// and if so, what meaning it has.
#[derive(Debug)]
enum ImplTraitContext<'b, 'a> {
    /// Treat `impl Trait` as shorthand for a new universal generic parameter.
    /// Example: `fn foo(x: impl Debug)`, where `impl Debug` is conceptually
    /// equivalent to a fresh universal parameter like `fn foo<T: Debug>(x: T)`.
    ///
    /// Newly generated parameters should be inserted into the given `Vec`.
    Universal(&'b mut Vec<hir::GenericParam<'a>>, LocalDefId),

    /// Treat `impl Trait` as shorthand for a new opaque type.
    /// Example: `fn foo() -> impl Debug`, where `impl Debug` is conceptually
    /// equivalent to a new opaque type like `type T = impl Debug; fn foo() -> T`.
    ///
    ReturnPositionOpaqueTy {
        /// `DefId` for the parent function, used to look up necessary
        /// information later.
        fn_def_id: LocalDefId,
        /// Origin: Either OpaqueTyOrigin::FnReturn or OpaqueTyOrigin::AsyncFn,
        origin: hir::OpaqueTyOrigin,
    },
    /// Impl trait in type aliases.
    TypeAliasesOpaqueTy {
        /// Set of lifetimes that this opaque type can capture, if it uses
        /// them. This includes lifetimes bound since we entered this context.
        /// For example:
        ///
        /// ```
        /// type A<'b> = impl for<'a> Trait<'a, Out = impl Sized + 'a>;
        /// ```
        ///
        /// Here the inner opaque type captures `'a` because it uses it. It doesn't
        /// need to capture `'b` because it already inherits the lifetime
        /// parameter from `A`.
        // FIXME(impl_trait): but `required_region_bounds` will ICE later
        // anyway.
        capturable_lifetimes: &'b mut FxHashSet<hir::LifetimeName>,
    },
    /// `impl Trait` is not accepted in this position.
    Disallowed(ImplTraitPosition),
}

/// Position in which `impl Trait` is disallowed.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum ImplTraitPosition {
    /// Disallowed in `let` / `const` / `static` bindings.
    Binding,

    /// All other positions.
    Other,
}

impl<'a> ImplTraitContext<'_, 'a> {
    #[inline]
    fn disallowed() -> Self {
        ImplTraitContext::Disallowed(ImplTraitPosition::Other)
    }

    fn reborrow<'this>(&'this mut self) -> ImplTraitContext<'this, 'a> {
        use self::ImplTraitContext::*;
        match self {
            Universal(params, parent) => Universal(params, *parent),
            ReturnPositionOpaqueTy { fn_def_id, origin } => {
                ReturnPositionOpaqueTy { fn_def_id: *fn_def_id, origin: *origin }
            }
            TypeAliasesOpaqueTy { capturable_lifetimes } => {
                TypeAliasesOpaqueTy { capturable_lifetimes }
            }
            Disallowed(pos) => Disallowed(*pos),
        }
    }
}

pub fn lower_crate<'a, 'hir>(
    sess: &'a Session,
    krate: &'a Crate,
    resolver: &'a mut dyn ResolverAstLowering,
    nt_to_tokenstream: NtToTokenstream,
    arena: &'hir Arena<'hir>,
) -> &'hir hir::Crate<'hir> {
    let _prof_timer = sess.prof.verbose_generic_activity("hir_lowering");

    let owners = IndexVec::from_fn_n(|_| None, resolver.definitions().def_index_count());
    LoweringContext {
        sess,
        resolver,
        nt_to_tokenstream,
        arena,
        owners,
        bodies: Vec::new(),
        attrs: SortedMap::new(),
        catch_scope: None,
        loop_scope: None,
        is_in_loop_condition: false,
        is_in_trait_impl: false,
        is_in_dyn_type: false,
        anonymous_lifetime_mode: AnonymousLifetimeMode::PassThrough,
        current_hir_id_owner: CRATE_DEF_ID,
        item_local_id_counter: hir::ItemLocalId::new(0),
        node_id_to_hir_id: IndexVec::new(),
        local_node_ids: Vec::new(),
        generator_kind: None,
        task_context: None,
        current_item: None,
        lifetimes_to_define: FxIndexMap::default(),
        is_collecting_in_band_lifetimes: None,
        captured_lifetimes: None,
        allow_try_trait: Some([sym::try_trait_v2][..].into()),
        allow_gen_future: Some([sym::gen_future][..].into()),
    }
    .lower_crate(krate)
}

#[derive(Copy, Clone, PartialEq)]
enum ParamMode {
    /// Any path in a type context.
    Explicit,
    /// Path in a type definition, where the anonymous lifetime `'_` is not allowed.
    ExplicitNamed,
    /// The `module::Type` in `module::Type::method` in an expression.
    Optional,
}

enum ParenthesizedGenericArgs {
    Ok,
    Err,
}

/// What to do when we encounter an **anonymous** lifetime
/// reference. Anonymous lifetime references come in two flavors. You
/// have implicit, or fully elided, references to lifetimes, like the
/// one in `&T` or `Ref<T>`, and you have `'_` lifetimes, like `&'_ T`
/// or `Ref<'_, T>`. These often behave the same, but not always:
///
/// - certain usages of implicit references are deprecated, like
///   `Ref<T>`, and we sometimes just give hard errors in those cases
///   as well.
/// - for object bounds there is a difference: `Box<dyn Foo>` is not
///   the same as `Box<dyn Foo + '_>`.
///
/// We describe the effects of the various modes in terms of three cases:
///
/// - **Modern** -- includes all uses of `'_`, but also the lifetime arg
///   of a `&` (e.g., the missing lifetime in something like `&T`)
/// - **Dyn Bound** -- if you have something like `Box<dyn Foo>`,
///   there is an elided lifetime bound (`Box<dyn Foo + 'X>`). These
///   elided bounds follow special rules. Note that this only covers
///   cases where *nothing* is written; the `'_` in `Box<dyn Foo +
///   '_>` is a case of "modern" elision.
/// - **Deprecated** -- this covers cases like `Ref<T>`, where the lifetime
///   parameter to ref is completely elided. `Ref<'_, T>` would be the modern,
///   non-deprecated equivalent.
///
/// Currently, the handling of lifetime elision is somewhat spread out
/// between HIR lowering and -- as described below -- the
/// `resolve_lifetime` module. Often we "fallthrough" to that code by generating
/// an "elided" or "underscore" lifetime name. In the future, we probably want to move
/// everything into HIR lowering.
#[derive(Copy, Clone, Debug)]
pub enum AnonymousLifetimeMode {
    /// For **Modern** cases, create a new anonymous region parameter
    /// and reference that.
    ///
    /// For **Dyn Bound** cases, pass responsibility to
    /// `resolve_lifetime` code.
    ///
    /// For **Deprecated** cases, report an error.
    CreateParameter,

    /// Give a hard error when either `&` or `'_` is written. Used to
    /// rule out things like `where T: Foo<'_>`. Does not imply an
    /// error on default object bounds (e.g., `Box<dyn Foo>`).
    ReportError,

    /// Pass responsibility to `resolve_lifetime` code for all cases.
    PassThrough,
}

impl<'a, 'hir> LoweringContext<'a, 'hir> {
    fn lower_crate(mut self, c: &Crate) -> &'hir hir::Crate<'hir> {
        debug_assert_eq!(self.resolver.local_def_id(CRATE_NODE_ID), CRATE_DEF_ID);

        visit::walk_crate(&mut item::ItemLowerer { lctx: &mut self }, c);

        self.with_hir_id_owner(CRATE_NODE_ID, |lctx| {
            let module = lctx.lower_mod(&c.items, c.span);
            lctx.lower_attrs(hir::CRATE_HIR_ID, &c.attrs);
            hir::OwnerNode::Crate(lctx.arena.alloc(module))
        });

        let hir_hash = self.compute_hir_hash();

        let mut def_id_to_hir_id = IndexVec::default();

        for (node_id, hir_id) in self.node_id_to_hir_id.into_iter_enumerated() {
            if let Some(def_id) = self.resolver.opt_local_def_id(node_id) {
                if def_id_to_hir_id.len() <= def_id.index() {
                    def_id_to_hir_id.resize(def_id.index() + 1, None);
                }
                def_id_to_hir_id[def_id] = hir_id;
            }
        }

        self.resolver.definitions().init_def_id_to_hir_id_mapping(def_id_to_hir_id);

        let krate = hir::Crate { owners: self.owners, hir_hash };
        self.arena.alloc(krate)
    }

    /// Compute the hash for the HIR of the full crate.
    /// This hash will then be part of the crate_hash which is stored in the metadata.
    fn compute_hir_hash(&mut self) -> Fingerprint {
        let definitions = self.resolver.definitions();
        let mut hir_body_nodes: Vec<_> = self
            .owners
            .iter_enumerated()
            .filter_map(|(def_id, info)| {
                let info = info.as_ref()?;
                let def_path_hash = definitions.def_path_hash(def_id);
                Some((def_path_hash, info))
            })
            .collect();
        hir_body_nodes.sort_unstable_by_key(|bn| bn.0);

        let mut stable_hasher = StableHasher::new();
        let mut hcx = self.resolver.create_stable_hashing_context();
        hir_body_nodes.hash_stable(&mut hcx, &mut stable_hasher);
        stable_hasher.finish()
    }

    fn with_hir_id_owner(
        &mut self,
        owner: NodeId,
        f: impl FnOnce(&mut Self) -> hir::OwnerNode<'hir>,
    ) -> LocalDefId {
        let def_id = self.resolver.local_def_id(owner);

        let current_attrs = std::mem::take(&mut self.attrs);
        let current_bodies = std::mem::take(&mut self.bodies);
        let current_node_ids = std::mem::take(&mut self.local_node_ids);
        let current_owner = std::mem::replace(&mut self.current_hir_id_owner, def_id);
        let current_local_counter =
            std::mem::replace(&mut self.item_local_id_counter, hir::ItemLocalId::new(1));

        // Always allocate the first `HirId` for the owner itself.
        let _old = self.node_id_to_hir_id.insert(owner, hir::HirId::make_owner(def_id));
        debug_assert_eq!(_old, None);
        self.local_node_ids.push(owner);

        let item = f(self);
        debug_assert_eq!(def_id, item.def_id());
        let info = self.make_owner_info(item);

        self.attrs = current_attrs;
        self.bodies = current_bodies;
        self.local_node_ids = current_node_ids;
        self.current_hir_id_owner = current_owner;
        self.item_local_id_counter = current_local_counter;

        let _old = self.owners.insert(def_id, info);
        debug_assert!(_old.is_none());

        def_id
    }

    fn make_owner_info(&mut self, node: hir::OwnerNode<'hir>) -> hir::OwnerInfo<'hir> {
        let attrs = std::mem::take(&mut self.attrs);
        let mut bodies = std::mem::take(&mut self.bodies);
        let local_node_ids = std::mem::take(&mut self.local_node_ids);
        let trait_map = local_node_ids
            .into_iter()
            .filter_map(|node_id| {
                let hir_id = self.node_id_to_hir_id[node_id]?;
                let traits = self.resolver.take_trait_map(node_id)?;
                Some((hir_id.local_id, traits.into_boxed_slice()))
            })
            .collect();

        #[cfg(debug_assertions)]
        for (id, attrs) in attrs.iter() {
            // Verify that we do not store empty slices in the map.
            if attrs.is_empty() {
                panic!("Stored empty attributes for {:?}", id);
            }
        }

        bodies.sort_by_key(|(k, _)| *k);
        let bodies = SortedMap::from_presorted_elements(bodies);
        let (hash_including_bodies, hash_without_bodies) = self.hash_owner(node, &bodies);
        let (nodes, parenting) =
            index::index_hir(self.sess, self.resolver.definitions(), node, &bodies);
        let nodes = hir::OwnerNodes { hash_including_bodies, hash_without_bodies, nodes, bodies };
        let attrs = {
            let mut hcx = self.resolver.create_stable_hashing_context();
            let mut stable_hasher = StableHasher::new();
            attrs.hash_stable(&mut hcx, &mut stable_hasher);
            let hash = stable_hasher.finish();
            hir::AttributeMap { map: attrs, hash }
        };

        hir::OwnerInfo { nodes, parenting, attrs, trait_map }
    }

    /// Hash the HIR node twice, one deep and one shallow hash.  This allows to differentiate
    /// queries which depend on the full HIR tree and those which only depend on the item signature.
    fn hash_owner(
        &mut self,
        node: hir::OwnerNode<'hir>,
        bodies: &SortedMap<hir::ItemLocalId, &'hir hir::Body<'hir>>,
    ) -> (Fingerprint, Fingerprint) {
        let mut hcx = self.resolver.create_stable_hashing_context();
        let mut stable_hasher = StableHasher::new();
        hcx.with_hir_bodies(true, node.def_id(), bodies, |hcx| {
            node.hash_stable(hcx, &mut stable_hasher)
        });
        let hash_including_bodies = stable_hasher.finish();
        let mut stable_hasher = StableHasher::new();
        hcx.with_hir_bodies(false, node.def_id(), bodies, |hcx| {
            node.hash_stable(hcx, &mut stable_hasher)
        });
        let hash_without_bodies = stable_hasher.finish();
        (hash_including_bodies, hash_without_bodies)
    }

    /// This method allocates a new `HirId` for the given `NodeId` and stores it in
    /// the `LoweringContext`'s `NodeId => HirId` map.
    /// Take care not to call this method if the resulting `HirId` is then not
    /// actually used in the HIR, as that would trigger an assertion in the
    /// `HirIdValidator` later on, which makes sure that all `NodeId`s got mapped
    /// properly. Calling the method twice with the same `NodeId` is fine though.
    fn lower_node_id(&mut self, ast_node_id: NodeId) -> hir::HirId {
        assert_ne!(ast_node_id, DUMMY_NODE_ID);

        *self.node_id_to_hir_id.get_or_insert_with(ast_node_id, || {
            // Generate a new `HirId`.
            let owner = self.current_hir_id_owner;
            let local_id = self.item_local_id_counter;
            self.item_local_id_counter.increment_by(1);
            self.local_node_ids.push(ast_node_id);
            hir::HirId { owner, local_id }
        })
    }

    fn next_id(&mut self) -> hir::HirId {
        let node_id = self.resolver.next_node_id();
        self.lower_node_id(node_id)
    }

    fn lower_res(&mut self, res: Res<NodeId>) -> Res {
        res.map_id(|id| {
            self.node_id_to_hir_id.get(id).copied().flatten().unwrap_or_else(|| {
                panic!("expected `NodeId` to be lowered already for res {:#?}", res);
            })
        })
    }

    fn expect_full_res(&mut self, id: NodeId) -> Res<NodeId> {
        self.resolver.get_partial_res(id).map_or(Res::Err, |pr| {
            if pr.unresolved_segments() != 0 {
                panic!("path not fully resolved: {:?}", pr);
            }
            pr.base_res()
        })
    }

    fn expect_full_res_from_use(&mut self, id: NodeId) -> impl Iterator<Item = Res<NodeId>> {
        self.resolver.get_import_res(id).present_items()
    }

    fn diagnostic(&self) -> &rustc_errors::Handler {
        self.sess.diagnostic()
    }

    /// Reuses the span but adds information like the kind of the desugaring and features that are
    /// allowed inside this span.
    fn mark_span_with_reason(
        &self,
        reason: DesugaringKind,
        span: Span,
        allow_internal_unstable: Option<Lrc<[Symbol]>>,
    ) -> Span {
        span.mark_with_reason(
            allow_internal_unstable,
            reason,
            self.sess.edition(),
            self.resolver.create_stable_hashing_context(),
        )
    }

    fn with_anonymous_lifetime_mode<R>(
        &mut self,
        anonymous_lifetime_mode: AnonymousLifetimeMode,
        op: impl FnOnce(&mut Self) -> R,
    ) -> R {
        debug!(
            "with_anonymous_lifetime_mode(anonymous_lifetime_mode={:?})",
            anonymous_lifetime_mode,
        );
        let old_anonymous_lifetime_mode = self.anonymous_lifetime_mode;
        self.anonymous_lifetime_mode = anonymous_lifetime_mode;
        let result = op(self);
        self.anonymous_lifetime_mode = old_anonymous_lifetime_mode;
        debug!(
            "with_anonymous_lifetime_mode: restoring anonymous_lifetime_mode={:?}",
            old_anonymous_lifetime_mode
        );
        result
    }

    /// Intercept all spans entering HIR.
    /// Mark a span as relative to the current owning item.
    fn lower_span(&self, span: Span) -> Span {
        if self.sess.opts.debugging_opts.incremental_relative_spans {
            span.with_parent(Some(self.current_hir_id_owner))
        } else {
            // Do not make spans relative when not using incremental compilation.
            span
        }
    }

    fn lower_ident(&self, ident: Ident) -> Ident {
        Ident::new(ident.name, self.lower_span(ident.span))
    }

    /// Creates a new `hir::GenericParam` for every new lifetime and
    /// type parameter encountered while evaluating `f`. Definitions
    /// are created with the parent provided. If no `parent_id` is
    /// provided, no definitions will be returned.
    ///
    /// Presuming that in-band lifetimes are enabled, then
    /// `self.anonymous_lifetime_mode` will be updated to match the
    /// parameter while `f` is running (and restored afterwards).
    fn collect_in_band_defs<T>(
        &mut self,
        parent_def_id: LocalDefId,
        f: impl FnOnce(&mut Self) -> T,
    ) -> (FxIndexMap<NodeId, (Span, ParamName)>, T) {
        let lifetime_stash = std::mem::take(&mut self.lifetimes_to_define);
        let was_collecting =
            std::mem::replace(&mut self.is_collecting_in_band_lifetimes, Some(parent_def_id));

        let res = f(self);

        self.is_collecting_in_band_lifetimes = was_collecting;
        let lifetimes_to_define = std::mem::replace(&mut self.lifetimes_to_define, lifetime_stash);

        (lifetimes_to_define, res)
    }

    /// Appends in-band lifetime defs and argument-position `impl
    /// Trait` defs to the existing set of generics.
    ///
    /// Presuming that in-band lifetimes are enabled, then
    /// `self.anonymous_lifetime_mode` will be updated to match the
    /// parameter while `f` is running (and restored afterwards).
    fn add_in_band_defs<T>(
        &mut self,
        generics: &Generics,
        parent_def_id: LocalDefId,
        anonymous_lifetime_mode: AnonymousLifetimeMode,
        f: impl FnOnce(&mut Self, &mut Vec<hir::GenericParam<'hir>>) -> T,
    ) -> (hir::Generics<'hir>, T) {
        let (lifetimes_to_define, (mut lowered_generics, impl_trait_defs, res)) = self
            .collect_in_band_defs(parent_def_id, |this| {
                this.with_anonymous_lifetime_mode(anonymous_lifetime_mode, |this| {
                    let mut impl_trait_defs = Vec::new();
                    let generics = this.lower_generics_mut(
                        generics,
                        ImplTraitContext::Universal(
                            &mut impl_trait_defs,
                            this.current_hir_id_owner,
                        ),
                    );
                    let res = f(this, &mut impl_trait_defs);
                    (generics, impl_trait_defs, res)
                })
            });

        lowered_generics.params.extend(
            lifetimes_to_define
                .into_iter()
                .map(|(node_id, (span, hir_name))| {
                    let kind = match hir_name {
                        ParamName::Plain(_) => hir::LifetimeParamKind::InBand,
                        ParamName::Fresh(_) => hir::LifetimeParamKind::Elided,
                        ParamName::Error => hir::LifetimeParamKind::Error,
                    };
                    hir::GenericParam {
                        hir_id: self.lower_node_id(node_id),
                        name: hir_name,
                        bounds: &[],
                        span: self.lower_span(span),
                        pure_wrt_drop: false,
                        kind: hir::GenericParamKind::Lifetime { kind },
                    }
                })
                .chain(impl_trait_defs),
        );

        let lowered_generics = lowered_generics.into_generics(self.arena);
        (lowered_generics, res)
    }

    fn with_dyn_type_scope<T>(&mut self, in_scope: bool, f: impl FnOnce(&mut Self) -> T) -> T {
        let was_in_dyn_type = self.is_in_dyn_type;
        self.is_in_dyn_type = in_scope;

        let result = f(self);

        self.is_in_dyn_type = was_in_dyn_type;

        result
    }

    fn with_new_scopes<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        let was_in_loop_condition = self.is_in_loop_condition;
        self.is_in_loop_condition = false;

        let catch_scope = self.catch_scope.take();
        let loop_scope = self.loop_scope.take();
        let ret = f(self);
        self.catch_scope = catch_scope;
        self.loop_scope = loop_scope;

        self.is_in_loop_condition = was_in_loop_condition;

        ret
    }

    fn lower_attrs(&mut self, id: hir::HirId, attrs: &[Attribute]) -> Option<&'hir [Attribute]> {
        if attrs.is_empty() {
            None
        } else {
            debug_assert_eq!(id.owner, self.current_hir_id_owner);
            let ret = self.arena.alloc_from_iter(attrs.iter().map(|a| self.lower_attr(a)));
            debug_assert!(!ret.is_empty());
            self.attrs.insert(id.local_id, ret);
            Some(ret)
        }
    }

    fn lower_attr(&self, attr: &Attribute) -> Attribute {
        // Note that we explicitly do not walk the path. Since we don't really
        // lower attributes (we use the AST version) there is nowhere to keep
        // the `HirId`s. We don't actually need HIR version of attributes anyway.
        // Tokens are also not needed after macro expansion and parsing.
        let kind = match attr.kind {
            AttrKind::Normal(ref item, _) => AttrKind::Normal(
                AttrItem {
                    path: item.path.clone(),
                    args: self.lower_mac_args(&item.args),
                    tokens: None,
                },
                None,
            ),
            AttrKind::DocComment(comment_kind, data) => AttrKind::DocComment(comment_kind, data),
        };

        Attribute { kind, id: attr.id, style: attr.style, span: self.lower_span(attr.span) }
    }

    fn alias_attrs(&mut self, id: hir::HirId, target_id: hir::HirId) {
        debug_assert_eq!(id.owner, self.current_hir_id_owner);
        debug_assert_eq!(target_id.owner, self.current_hir_id_owner);
        if let Some(&a) = self.attrs.get(&target_id.local_id) {
            debug_assert!(!a.is_empty());
            self.attrs.insert(id.local_id, a);
        }
    }

    fn lower_mac_args(&self, args: &MacArgs) -> MacArgs {
        match *args {
            MacArgs::Empty => MacArgs::Empty,
            MacArgs::Delimited(dspan, delim, ref tokens) => {
                // This is either a non-key-value attribute, or a `macro_rules!` body.
                // We either not have any nonterminals present (in the case of an attribute),
                // or have tokens available for all nonterminals in the case of a nested
                // `macro_rules`: e.g:
                //
                // ```rust
                // macro_rules! outer {
                //     ($e:expr) => {
                //         macro_rules! inner {
                //             () => { $e }
                //         }
                //     }
                // }
                // ```
                //
                // In both cases, we don't want to synthesize any tokens
                MacArgs::Delimited(
                    dspan,
                    delim,
                    self.lower_token_stream(tokens.clone(), CanSynthesizeMissingTokens::No),
                )
            }
            // This is an inert key-value attribute - it will never be visible to macros
            // after it gets lowered to HIR. Therefore, we can synthesize tokens with fake
            // spans to handle nonterminals in `#[doc]` (e.g. `#[doc = $e]`).
            MacArgs::Eq(eq_span, ref token) => {
                // In valid code the value is always representable as a single literal token.
                fn unwrap_single_token(sess: &Session, tokens: TokenStream, span: Span) -> Token {
                    if tokens.len() != 1 {
                        sess.diagnostic()
                            .delay_span_bug(span, "multiple tokens in key-value attribute's value");
                    }
                    match tokens.into_trees().next() {
                        Some(TokenTree::Token(token)) => token,
                        Some(TokenTree::Delimited(_, delim, tokens)) => {
                            if delim != token::NoDelim {
                                sess.diagnostic().delay_span_bug(
                                    span,
                                    "unexpected delimiter in key-value attribute's value",
                                )
                            }
                            unwrap_single_token(sess, tokens, span)
                        }
                        None => Token::dummy(),
                    }
                }

                let tokens = FlattenNonterminals {
                    parse_sess: &self.sess.parse_sess,
                    synthesize_tokens: CanSynthesizeMissingTokens::Yes,
                    nt_to_tokenstream: self.nt_to_tokenstream,
                }
                .process_token(token.clone());
                MacArgs::Eq(eq_span, unwrap_single_token(self.sess, tokens, token.span))
            }
        }
    }

    fn lower_token_stream(
        &self,
        tokens: TokenStream,
        synthesize_tokens: CanSynthesizeMissingTokens,
    ) -> TokenStream {
        FlattenNonterminals {
            parse_sess: &self.sess.parse_sess,
            synthesize_tokens,
            nt_to_tokenstream: self.nt_to_tokenstream,
        }
        .process_token_stream(tokens)
    }

    /// Given an associated type constraint like one of these:
    ///
    /// ```
    /// T: Iterator<Item: Debug>
    ///             ^^^^^^^^^^^
    /// T: Iterator<Item = Debug>
    ///             ^^^^^^^^^^^^
    /// ```
    ///
    /// returns a `hir::TypeBinding` representing `Item`.
    fn lower_assoc_ty_constraint(
        &mut self,
        constraint: &AssocTyConstraint,
        mut itctx: ImplTraitContext<'_, 'hir>,
    ) -> hir::TypeBinding<'hir> {
        debug!("lower_assoc_ty_constraint(constraint={:?}, itctx={:?})", constraint, itctx);

        // lower generic arguments of identifier in constraint
        let gen_args = if let Some(ref gen_args) = constraint.gen_args {
            let gen_args_ctor = match gen_args {
                GenericArgs::AngleBracketed(ref data) => {
                    self.lower_angle_bracketed_parameter_data(
                        data,
                        ParamMode::Explicit,
                        itctx.reborrow(),
                    )
                    .0
                }
                GenericArgs::Parenthesized(ref data) => {
                    let mut err = self.sess.struct_span_err(
                        gen_args.span(),
                        "parenthesized generic arguments cannot be used in associated type constraints"
                    );
                    // FIXME: try to write a suggestion here
                    err.emit();
                    self.lower_angle_bracketed_parameter_data(
                        &data.as_angle_bracketed_args(),
                        ParamMode::Explicit,
                        itctx.reborrow(),
                    )
                    .0
                }
            };
            gen_args_ctor.into_generic_args(self)
        } else {
            self.arena.alloc(hir::GenericArgs::none())
        };

        let kind = match constraint.kind {
            AssocTyConstraintKind::Equality { ref ty } => {
                hir::TypeBindingKind::Equality { ty: self.lower_ty(ty, itctx) }
            }
            AssocTyConstraintKind::Bound { ref bounds } => {
                let mut capturable_lifetimes;
                let mut parent_def_id = self.current_hir_id_owner;
                // Piggy-back on the `impl Trait` context to figure out the correct behavior.
                let (desugar_to_impl_trait, itctx) = match itctx {
                    // We are in the return position:
                    //
                    //     fn foo() -> impl Iterator<Item: Debug>
                    //
                    // so desugar to
                    //
                    //     fn foo() -> impl Iterator<Item = impl Debug>
                    ImplTraitContext::ReturnPositionOpaqueTy { .. }
                    | ImplTraitContext::TypeAliasesOpaqueTy { .. } => (true, itctx),

                    // We are in the argument position, but within a dyn type:
                    //
                    //     fn foo(x: dyn Iterator<Item: Debug>)
                    //
                    // so desugar to
                    //
                    //     fn foo(x: dyn Iterator<Item = impl Debug>)
                    ImplTraitContext::Universal(_, parent) if self.is_in_dyn_type => {
                        parent_def_id = parent;
                        (true, itctx)
                    }

                    // In `type Foo = dyn Iterator<Item: Debug>` we desugar to
                    // `type Foo = dyn Iterator<Item = impl Debug>` but we have to override the
                    // "impl trait context" to permit `impl Debug` in this position (it desugars
                    // then to an opaque type).
                    //
                    // FIXME: this is only needed until `impl Trait` is allowed in type aliases.
                    ImplTraitContext::Disallowed(_) if self.is_in_dyn_type => {
                        capturable_lifetimes = FxHashSet::default();
                        (
                            true,
                            ImplTraitContext::TypeAliasesOpaqueTy {
                                capturable_lifetimes: &mut capturable_lifetimes,
                            },
                        )
                    }

                    // We are in the parameter position, but not within a dyn type:
                    //
                    //     fn foo(x: impl Iterator<Item: Debug>)
                    //
                    // so we leave it as is and this gets expanded in astconv to a bound like
                    // `<T as Iterator>::Item: Debug` where `T` is the type parameter for the
                    // `impl Iterator`.
                    _ => (false, itctx),
                };

                if desugar_to_impl_trait {
                    // Desugar `AssocTy: Bounds` into `AssocTy = impl Bounds`. We do this by
                    // constructing the HIR for `impl bounds...` and then lowering that.

                    let impl_trait_node_id = self.resolver.next_node_id();
                    self.resolver.create_def(
                        parent_def_id,
                        impl_trait_node_id,
                        DefPathData::ImplTrait,
                        ExpnId::root(),
                        constraint.span,
                    );

                    self.with_dyn_type_scope(false, |this| {
                        let node_id = this.resolver.next_node_id();
                        let ty = this.lower_ty(
                            &Ty {
                                id: node_id,
                                kind: TyKind::ImplTrait(impl_trait_node_id, bounds.clone()),
                                span: this.lower_span(constraint.span),
                                tokens: None,
                            },
                            itctx,
                        );

                        hir::TypeBindingKind::Equality { ty }
                    })
                } else {
                    // Desugar `AssocTy: Bounds` into a type binding where the
                    // later desugars into a trait predicate.
                    let bounds = self.lower_param_bounds(bounds, itctx);

                    hir::TypeBindingKind::Constraint { bounds }
                }
            }
        };

        hir::TypeBinding {
            hir_id: self.lower_node_id(constraint.id),
            ident: self.lower_ident(constraint.ident),
            gen_args,
            kind,
            span: self.lower_span(constraint.span),
        }
    }

    fn lower_generic_arg(
        &mut self,
        arg: &ast::GenericArg,
        itctx: ImplTraitContext<'_, 'hir>,
    ) -> hir::GenericArg<'hir> {
        match arg {
            ast::GenericArg::Lifetime(lt) => GenericArg::Lifetime(self.lower_lifetime(&lt)),
            ast::GenericArg::Type(ty) => {
                match ty.kind {
                    TyKind::Infer if self.sess.features_untracked().generic_arg_infer => {
                        return GenericArg::Infer(hir::InferArg {
                            hir_id: self.lower_node_id(ty.id),
                            span: self.lower_span(ty.span),
                            kind: InferKind::Type,
                        });
                    }
                    // We parse const arguments as path types as we cannot distinguish them during
                    // parsing. We try to resolve that ambiguity by attempting resolution in both the
                    // type and value namespaces. If we resolved the path in the value namespace, we
                    // transform it into a generic const argument.
                    TyKind::Path(ref qself, ref path) => {
                        if let Some(partial_res) = self.resolver.get_partial_res(ty.id) {
                            let res = partial_res.base_res();
                            if !res.matches_ns(Namespace::TypeNS) {
                                debug!(
                                    "lower_generic_arg: Lowering type argument as const argument: {:?}",
                                    ty,
                                );

                                // Construct an AnonConst where the expr is the "ty"'s path.

                                let parent_def_id = self.current_hir_id_owner;
                                let node_id = self.resolver.next_node_id();

                                // Add a definition for the in-band const def.
                                self.resolver.create_def(
                                    parent_def_id,
                                    node_id,
                                    DefPathData::AnonConst,
                                    ExpnId::root(),
                                    ty.span,
                                );

                                let span = self.lower_span(ty.span);
                                let path_expr = Expr {
                                    id: ty.id,
                                    kind: ExprKind::Path(qself.clone(), path.clone()),
                                    span,
                                    attrs: AttrVec::new(),
                                    tokens: None,
                                };

                                let ct = self.with_new_scopes(|this| hir::AnonConst {
                                    hir_id: this.lower_node_id(node_id),
                                    body: this.lower_const_body(path_expr.span, Some(&path_expr)),
                                });
                                return GenericArg::Const(ConstArg { value: ct, span });
                            }
                        }
                    }
                    _ => {}
                }
                GenericArg::Type(self.lower_ty_direct(&ty, itctx))
            }
            ast::GenericArg::Const(ct) => GenericArg::Const(ConstArg {
                value: self.lower_anon_const(&ct),
                span: self.lower_span(ct.value.span),
            }),
        }
    }

    fn lower_ty(&mut self, t: &Ty, itctx: ImplTraitContext<'_, 'hir>) -> &'hir hir::Ty<'hir> {
        self.arena.alloc(self.lower_ty_direct(t, itctx))
    }

    fn lower_path_ty(
        &mut self,
        t: &Ty,
        qself: &Option<QSelf>,
        path: &Path,
        param_mode: ParamMode,
        itctx: ImplTraitContext<'_, 'hir>,
    ) -> hir::Ty<'hir> {
        let id = self.lower_node_id(t.id);
        let qpath = self.lower_qpath(t.id, qself, path, param_mode, itctx);
        let ty = self.ty_path(id, t.span, qpath);
        if let hir::TyKind::TraitObject(..) = ty.kind {
            self.maybe_lint_bare_trait(t.span, t.id, qself.is_none() && path.is_global());
        }
        ty
    }

    fn ty(&mut self, span: Span, kind: hir::TyKind<'hir>) -> hir::Ty<'hir> {
        hir::Ty { hir_id: self.next_id(), kind, span: self.lower_span(span) }
    }

    fn ty_tup(&mut self, span: Span, tys: &'hir [hir::Ty<'hir>]) -> hir::Ty<'hir> {
        self.ty(span, hir::TyKind::Tup(tys))
    }

    fn lower_ty_direct(&mut self, t: &Ty, mut itctx: ImplTraitContext<'_, 'hir>) -> hir::Ty<'hir> {
        let kind = match t.kind {
            TyKind::Infer => hir::TyKind::Infer,
            TyKind::Err => hir::TyKind::Err,
            TyKind::Slice(ref ty) => hir::TyKind::Slice(self.lower_ty(ty, itctx)),
            TyKind::Ptr(ref mt) => hir::TyKind::Ptr(self.lower_mt(mt, itctx)),
            TyKind::Rptr(ref region, ref mt) => {
                let lifetime = if let Some(ref region) = region {
                    self.lower_lifetime(region)
                } else {
                    let span = self.sess.source_map().next_point(t.span.shrink_to_lo());
                    let span = self.lower_span(span);
                    let id = self.resolver.next_node_id();
                    self.new_named_lifetime_with_res(
                        id,
                        span,
                        Ident::new(kw::UnderscoreLifetime, span),
                        LifetimeRes::Anonymous { binder: id, elided: false },
                    )
                };
                hir::TyKind::Rptr(lifetime, self.lower_mt(mt, itctx))
            }
            TyKind::BareFn(ref f) => {
                self.with_anonymous_lifetime_mode(AnonymousLifetimeMode::PassThrough, |this| {
                    if let Some((_, binders)) = &mut this.captured_lifetimes {
                        binders.insert(t.id);
                    }
                    let ret = hir::TyKind::BareFn(this.arena.alloc(hir::BareFnTy {
                        generic_params: this.lower_generic_params(
                            &f.generic_params,
                            ImplTraitContext::disallowed(),
                        ),
                        unsafety: this.lower_unsafety(f.unsafety),
                        abi: this.lower_extern(f.ext),
                        decl: this.lower_fn_decl(&f.decl, None, false, None),
                        param_names: this.lower_fn_params_to_names(&f.decl),
                    }));

                    if let Some((_, binders)) = &mut this.captured_lifetimes {
                        binders.remove(&t.id);
                    }
                    ret
                })
            }
            TyKind::Never => hir::TyKind::Never,
            TyKind::Tup(ref tys) => {
                hir::TyKind::Tup(self.arena.alloc_from_iter(
                    tys.iter().map(|ty| self.lower_ty_direct(ty, itctx.reborrow())),
                ))
            }
            TyKind::Paren(ref ty) => {
                return self.lower_ty_direct(ty, itctx);
            }
            TyKind::Path(ref qself, ref path) => {
                return self.lower_path_ty(t, qself, path, ParamMode::Explicit, itctx);
            }
            TyKind::ImplicitSelf => {
                let res = self.expect_full_res(t.id);
                let res = self.lower_res(res);
                hir::TyKind::Path(hir::QPath::Resolved(
                    None,
                    self.arena.alloc(hir::Path {
                        res,
                        segments: arena_vec![self; hir::PathSegment::from_ident(
                            Ident::with_dummy_span(kw::SelfUpper)
                        )],
                        span: self.lower_span(t.span),
                    }),
                ))
            }
            TyKind::Array(ref ty, ref length) => {
                hir::TyKind::Array(self.lower_ty(ty, itctx), self.lower_anon_const(length))
            }
            TyKind::Typeof(ref expr) => hir::TyKind::Typeof(self.lower_anon_const(expr)),
            TyKind::TraitObject(ref bounds, kind) => {
                let mut lifetime_bound = None;
                let (bounds, lifetime_bound) = self.with_dyn_type_scope(true, |this| {
                    let bounds =
                        this.arena.alloc_from_iter(bounds.iter().filter_map(
                            |bound| match *bound {
                                GenericBound::Trait(
                                    ref ty,
                                    TraitBoundModifier::None | TraitBoundModifier::MaybeConst,
                                ) => Some(this.lower_poly_trait_ref(ty, itctx.reborrow())),
                                // `~const ?Bound` will cause an error during AST validation
                                // anyways, so treat it like `?Bound` as compilation proceeds.
                                GenericBound::Trait(
                                    _,
                                    TraitBoundModifier::Maybe | TraitBoundModifier::MaybeConstMaybe,
                                ) => None,
                                GenericBound::Outlives(ref lifetime) => {
                                    if lifetime_bound.is_none() {
                                        lifetime_bound = Some(this.lower_lifetime(lifetime));
                                    }
                                    None
                                }
                            },
                        ));
                    let lifetime_bound =
                        lifetime_bound.unwrap_or_else(|| this.elided_dyn_bound(t.span));
                    (bounds, lifetime_bound)
                });
                if kind != TraitObjectSyntax::Dyn {
                    self.maybe_lint_bare_trait(t.span, t.id, false);
                }
                hir::TyKind::TraitObject(bounds, lifetime_bound, kind)
            }
            TyKind::ImplTrait(def_node_id, ref bounds) => {
                let span = t.span;
                match itctx {
                    ImplTraitContext::ReturnPositionOpaqueTy { fn_def_id, origin } => self
                        .lower_opaque_impl_trait(
                            span,
                            Some(fn_def_id),
                            origin,
                            def_node_id,
                            None,
                            |this| this.lower_param_bounds(bounds, itctx),
                        ),
                    ImplTraitContext::TypeAliasesOpaqueTy { ref capturable_lifetimes } => {
                        // Reset capturable lifetimes, any nested impl trait
                        // types will inherit lifetimes from this opaque type,
                        // so don't need to capture them again.
                        let nested_itctx = ImplTraitContext::TypeAliasesOpaqueTy {
                            capturable_lifetimes: &mut FxHashSet::default(),
                        };
                        self.lower_opaque_impl_trait(
                            span,
                            None,
                            hir::OpaqueTyOrigin::TyAlias,
                            def_node_id,
                            Some(capturable_lifetimes),
                            |this| this.lower_param_bounds(bounds, nested_itctx),
                        )
                    }
                    ImplTraitContext::Universal(in_band_ty_params, parent_def_id) => {
                        // Add a definition for the in-band `Param`.
                        let def_id = self.resolver.local_def_id(def_node_id);

                        let hir_bounds = self.lower_param_bounds(
                            bounds,
                            ImplTraitContext::Universal(in_band_ty_params, parent_def_id),
                        );
                        // Set the name to `impl Bound1 + Bound2`.
                        let ident = Ident::from_str_and_span(&pprust::ty_to_string(t), span);
                        in_band_ty_params.push(hir::GenericParam {
                            hir_id: self.lower_node_id(def_node_id),
                            name: ParamName::Plain(self.lower_ident(ident)),
                            pure_wrt_drop: false,
                            bounds: hir_bounds,
                            span: self.lower_span(span),
                            kind: hir::GenericParamKind::Type { default: None, synthetic: true },
                        });

                        hir::TyKind::Path(hir::QPath::Resolved(
                            None,
                            self.arena.alloc(hir::Path {
                                span: self.lower_span(span),
                                res: Res::Def(DefKind::TyParam, def_id.to_def_id()),
                                segments: arena_vec![self; hir::PathSegment::from_ident(self.lower_ident(ident))],
                            }),
                        ))
                    }
                    ImplTraitContext::Disallowed(_) => {
                        let mut err = struct_span_err!(
                            self.sess,
                            t.span,
                            E0562,
                            "`impl Trait` not allowed outside of {}",
                            "function and method return types",
                        );
                        err.emit();
                        hir::TyKind::Err
                    }
                }
            }
            TyKind::MacCall(_) => panic!("`TyKind::MacCall` should have been expanded by now"),
            TyKind::CVarArgs => {
                self.sess.delay_span_bug(
                    t.span,
                    "`TyKind::CVarArgs` should have been handled elsewhere",
                );
                hir::TyKind::Err
            }
        };

        hir::Ty { kind, span: self.lower_span(t.span), hir_id: self.lower_node_id(t.id) }
    }

    #[tracing::instrument(level = "debug", skip(self, lower_bounds))]
    fn lower_opaque_impl_trait(
        &mut self,
        span: Span,
        fn_def_id: Option<LocalDefId>,
        origin: hir::OpaqueTyOrigin,
        opaque_ty_node_id: NodeId,
        capturable_lifetimes: Option<&FxHashSet<hir::LifetimeName>>,
        lower_bounds: impl FnOnce(&mut Self) -> hir::GenericBounds<'hir>,
    ) -> hir::TyKind<'hir> {
        // Make sure we know that some funky desugaring has been going on here.
        // This is a first: there is code in other places like for loop
        // desugaring that explicitly states that we don't want to track that.
        // Not tracking it makes lints in rustc and clippy very fragile, as
        // frequently opened issues show.
        let opaque_ty_span = self.mark_span_with_reason(DesugaringKind::OpaqueTy, span, None);

        let opaque_ty_def_id = self.resolver.local_def_id(opaque_ty_node_id);

        let mut collected_lifetimes = FxIndexMap::default();
        self.with_hir_id_owner(opaque_ty_node_id, |lctx| {
            let lifetime_stash = std::mem::replace(
                &mut lctx.captured_lifetimes,
                Some((FxIndexMap::default(), FxHashSet::default())),
            );
            let hir_bounds = lower_bounds(lctx);
            collected_lifetimes =
                std::mem::replace(&mut lctx.captured_lifetimes, lifetime_stash).unwrap().0;

            if let Some(capturable_lifetimes) = capturable_lifetimes {
                collected_lifetimes
                    .retain(|_, (_, _, _, name, _)| capturable_lifetimes.contains(name));
            }
            debug!(?collected_lifetimes);

            let lifetime_defs = lctx.arena.alloc_from_iter(collected_lifetimes.iter().map(
                |(_, &(span, p_id, p_name, l_name, _))| {
                    let hir_id = lctx.lower_node_id(p_id);

                    let kind = match l_name {
                        hir::LifetimeName::Implicit | hir::LifetimeName::Underscore => {
                            hir::LifetimeParamKind::Elided
                        }
                        hir::LifetimeName::Param(_) => hir::LifetimeParamKind::Explicit,
                        _ => panic!("expected `LifetimeName::Param` or `ParamName::Plain`"),
                    };

                    debug!("create_def({:?}", (opaque_ty_def_id, p_id, p_name, span));
                    lctx.resolver.create_def(
                        opaque_ty_def_id,
                        p_id,
                        DefPathData::LifetimeNs(p_name.ident().name),
                        ExpnId::root(),
                        span.with_parent(None),
                    );

                    hir::GenericParam {
                        hir_id,
                        name: p_name,
                        span,
                        pure_wrt_drop: false,
                        bounds: &[],
                        kind: hir::GenericParamKind::Lifetime { kind },
                    }
                },
            ));

            debug!("lower_opaque_impl_trait: lifetime_defs={:#?}", lifetime_defs);

            let opaque_ty_item = hir::OpaqueTy {
                generics: hir::Generics {
                    params: lifetime_defs,
                    where_clause: hir::WhereClause { predicates: &[], span: lctx.lower_span(span) },
                    span: lctx.lower_span(span),
                },
                bounds: hir_bounds,
                origin,
            };

            trace!("lower_opaque_impl_trait: {:#?}", opaque_ty_def_id);
            lctx.generate_opaque_type(opaque_ty_def_id, opaque_ty_item, span, opaque_ty_span)
        });

        let lifetimes = self.arena.alloc_from_iter(collected_lifetimes.into_iter().map(
            |(_, (span, _, _, l_name, res))| {
                let id = self.resolver.next_node_id();
                let l = self.new_named_lifetime_with_res(id, span, l_name.ident(), res);
                hir::GenericArg::Lifetime(l)
            },
        ));

        debug!("lower_opaque_impl_trait: lifetimes={:#?}", lifetimes);

        // `impl Trait` now just becomes `Foo<'a, 'b, ..>`.
        hir::TyKind::OpaqueDef(hir::ItemId { def_id: opaque_ty_def_id }, lifetimes)
    }

    /// Registers a new opaque type with the proper `NodeId`s and
    /// returns the lowered node-ID for the opaque type.
    fn generate_opaque_type(
        &mut self,
        opaque_ty_id: LocalDefId,
        opaque_ty_item: hir::OpaqueTy<'hir>,
        span: Span,
        opaque_ty_span: Span,
    ) -> hir::OwnerNode<'hir> {
        let opaque_ty_item_kind = hir::ItemKind::OpaqueTy(opaque_ty_item);
        // Generate an `type Foo = impl Trait;` declaration.
        trace!("registering opaque type with id {:#?}", opaque_ty_id);
        let opaque_ty_item = hir::Item {
            def_id: opaque_ty_id,
            ident: Ident::empty(),
            kind: opaque_ty_item_kind,
            vis: respan(self.lower_span(span.shrink_to_lo()), hir::VisibilityKind::Inherited),
            span: self.lower_span(opaque_ty_span),
        };
        hir::OwnerNode::Item(self.arena.alloc(opaque_ty_item))
    }

    fn lower_fn_params_to_names(&mut self, decl: &FnDecl) -> &'hir [Ident] {
        // Skip the `...` (`CVarArgs`) trailing arguments from the AST,
        // as they are not explicit in HIR/Ty function signatures.
        // (instead, the `c_variadic` flag is set to `true`)
        let mut inputs = &decl.inputs[..];
        if decl.c_variadic() {
            inputs = &inputs[..inputs.len() - 1];
        }
        self.arena.alloc_from_iter(inputs.iter().map(|param| match param.pat.kind {
            PatKind::Ident(_, ident, _) => self.lower_ident(ident),
            _ => Ident::new(kw::Empty, self.lower_span(param.pat.span)),
        }))
    }

    // Lowers a function declaration.
    //
    // `decl`: the unlowered (AST) function declaration.
    // `fn_def_id`: if `Some`, impl Trait arguments are lowered into generic parameters on the
    //      given DefId, otherwise impl Trait is disallowed. Must be `Some` if
    //      `make_ret_async` is also `Some`.
    // `impl_trait_return_allow`: determines whether `impl Trait` can be used in return position.
    //      This guards against trait declarations and implementations where `impl Trait` is
    //      disallowed.
    // `make_ret_async`: if `Some`, converts `-> T` into `-> impl Future<Output = T>` in the
    //      return type. This is used for `async fn` declarations. The `NodeId` is the ID of the
    //      return type `impl Trait` item.
    fn lower_fn_decl(
        &mut self,
        decl: &FnDecl,
        mut in_band_ty_params: Option<(NodeId, &mut Vec<hir::GenericParam<'hir>>)>,
        impl_trait_return_allow: bool,
        make_ret_async: Option<NodeId>,
    ) -> &'hir hir::FnDecl<'hir> {
        debug!(
            "lower_fn_decl(\
            fn_decl: {:?}, \
            in_band_ty_params: {:?}, \
            impl_trait_return_allow: {}, \
            make_ret_async: {:?})",
            decl, in_band_ty_params, impl_trait_return_allow, make_ret_async,
        );
        let lt_mode = if make_ret_async.is_some() {
            // In `async fn`, argument-position elided lifetimes
            // must be transformed into fresh generic parameters so that
            // they can be applied to the opaque `impl Trait` return type.
            AnonymousLifetimeMode::CreateParameter
        } else {
            self.anonymous_lifetime_mode
        };

        let c_variadic = decl.c_variadic();

        // Remember how many lifetimes were already around so that we can
        // only look at the lifetime parameters introduced by the arguments.
        let inputs = self.with_anonymous_lifetime_mode(lt_mode, |this| {
            // Skip the `...` (`CVarArgs`) trailing arguments from the AST,
            // as they are not explicit in HIR/Ty function signatures.
            // (instead, the `c_variadic` flag is set to `true`)
            let mut inputs = &decl.inputs[..];
            if c_variadic {
                inputs = &inputs[..inputs.len() - 1];
            }
            this.arena.alloc_from_iter(inputs.iter().map(|param| {
                if let Some((_, ibty)) = &mut in_band_ty_params {
                    this.lower_ty_direct(
                        &param.ty,
                        ImplTraitContext::Universal(ibty, this.current_hir_id_owner),
                    )
                } else {
                    this.lower_ty_direct(&param.ty, ImplTraitContext::disallowed())
                }
            }))
        });

        let output = if let Some(ret_id) = make_ret_async {
            self.lower_async_fn_ret_ty(
                &decl.output,
                in_band_ty_params.expect("`make_ret_async` but no `fn_def_id`").0,
                ret_id,
            )
        } else {
            match decl.output {
                FnRetTy::Ty(ref ty) => {
                    let context = match in_band_ty_params {
                        Some((node_id, _)) if impl_trait_return_allow => {
                            let fn_def_id = self.resolver.local_def_id(node_id);
                            ImplTraitContext::ReturnPositionOpaqueTy {
                                fn_def_id,
                                origin: hir::OpaqueTyOrigin::FnReturn(fn_def_id),
                            }
                        }
                        _ => ImplTraitContext::disallowed(),
                    };
                    hir::FnRetTy::Return(self.lower_ty(ty, context))
                }
                FnRetTy::Default(span) => hir::FnRetTy::DefaultReturn(self.lower_span(span)),
            }
        };

        self.arena.alloc(hir::FnDecl {
            inputs,
            output,
            c_variadic,
            implicit_self: decl.inputs.get(0).map_or(hir::ImplicitSelfKind::None, |arg| {
                use BindingMode::{ByRef, ByValue};
                let is_mutable_pat = matches!(
                    arg.pat.kind,
                    PatKind::Ident(ByValue(Mutability::Mut) | ByRef(Mutability::Mut), ..)
                );

                match arg.ty.kind {
                    TyKind::ImplicitSelf if is_mutable_pat => hir::ImplicitSelfKind::Mut,
                    TyKind::ImplicitSelf => hir::ImplicitSelfKind::Imm,
                    // Given we are only considering `ImplicitSelf` types, we needn't consider
                    // the case where we have a mutable pattern to a reference as that would
                    // no longer be an `ImplicitSelf`.
                    TyKind::Rptr(_, ref mt)
                        if mt.ty.kind.is_implicit_self() && mt.mutbl == ast::Mutability::Mut =>
                    {
                        hir::ImplicitSelfKind::MutRef
                    }
                    TyKind::Rptr(_, ref mt) if mt.ty.kind.is_implicit_self() => {
                        hir::ImplicitSelfKind::ImmRef
                    }
                    _ => hir::ImplicitSelfKind::None,
                }
            }),
        })
    }

    // Transforms `-> T` for `async fn` into `-> OpaqueTy { .. }`
    // combined with the following definition of `OpaqueTy`:
    //
    //     type OpaqueTy<generics_from_parent_fn> = impl Future<Output = T>;
    //
    // `inputs`: lowered types of parameters to the function (used to collect lifetimes)
    // `output`: unlowered output type (`T` in `-> T`)
    // `fn_def_id`: `DefId` of the parent function (used to create child impl trait definition)
    // `opaque_ty_node_id`: `NodeId` of the opaque `impl Trait` type that should be created
    // `elided_lt_replacement`: replacement for elided lifetimes in the return type
    #[tracing::instrument(level = "debug", skip(self))]
    fn lower_async_fn_ret_ty(
        &mut self,
        output: &FnRetTy,
        fn_node_id: NodeId,
        opaque_ty_node_id: NodeId,
    ) -> hir::FnRetTy<'hir> {
        let span = output.span();

        let opaque_ty_span = self.mark_span_with_reason(DesugaringKind::Async, span, None);

        let opaque_ty_def_id = self.resolver.local_def_id(opaque_ty_node_id);
        let fn_def_id = self.resolver.local_def_id(fn_node_id);

        // When we create the opaque type for this async fn, it is going to have
        // to capture all the lifetimes involved in the signature (including in the
        // return type). This is done by:
        //
        // - making the opaque type inherit all lifetime parameters from its parent;
        // - make all the elided lifetimes in the fn arguments into parameters;
        // - manually introducing parameters on the opaque type for elided
        //   lifetimes in the return type.
        //
        // So for example in this snippet:
        //
        // ```rust
        // impl<'a> Foo<'a> {
        //   async fn bar<'b>(&self, x: &'b Vec<f64>, y: &str) -> &u32 {
        //   //               ^ '0                       ^ '1     ^ '2
        //   // elided lifetimes used below
        //   }
        // }
        // ```
        //
        // we would create an opaque type like:
        //
        // ```
        // type Foo<'a>::bar<'b, '0, '1>::Bar<'2> = impl Future<Output = &'2 u32>;
        // ```
        //
        // and we would then desugar `bar` to the equivalent of:
        //
        // ```rust
        // impl<'a> Foo<'a> {
        //   fn bar<'b, '0, '1>(&'0 self, x: &'b Vec<f64>, y: &'1 str) -> Bar<'_>
        // }
        // ```
        //
        // Note that the final parameter to `Bar` is `'_`, not `'2` --
        // this is because the elided lifetimes from the return type
        // should be figured out using the ordinary elision rules, and
        // this desugaring achieves that.
        let mut lifetime_params = FxIndexMap::default();
        self.with_hir_id_owner(opaque_ty_node_id, |this| {
            debug!("lower_async_fn_ret_ty: lifetimes_to_define={:#?}", this.lifetimes_to_define);

            // We have to be careful to get elision right here. The
            // idea is that we create a lifetime parameter for each
            // lifetime in the return type.  So, given a return type
            // like `async fn foo(..) -> &[&u32]`, we lower to `impl
            // Future<Output = &'1 [ &'2 u32 ]>`.
            //
            // Then, we will create `fn foo(..) -> Foo<'_, '_>`, and
            // hence the elision takes place at the fn site.
            let (lifetimes_to_define, future_bound) =
                this.with_anonymous_lifetime_mode(AnonymousLifetimeMode::PassThrough, |this| {
                    this.collect_in_band_defs(opaque_ty_def_id, |this| {
                        this.lower_async_fn_output_type_to_future_bound(output, fn_def_id, span)
                    })
                });
            debug!("lower_async_fn_ret_ty: future_bound={:#?}", future_bound);

            // Output lifetime like `'_`:
            lifetime_params = lifetimes_to_define;
            debug!("lower_async_fn_ret_ty: lifetime_params={:#?}", lifetime_params);

            let generic_params = this.arena.alloc_from_iter(lifetime_params.iter().map(
                |(&p_id, &(span, p_name))| {
                    // Get the name we'll use to make the def-path. Note
                    // that collisions are ok here and this shouldn't
                    // really show up for end-user.
                    let (str_name, kind) = match p_name {
                        ParamName::Plain(ident) => (ident.name, hir::LifetimeParamKind::InBand),
                        ParamName::Fresh(_) => {
                            (kw::UnderscoreLifetime, hir::LifetimeParamKind::Elided)
                        }
                        ParamName::Error => (kw::UnderscoreLifetime, hir::LifetimeParamKind::Error),
                    };

                    // Add a definition for the in-band lifetime def.
                    this.resolver.opt_local_def_id(p_id).unwrap_or_else(|| {
                        this.resolver.create_def(
                            opaque_ty_def_id,
                            p_id,
                            DefPathData::LifetimeNs(str_name),
                            ExpnId::root(),
                            span.with_parent(None),
                        )
                    });

                    hir::GenericParam {
                        hir_id: this.lower_node_id(p_id),
                        name: p_name,
                        bounds: &[],
                        span: this.lower_span(span),
                        pure_wrt_drop: false,
                        kind: hir::GenericParamKind::Lifetime { kind },
                    }
                },
            ));

            let opaque_ty_item = hir::OpaqueTy {
                generics: hir::Generics {
                    params: generic_params,
                    where_clause: hir::WhereClause { predicates: &[], span: this.lower_span(span) },
                    span: this.lower_span(span),
                },
                bounds: arena_vec![this; future_bound],
                origin: hir::OpaqueTyOrigin::AsyncFn(fn_def_id),
            };

            trace!("exist ty from async fn def id: {:#?}", opaque_ty_def_id);
            this.generate_opaque_type(opaque_ty_def_id, opaque_ty_item, span, opaque_ty_span)
        });

        // We need to create the lifetime arguments to our opaque type.
        // Continuing with our example, we're creating the type arguments
        // for the return type:
        //
        // ```
        // For<'a>::bar<'b, '0, '1>::Bar<'_>
        // ```
        //
        // For the "input" lifetime parameters are inherited automatically.
        // For the "output" lifetime parameters, we just want to generate `'_`.
        let generic_args =
            self.arena.alloc_from_iter(lifetime_params.into_iter().map(|(_, (span, _))| {
                // Input lifetime like `'a` or `'1`:
                let id = self.resolver.next_node_id();
                GenericArg::Lifetime(self.new_named_lifetime_with_res(
                    id,
                    span,
                    Ident::new(kw::UnderscoreLifetime, span),
                    LifetimeRes::Anonymous { binder: fn_node_id, elided: true },
                ))
            }));

        // Create the `Foo<...>` reference itself. Note that the `type
        // Foo = impl Trait` is, internally, created as a child of the
        // async fn, so the *type parameters* are inherited.  It's
        // only the lifetime parameters that we must supply.
        let opaque_ty_ref =
            hir::TyKind::OpaqueDef(hir::ItemId { def_id: opaque_ty_def_id }, generic_args);
        let opaque_ty = self.ty(opaque_ty_span, opaque_ty_ref);
        hir::FnRetTy::Return(self.arena.alloc(opaque_ty))
    }

    /// Transforms `-> T` into `Future<Output = T>`.
    fn lower_async_fn_output_type_to_future_bound(
        &mut self,
        output: &FnRetTy,
        fn_def_id: LocalDefId,
        span: Span,
    ) -> hir::GenericBound<'hir> {
        // Compute the `T` in `Future<Output = T>` from the return type.
        let output_ty = match output {
            FnRetTy::Ty(ty) => {
                // Not `OpaqueTyOrigin::AsyncFn`: that's only used for the
                // `impl Future` opaque type that `async fn` implicitly
                // generates.
                let context = ImplTraitContext::ReturnPositionOpaqueTy {
                    fn_def_id,
                    origin: hir::OpaqueTyOrigin::FnReturn(fn_def_id),
                };
                self.lower_ty(ty, context)
            }
            FnRetTy::Default(ret_ty_span) => self.arena.alloc(self.ty_tup(*ret_ty_span, &[])),
        };

        // "<Output = T>"
        let future_args = self.arena.alloc(hir::GenericArgs {
            args: &[],
            bindings: arena_vec![self; self.output_ty_binding(span, output_ty)],
            parenthesized: false,
            span_ext: DUMMY_SP,
        });

        hir::GenericBound::LangItemTrait(
            // ::std::future::Future<future_params>
            hir::LangItem::Future,
            self.lower_span(span),
            self.next_id(),
            future_args,
        )
    }

    fn lower_param_bound(
        &mut self,
        tpb: &GenericBound,
        itctx: ImplTraitContext<'_, 'hir>,
    ) -> hir::GenericBound<'hir> {
        match tpb {
            GenericBound::Trait(p, modifier) => hir::GenericBound::Trait(
                self.lower_poly_trait_ref(p, itctx),
                self.lower_trait_bound_modifier(*modifier),
            ),
            GenericBound::Outlives(lifetime) => {
                hir::GenericBound::Outlives(self.lower_lifetime(lifetime))
            }
        }
    }

    fn lower_lifetime(&mut self, l: &Lifetime) -> hir::Lifetime {
        let span = self.lower_span(l.ident.span);
        let ident = self.lower_ident(l.ident);
        let res = self.resolver.get_lifetime_res(l.id).unwrap_or(LifetimeRes::Error);
        self.new_named_lifetime_with_res(l.id, span, ident, res)
    }

    #[tracing::instrument(level = "debug", skip(self))]
    fn new_named_lifetime_with_res(
        &mut self,
        id: NodeId,
        span: Span,
        ident: Ident,
        res: LifetimeRes,
    ) -> hir::Lifetime {
        debug!(?self.captured_lifetimes);
        let (region, name) = match res {
            LifetimeRes::Param { mut param, binder, in_band, fresh } => {
                debug_assert_ne!(ident.name, kw::StaticLifetime);
                // Only items are allowed to introduce in-band lifetimes,
                // so we know `binder` has a `LocalDefId`.
                let in_band = in_band
                    && self.is_collecting_in_band_lifetimes
                        == Some(self.resolver.local_def_id(binder));
                let p_name = if let Some(index) = fresh {
                    debug_assert_eq!(ident.name, kw::UnderscoreLifetime);
                    debug_assert_eq!(
                        self.is_collecting_in_band_lifetimes,
                        Some(self.resolver.local_def_id(binder)),
                    );
                    debug_assert!(in_band);
                    ParamName::Fresh(index)
                } else {
                    debug_assert_ne!(ident.name, kw::UnderscoreLifetime);
                    ParamName::Plain(ident)
                };
                if in_band {
                    self.lifetimes_to_define.insert(param, (span, p_name));
                }
                let l_name = hir::LifetimeName::Param(p_name);
                if let Some((captures, binders)) = &mut self.captured_lifetimes {
                    match captures.entry(param) {
                        Entry::Occupied(o) => param = o.get().1,
                        Entry::Vacant(v) => {
                            if !binders.contains(&binder) {
                                let p_id = self.resolver.next_node_id();
                                v.insert((span, p_id, p_name, l_name, res));
                                param = p_id;
                            }
                        }
                    }
                }
                (Some(param), l_name)
            }
            LifetimeRes::Anonymous { binder, elided } => {
                let l_name = if elided {
                    hir::LifetimeName::Implicit
                } else {
                    hir::LifetimeName::Underscore
                };
                match &mut self.captured_lifetimes {
                    Some((captures, binders)) if !binders.contains(&binder) => {
                        let p_id = self.resolver.next_node_id();
                        let p_name = ParamName::Fresh(p_id.index());
                        captures.insert(p_id, (span, p_id, p_name, l_name, res));
                        (Some(p_id), hir::LifetimeName::Param(p_name))
                    }
                    _ => (None, l_name),
                }
            }
            LifetimeRes::Static => (None, hir::LifetimeName::Static),
            LifetimeRes::Error => (None, hir::LifetimeName::Error),
        };
        debug!(?self.captured_lifetimes);
        debug!(?region);
        let region = region.and_then(|id| self.resolver.opt_local_def_id(id));
        debug!(?region);
        hir::Lifetime { hir_id: self.lower_node_id(id), span: self.lower_span(span), name, region }
    }

    fn lower_generic_params_mut<'s>(
        &'s mut self,
        params: &'s [GenericParam],
        mut itctx: ImplTraitContext<'s, 'hir>,
    ) -> impl Iterator<Item = hir::GenericParam<'hir>> + Captures<'a> + Captures<'s> {
        params.iter().map(move |param| self.lower_generic_param(param, itctx.reborrow()))
    }

    fn lower_generic_params(
        &mut self,
        params: &[GenericParam],
        itctx: ImplTraitContext<'_, 'hir>,
    ) -> &'hir [hir::GenericParam<'hir>] {
        self.arena.alloc_from_iter(self.lower_generic_params_mut(params, itctx))
    }

    fn lower_generic_param(
        &mut self,
        param: &GenericParam,
        mut itctx: ImplTraitContext<'_, 'hir>,
    ) -> hir::GenericParam<'hir> {
        let bounds: Vec<_> = self
            .with_anonymous_lifetime_mode(AnonymousLifetimeMode::ReportError, |this| {
                this.lower_param_bounds_mut(&param.bounds, itctx.reborrow()).collect()
            });

        let (name, kind) = match param.kind {
            GenericParamKind::Lifetime => {
                let ident = self.lower_ident(param.ident);
                let param_name = ParamName::Plain(ident);
                let kind =
                    hir::GenericParamKind::Lifetime { kind: hir::LifetimeParamKind::Explicit };

                (param_name, kind)
            }
            GenericParamKind::Type { ref default, .. } => {
                let kind = hir::GenericParamKind::Type {
                    default: default.as_ref().map(|x| {
                        self.lower_ty(x, ImplTraitContext::Disallowed(ImplTraitPosition::Other))
                    }),
                    synthetic: false,
                };

                (hir::ParamName::Plain(self.lower_ident(param.ident)), kind)
            }
            GenericParamKind::Const { ref ty, kw_span: _, ref default } => {
                let ty = self
                    .with_anonymous_lifetime_mode(AnonymousLifetimeMode::ReportError, |this| {
                        this.lower_ty(&ty, ImplTraitContext::disallowed())
                    });
                let default = default.as_ref().map(|def| self.lower_anon_const(def));
                (
                    hir::ParamName::Plain(self.lower_ident(param.ident)),
                    hir::GenericParamKind::Const { ty, default },
                )
            }
        };
        let name = match name {
            hir::ParamName::Plain(ident) => hir::ParamName::Plain(self.lower_ident(ident)),
            name => name,
        };

        let hir_id = self.lower_node_id(param.id);
        self.lower_attrs(hir_id, &param.attrs);
        hir::GenericParam {
            hir_id,
            name,
            span: self.lower_span(param.ident.span),
            pure_wrt_drop: self.sess.contains_name(&param.attrs, sym::may_dangle),
            bounds: self.arena.alloc_from_iter(bounds),
            kind,
        }
    }

    fn lower_trait_ref(
        &mut self,
        p: &TraitRef,
        itctx: ImplTraitContext<'_, 'hir>,
    ) -> hir::TraitRef<'hir> {
        let path = match self.lower_qpath(p.ref_id, &None, &p.path, ParamMode::Explicit, itctx) {
            hir::QPath::Resolved(None, path) => path,
            qpath => panic!("lower_trait_ref: unexpected QPath `{:?}`", qpath),
        };
        hir::TraitRef { path, hir_ref_id: self.lower_node_id(p.ref_id) }
    }

    #[tracing::instrument(level = "debug", skip(self))]
    fn lower_poly_trait_ref(
        &mut self,
        p: &PolyTraitRef,
        mut itctx: ImplTraitContext<'_, 'hir>,
    ) -> hir::PolyTraitRef<'hir> {
        let bound_generic_params =
            self.lower_generic_params(&p.bound_generic_params, itctx.reborrow());

        // Any impl Trait types defined within this scope can capture
        // lifetimes bound on this predicate.
        let lt_def_names = p.bound_generic_params.iter().filter_map(|param| match param.kind {
            GenericParamKind::Lifetime { .. } => Some(hir::LifetimeName::Param(ParamName::Plain(
                param.ident.normalize_to_macros_2_0(),
            ))),
            _ => None,
        });
        if let ImplTraitContext::TypeAliasesOpaqueTy { ref mut capturable_lifetimes, .. } = itctx {
            capturable_lifetimes.extend(lt_def_names.clone());
        }
        if let Some((_, binders)) = &mut self.captured_lifetimes {
            binders.insert(p.trait_ref.ref_id);
        }

        let trait_ref = self.lower_trait_ref(&p.trait_ref, itctx.reborrow());

        if let ImplTraitContext::TypeAliasesOpaqueTy { ref mut capturable_lifetimes, .. } = itctx {
            for param in lt_def_names {
                capturable_lifetimes.remove(&param);
            }
        }
        if let Some((_, binders)) = &mut self.captured_lifetimes {
            binders.remove(&p.trait_ref.ref_id);
        }

        hir::PolyTraitRef { bound_generic_params, trait_ref, span: self.lower_span(p.span) }
    }

    fn lower_mt(&mut self, mt: &MutTy, itctx: ImplTraitContext<'_, 'hir>) -> hir::MutTy<'hir> {
        hir::MutTy { ty: self.lower_ty(&mt.ty, itctx), mutbl: mt.mutbl }
    }

    fn lower_param_bounds(
        &mut self,
        bounds: &[GenericBound],
        itctx: ImplTraitContext<'_, 'hir>,
    ) -> hir::GenericBounds<'hir> {
        self.arena.alloc_from_iter(self.lower_param_bounds_mut(bounds, itctx))
    }

    fn lower_param_bounds_mut<'s>(
        &'s mut self,
        bounds: &'s [GenericBound],
        mut itctx: ImplTraitContext<'s, 'hir>,
    ) -> impl Iterator<Item = hir::GenericBound<'hir>> + Captures<'s> + Captures<'a> {
        bounds.iter().map(move |bound| self.lower_param_bound(bound, itctx.reborrow()))
    }

    /// Lowers a block directly to an expression, presuming that it
    /// has no attributes and is not targeted by a `break`.
    fn lower_block_expr(&mut self, b: &Block) -> hir::Expr<'hir> {
        let block = self.lower_block(b, false);
        self.expr_block(block, AttrVec::new())
    }

    fn lower_anon_const(&mut self, c: &AnonConst) -> hir::AnonConst {
        self.with_new_scopes(|this| hir::AnonConst {
            hir_id: this.lower_node_id(c.id),
            body: this.lower_const_body(c.value.span, Some(&c.value)),
        })
    }

    fn lower_unsafe_source(&mut self, u: UnsafeSource) -> hir::UnsafeSource {
        match u {
            CompilerGenerated => hir::UnsafeSource::CompilerGenerated,
            UserProvided => hir::UnsafeSource::UserProvided,
        }
    }

    fn lower_trait_bound_modifier(&mut self, f: TraitBoundModifier) -> hir::TraitBoundModifier {
        match f {
            TraitBoundModifier::None => hir::TraitBoundModifier::None,
            TraitBoundModifier::MaybeConst => hir::TraitBoundModifier::MaybeConst,

            // `MaybeConstMaybe` will cause an error during AST validation, but we need to pick a
            // placeholder for compilation to proceed.
            TraitBoundModifier::MaybeConstMaybe | TraitBoundModifier::Maybe => {
                hir::TraitBoundModifier::Maybe
            }
        }
    }

    // Helper methods for building HIR.

    fn stmt(&mut self, span: Span, kind: hir::StmtKind<'hir>) -> hir::Stmt<'hir> {
        hir::Stmt { span: self.lower_span(span), kind, hir_id: self.next_id() }
    }

    fn stmt_expr(&mut self, span: Span, expr: hir::Expr<'hir>) -> hir::Stmt<'hir> {
        self.stmt(span, hir::StmtKind::Expr(self.arena.alloc(expr)))
    }

    fn stmt_let_pat(
        &mut self,
        attrs: Option<&'hir [Attribute]>,
        span: Span,
        init: Option<&'hir hir::Expr<'hir>>,
        pat: &'hir hir::Pat<'hir>,
        source: hir::LocalSource,
    ) -> hir::Stmt<'hir> {
        let hir_id = self.next_id();
        if let Some(a) = attrs {
            debug_assert!(!a.is_empty());
            self.attrs.insert(hir_id.local_id, a);
        }
        let local = hir::Local { hir_id, init, pat, source, span: self.lower_span(span), ty: None };
        self.stmt(span, hir::StmtKind::Local(self.arena.alloc(local)))
    }

    fn block_expr(&mut self, expr: &'hir hir::Expr<'hir>) -> &'hir hir::Block<'hir> {
        self.block_all(expr.span, &[], Some(expr))
    }

    fn block_all(
        &mut self,
        span: Span,
        stmts: &'hir [hir::Stmt<'hir>],
        expr: Option<&'hir hir::Expr<'hir>>,
    ) -> &'hir hir::Block<'hir> {
        let blk = hir::Block {
            stmts,
            expr,
            hir_id: self.next_id(),
            rules: hir::BlockCheckMode::DefaultBlock,
            span: self.lower_span(span),
            targeted_by_break: false,
        };
        self.arena.alloc(blk)
    }

    fn pat_cf_continue(&mut self, span: Span, pat: &'hir hir::Pat<'hir>) -> &'hir hir::Pat<'hir> {
        let field = self.single_pat_field(span, pat);
        self.pat_lang_item_variant(span, hir::LangItem::ControlFlowContinue, field)
    }

    fn pat_cf_break(&mut self, span: Span, pat: &'hir hir::Pat<'hir>) -> &'hir hir::Pat<'hir> {
        let field = self.single_pat_field(span, pat);
        self.pat_lang_item_variant(span, hir::LangItem::ControlFlowBreak, field)
    }

    fn pat_some(&mut self, span: Span, pat: &'hir hir::Pat<'hir>) -> &'hir hir::Pat<'hir> {
        let field = self.single_pat_field(span, pat);
        self.pat_lang_item_variant(span, hir::LangItem::OptionSome, field)
    }

    fn pat_none(&mut self, span: Span) -> &'hir hir::Pat<'hir> {
        self.pat_lang_item_variant(span, hir::LangItem::OptionNone, &[])
    }

    fn single_pat_field(
        &mut self,
        span: Span,
        pat: &'hir hir::Pat<'hir>,
    ) -> &'hir [hir::PatField<'hir>] {
        let field = hir::PatField {
            hir_id: self.next_id(),
            ident: Ident::new(sym::integer(0), self.lower_span(span)),
            is_shorthand: false,
            pat,
            span: self.lower_span(span),
        };
        arena_vec![self; field]
    }

    fn pat_lang_item_variant(
        &mut self,
        span: Span,
        lang_item: hir::LangItem,
        fields: &'hir [hir::PatField<'hir>],
    ) -> &'hir hir::Pat<'hir> {
        let qpath = hir::QPath::LangItem(lang_item, self.lower_span(span));
        self.pat(span, hir::PatKind::Struct(qpath, fields, false))
    }

    fn pat_ident(&mut self, span: Span, ident: Ident) -> (&'hir hir::Pat<'hir>, hir::HirId) {
        self.pat_ident_binding_mode(span, ident, hir::BindingAnnotation::Unannotated)
    }

    fn pat_ident_mut(&mut self, span: Span, ident: Ident) -> (hir::Pat<'hir>, hir::HirId) {
        self.pat_ident_binding_mode_mut(span, ident, hir::BindingAnnotation::Unannotated)
    }

    fn pat_ident_binding_mode(
        &mut self,
        span: Span,
        ident: Ident,
        bm: hir::BindingAnnotation,
    ) -> (&'hir hir::Pat<'hir>, hir::HirId) {
        let (pat, hir_id) = self.pat_ident_binding_mode_mut(span, ident, bm);
        (self.arena.alloc(pat), hir_id)
    }

    fn pat_ident_binding_mode_mut(
        &mut self,
        span: Span,
        ident: Ident,
        bm: hir::BindingAnnotation,
    ) -> (hir::Pat<'hir>, hir::HirId) {
        let hir_id = self.next_id();

        (
            hir::Pat {
                hir_id,
                kind: hir::PatKind::Binding(bm, hir_id, self.lower_ident(ident), None),
                span: self.lower_span(span),
                default_binding_modes: true,
            },
            hir_id,
        )
    }

    fn pat(&mut self, span: Span, kind: hir::PatKind<'hir>) -> &'hir hir::Pat<'hir> {
        self.arena.alloc(hir::Pat {
            hir_id: self.next_id(),
            kind,
            span: self.lower_span(span),
            default_binding_modes: true,
        })
    }

    fn pat_without_dbm(&mut self, span: Span, kind: hir::PatKind<'hir>) -> hir::Pat<'hir> {
        hir::Pat {
            hir_id: self.next_id(),
            kind,
            span: self.lower_span(span),
            default_binding_modes: false,
        }
    }

    fn ty_path(
        &mut self,
        mut hir_id: hir::HirId,
        span: Span,
        qpath: hir::QPath<'hir>,
    ) -> hir::Ty<'hir> {
        let kind = match qpath {
            hir::QPath::Resolved(None, path) => {
                // Turn trait object paths into `TyKind::TraitObject` instead.
                match path.res {
                    Res::Def(DefKind::Trait | DefKind::TraitAlias, _) => {
                        let principal = hir::PolyTraitRef {
                            bound_generic_params: &[],
                            trait_ref: hir::TraitRef { path, hir_ref_id: hir_id },
                            span: self.lower_span(span),
                        };

                        // The original ID is taken by the `PolyTraitRef`,
                        // so the `Ty` itself needs a different one.
                        hir_id = self.next_id();
                        hir::TyKind::TraitObject(
                            arena_vec![self; principal],
                            self.elided_dyn_bound(span),
                            TraitObjectSyntax::None,
                        )
                    }
                    _ => hir::TyKind::Path(hir::QPath::Resolved(None, path)),
                }
            }
            _ => hir::TyKind::Path(qpath),
        };

        hir::Ty { hir_id, kind, span: self.lower_span(span) }
    }

    /// Invoked to create the lifetime argument(s) for an elided trait object
    /// bound, like the bound in `Box<dyn Debug>`. This method is not invoked
    /// when the bound is written, even if it is written with `'_` like in
    /// `Box<dyn Debug + '_>`. In those cases, `lower_lifetime` is invoked.
    fn elided_dyn_bound(&mut self, span: Span) -> hir::Lifetime {
        match self.anonymous_lifetime_mode {
            // NB. We intentionally ignore the create-parameter mode here.
            // and instead "pass through" to resolve-lifetimes, which will apply
            // the object-lifetime-defaulting rules. Elided object lifetime defaults
            // do not act like other elided lifetimes. In other words, given this:
            //
            //     impl Foo for Box<dyn Debug>
            //
            // we do not introduce a fresh `'_` to serve as the bound, but instead
            // ultimately translate to the equivalent of:
            //
            //     impl Foo for Box<dyn Debug + 'static>
            //
            // `resolve_lifetime` has the code to make that happen.
            AnonymousLifetimeMode::CreateParameter => {}

            AnonymousLifetimeMode::ReportError => {
                // ReportError applies to explicit use of `'_`.
            }

            // This is the normal case.
            AnonymousLifetimeMode::PassThrough => {}
        }

        let r = hir::Lifetime {
            hir_id: self.next_id(),
            span: self.lower_span(span),
            name: hir::LifetimeName::ImplicitObjectLifetimeDefault,
            region: None,
        };
        debug!("elided_dyn_bound: r={:?}", r);
        r
    }

    fn maybe_lint_bare_trait(&mut self, span: Span, id: NodeId, is_global: bool) {
        // FIXME(davidtwco): This is a hack to detect macros which produce spans of the
        // call site which do not have a macro backtrace. See #61963.
        let is_macro_callsite = self
            .sess
            .source_map()
            .span_to_snippet(span)
            .map(|snippet| snippet.starts_with("#["))
            .unwrap_or(true);
        if !is_macro_callsite {
            if span.edition() < Edition::Edition2021 {
                self.resolver.lint_buffer().buffer_lint_with_diagnostic(
                    BARE_TRAIT_OBJECTS,
                    id,
                    span,
                    "trait objects without an explicit `dyn` are deprecated",
                    BuiltinLintDiagnostics::BareTraitObject(span, is_global),
                )
            } else {
                let msg = "trait objects must include the `dyn` keyword";
                let label = "add `dyn` keyword before this trait";
                let mut err = struct_span_err!(self.sess, span, E0782, "{}", msg,);
                err.span_suggestion_verbose(
                    span.shrink_to_lo(),
                    label,
                    String::from("dyn "),
                    Applicability::MachineApplicable,
                );
                err.emit();
            }
        }
    }
}

/// Helper struct for delayed construction of GenericArgs.
struct GenericArgsCtor<'hir> {
    args: SmallVec<[hir::GenericArg<'hir>; 4]>,
    bindings: &'hir [hir::TypeBinding<'hir>],
    parenthesized: bool,
    span: Span,
}

impl<'hir> GenericArgsCtor<'hir> {
    fn is_empty(&self) -> bool {
        self.args.is_empty() && self.bindings.is_empty() && !self.parenthesized
    }

    fn into_generic_args(self, this: &LoweringContext<'_, 'hir>) -> &'hir hir::GenericArgs<'hir> {
        let ga = hir::GenericArgs {
            args: this.arena.alloc_from_iter(self.args),
            bindings: self.bindings,
            parenthesized: self.parenthesized,
            span_ext: this.lower_span(self.span),
        };
        this.arena.alloc(ga)
    }
}

#[tracing::instrument(level = "debug")]
fn lifetimes_from_impl_trait_bounds(
    opaque_ty_id: NodeId,
    bounds: hir::GenericBounds<'_>,
    lifetimes_to_include: Option<&FxHashSet<hir::LifetimeName>>,
) -> Vec<(hir::LifetimeName, Span)> {
    // This visitor walks over `impl Trait` bounds and creates defs for all lifetimes that
    // appear in the bounds, excluding lifetimes that are created within the bounds.
    // E.g., `'a`, `'b`, but not `'c` in `impl for<'c> SomeTrait<'a, 'b, 'c>`.
    struct ImplTraitLifetimeCollector<'r> {
        collect_elided_lifetimes: bool,
        currently_bound_lifetimes: Vec<hir::LifetimeName>,
        already_defined_lifetimes: FxHashSet<hir::LifetimeName>,
        lifetimes: Vec<(hir::LifetimeName, Span)>,
        lifetimes_to_include: Option<&'r FxHashSet<hir::LifetimeName>>,
    }

    impl<'r, 'v> intravisit::Visitor<'v> for ImplTraitLifetimeCollector<'r> {
        type Map = intravisit::ErasedMap<'v>;

        fn nested_visit_map(&mut self) -> intravisit::NestedVisitorMap<Self::Map> {
            intravisit::NestedVisitorMap::None
        }

        fn visit_generic_args(&mut self, span: Span, parameters: &'v hir::GenericArgs<'v>) {
            // Don't collect elided lifetimes used inside of `Fn()` syntax.
            if parameters.parenthesized {
                let old_collect_elided_lifetimes = self.collect_elided_lifetimes;
                self.collect_elided_lifetimes = false;
                intravisit::walk_generic_args(self, span, parameters);
                self.collect_elided_lifetimes = old_collect_elided_lifetimes;
            } else {
                intravisit::walk_generic_args(self, span, parameters);
            }
        }

        fn visit_ty(&mut self, t: &'v hir::Ty<'v>) {
            // Don't collect elided lifetimes used inside of `fn()` syntax.
            if let hir::TyKind::BareFn(_) = t.kind {
                let old_collect_elided_lifetimes = self.collect_elided_lifetimes;
                self.collect_elided_lifetimes = false;

                // Record the "stack height" of `for<'a>` lifetime bindings
                // to be able to later fully undo their introduction.
                let old_len = self.currently_bound_lifetimes.len();
                intravisit::walk_ty(self, t);
                self.currently_bound_lifetimes.truncate(old_len);

                self.collect_elided_lifetimes = old_collect_elided_lifetimes;
            } else {
                intravisit::walk_ty(self, t)
            }
        }

        fn visit_poly_trait_ref(
            &mut self,
            trait_ref: &'v hir::PolyTraitRef<'v>,
            modifier: hir::TraitBoundModifier,
        ) {
            // Record the "stack height" of `for<'a>` lifetime bindings
            // to be able to later fully undo their introduction.
            let old_len = self.currently_bound_lifetimes.len();
            intravisit::walk_poly_trait_ref(self, trait_ref, modifier);
            self.currently_bound_lifetimes.truncate(old_len);
        }

        fn visit_generic_param(&mut self, param: &'v hir::GenericParam<'v>) {
            // Record the introduction of 'a in `for<'a> ...`.
            if let hir::GenericParamKind::Lifetime { .. } = param.kind {
                // Introduce lifetimes one at a time so that we can handle
                // cases like `fn foo<'d>() -> impl for<'a, 'b: 'a, 'c: 'b + 'd>`.
                let lt_name = hir::LifetimeName::Param(param.name);
                self.currently_bound_lifetimes.push(lt_name);
            }

            intravisit::walk_generic_param(self, param);
        }

        fn visit_lifetime(&mut self, lifetime: &'v hir::Lifetime) {
            let name = match lifetime.name {
                hir::LifetimeName::Implicit | hir::LifetimeName::Underscore => {
                    if self.collect_elided_lifetimes {
                        // Use `'_` for both implicit and underscore lifetimes in
                        // `type Foo<'_> = impl SomeTrait<'_>;`.
                        hir::LifetimeName::Underscore
                    } else {
                        return;
                    }
                }
                hir::LifetimeName::Param(_) => lifetime.name,

                // Refers to some other lifetime that is "in
                // scope" within the type.
                hir::LifetimeName::ImplicitObjectLifetimeDefault => return,

                hir::LifetimeName::Error | hir::LifetimeName::Static => return,
            };

            if !self.currently_bound_lifetimes.contains(&name)
                && !self.already_defined_lifetimes.contains(&name)
                && self
                    .lifetimes_to_include
                    .map_or(name.is_elided(), |lifetimes| lifetimes.contains(&name))
            {
                self.already_defined_lifetimes.insert(name);

                self.lifetimes.push((name, lifetime.span));
            }
        }
    }

    let mut lifetime_collector = ImplTraitLifetimeCollector {
        collect_elided_lifetimes: true,
        currently_bound_lifetimes: Vec::new(),
        already_defined_lifetimes: FxHashSet::default(),
        lifetimes: Vec::new(),
        lifetimes_to_include,
    };

    for bound in bounds {
        intravisit::walk_param_bound(&mut lifetime_collector, &bound);
    }

    lifetime_collector.lifetimes
}
