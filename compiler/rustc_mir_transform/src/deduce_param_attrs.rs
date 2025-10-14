//! Deduces supplementary parameter attributes from MIR.
//!
//! Deduced parameter attributes are those that can only be soundly determined by examining the
//! body of the function instead of just the signature. These can be useful for optimization
//! purposes on a best-effort basis. We compute them here and store them into the crate metadata so
//! dependent crates can use them.

use rustc_hir::def_id::LocalDefId;
use rustc_index::IndexVec;
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::visit::{MutVisitor, NonMutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::{self, DeducedParamAttrs, Ty, TyCtxt};
use rustc_session::config::OptLevel;

use crate::MirPass;

/// A visitor that determines which arguments have been mutated. We can't use the mutability field
/// on LocalDecl for this because it has no meaning post-optimization.
struct DeduceReadOnly {
    /// Each bit is indexed by argument number, starting at zero (so 0 corresponds to local decl
    /// 1). The bit is true if the argument may have been mutated or false if we know it hasn't
    /// been up to the point we're at.
    mutable_args: DenseBitSet<usize>,
}

impl DeduceReadOnly {
    /// Returns a new DeduceReadOnly instance.
    fn new(arg_count: usize) -> Self {
        Self { mutable_args: DenseBitSet::new_empty(arg_count) }
    }
}

impl<'tcx> Visitor<'tcx> for DeduceReadOnly {
    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, _location: Location) {
        // We're only interested in arguments.
        if place.local == RETURN_PLACE || place.local.index() > self.mutable_args.domain_size() {
            return;
        }

        let mark_as_mutable = match context {
            PlaceContext::MutatingUse(..) => {
                // This is a mutation, so mark it as such.
                true
            }
            PlaceContext::NonMutatingUse(NonMutatingUseContext::RawBorrow) => {
                // Whether mutating though a `&raw const` is allowed is still undecided, so we
                // disable any sketchy `readonly` optimizations for now. But we only need to do
                // this if the pointer would point into the argument. IOW: for indirect places,
                // like `&raw (*local).field`, this surely cannot mutate `local`.
                !place.is_indirect()
            }
            PlaceContext::NonMutatingUse(..) | PlaceContext::NonUse(..) => {
                // Not mutating, so it's fine.
                false
            }
        };

        if mark_as_mutable {
            self.mutable_args.insert(place.local.index() - 1);
        }
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        // OK, this is subtle. Suppose that we're trying to deduce whether `x` in `f` is read-only
        // and we have the following:
        //
        //     fn f(x: BigStruct) { g(x) }
        //     fn g(mut y: BigStruct) { y.foo = 1 }
        //
        // If, at the generated MIR level, `f` turned into something like:
        //
        //      fn f(_1: BigStruct) -> () {
        //          let mut _0: ();
        //          bb0: {
        //              _0 = g(move _1) -> bb1;
        //          }
        //          ...
        //      }
        //
        // then it would be incorrect to mark `x` (i.e. `_1`) as `readonly`, because `g`'s write to
        // its copy of the indirect parameter would actually be a write directly to the pointer that
        // `f` passes. Note that function arguments are the only situation in which this problem can
        // arise: every other use of `move` in MIR doesn't actually write to the value it moves
        // from.
        if let TerminatorKind::Call { ref args, .. } = terminator.kind {
            for arg in args {
                if let Operand::Move(place) = arg.node {
                    let local = place.local;
                    if place.is_indirect()
                        || local == RETURN_PLACE
                        || local.index() > self.mutable_args.domain_size()
                    {
                        continue;
                    }

                    self.mutable_args.insert(local.index() - 1);
                }
            }
        };

        self.super_terminator(terminator, location);
    }
}

/// Returns true if values of a given type will never be passed indirectly, regardless of ABI.
fn type_will_always_be_passed_directly(ty: Ty<'_>) -> bool {
    matches!(
        ty.kind(),
        ty::Bool
            | ty::Char
            | ty::Float(..)
            | ty::Int(..)
            | ty::RawPtr(..)
            | ty::Ref(..)
            | ty::Slice(..)
            | ty::Uint(..)
    )
}

fn is_enabled(sess: &rustc_session::Session) -> bool {
    // This computation is unfortunately rather expensive, so don't do it unless we're optimizing.
    // Also skip it in incremental mode.
    sess.opts.optimize != OptLevel::No && sess.opts.incremental.is_none()
}

/// Returns the deduced parameter attributes for a function.
///
/// Deduced parameter attributes are those that can only be soundly determined by examining the
/// body of the function instead of just the signature. These can be useful for optimization
/// purposes on a best-effort basis. We compute them here and store them into the crate metadata so
/// dependent crates can use them.
pub(super) fn deduced_param_attrs<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> &'tcx [DeducedParamAttrs] {
    if !is_enabled(tcx.sess) {
        return &[];
    }

    // If the Freeze lang item isn't present, then don't bother.
    if tcx.lang_items().freeze_trait().is_none() {
        return &[];
    }

    // Codegen won't use this information for anything if all the function parameters are passed
    // directly. Detect that and bail, for compilation speed.
    let fn_ty = tcx.type_of(def_id).instantiate_identity();
    if matches!(fn_ty.kind(), ty::FnDef(..))
        && fn_ty
            .fn_sig(tcx)
            .inputs()
            .skip_binder()
            .iter()
            .cloned()
            .all(type_will_always_be_passed_directly)
    {
        return &[];
    }

    // Don't deduce any attributes for functions that have no MIR.
    if !tcx.is_mir_available(def_id) {
        return &[];
    }

    // Grab the optimized MIR. Analyze it to determine which arguments have been mutated.
    let body: &Body<'tcx> = tcx.optimized_mir(def_id);
    let mut deduce_read_only = DeduceReadOnly::new(body.arg_count);
    deduce_read_only.visit_body(body);

    // Set the `readonly` attribute for every argument that we concluded is immutable and that
    // contains no UnsafeCells.
    //
    // FIXME: This is overly conservative around generic parameters: `is_freeze()` will always
    // return false for them. For a description of alternatives that could do a better job here,
    // see [1].
    //
    // [1]: https://github.com/rust-lang/rust/pull/103172#discussion_r999139997
    let typing_env = body.typing_env(tcx);
    let mut deduced_param_attrs = tcx.arena.alloc_from_iter(
        body.local_decls.iter().skip(1).take(body.arg_count).enumerate().map(
            |(arg_index, local_decl)| DeducedParamAttrs {
                read_only: !deduce_read_only.mutable_args.contains(arg_index)
                    // We must normalize here to reveal opaques and normalize
                    // their generic parameters, otherwise we'll see exponential
                    // blow-up in compile times: #113372
                    && tcx
                        .normalize_erasing_regions(typing_env, local_decl.ty)
                        .is_freeze(tcx, typing_env),
            },
        ),
    );

    // Trailing parameters past the size of the `deduced_param_attrs` array are assumed to have the
    // default set of attributes, so we don't have to store them explicitly. Pop them off to save a
    // few bytes in metadata.
    while deduced_param_attrs.last() == Some(&DeducedParamAttrs::default()) {
        let last_index = deduced_param_attrs.len() - 1;
        deduced_param_attrs = &mut deduced_param_attrs[0..last_index];
    }

    deduced_param_attrs
}

/// `deduced_param_attrs` works on polymorphic optimized MIR. However, codegen works on
/// monomorphic MIR that may have modified in between. This pass makes sure that deduced and actual
/// param attrs match.
pub(crate) struct RecoverDeducedParamAttrs;

impl<'tcx> MirPass<'tcx> for RecoverDeducedParamAttrs {
    fn is_required(&self) -> bool {
        true
    }

    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        is_enabled(sess)
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // If we deduced nothing, there is no fixup to perform.
        let deduced_param_attrs = tcx.deduced_param_attrs(body.source.def_id());
        if deduced_param_attrs.is_empty() {
            return;
        }

        // Reuse the computation from `deduced_param_attrs`.
        let mut actual_read_only = DeduceReadOnly::new(body.arg_count);
        actual_read_only.visit_body(body);

        let wrong_mutated_args = deduced_param_attrs
            .into_iter()
            .enumerate()
            .filter(|&(arg_index, deduced)| {
                // If we deduced `read_only` and the actual MIR is mutable, we must do something.
                deduced.read_only && actual_read_only.mutable_args.contains(arg_index)
            })
            .map(|(arg_index, _)| arg_index)
            .collect::<Vec<_>>();

        if wrong_mutated_args.is_empty() {
            // All arguments match, we have nothing to do.
            return;
        }

        // For each flagged local, insert a move at the beginning of MIR. This ensures that the
        // original argument's value is not mutated, and that all mutation happens on the newly
        // introduced local.
        let mut renamed_args = IndexVec::from_fn_n(|l| l, body.arg_count + 1);
        let mut new_statements = Vec::with_capacity(wrong_mutated_args.len());
        for arg_index in wrong_mutated_args {
            let local = Local::from_usize(arg_index + 1);
            let decl = &body.local_decls[local];
            let source_info = decl.source_info;
            let new_local = body.local_decls.push(decl.clone());
            renamed_args[local] = new_local;

            // `new_local = move local`
            let stmt = StatementKind::Assign(Box::new((
                new_local.into(),
                Rvalue::Use(Operand::Move(local.into())),
            )));
            new_statements.push(Statement::new(source_info, stmt));
        }

        RenameLocals { tcx, renamed_args }.visit_body_preserves_cfg(body);
        body.basic_blocks.as_mut_preserves_cfg()[START_BLOCK]
            .statements
            .splice(0..0, new_statements);

        struct RenameLocals<'tcx> {
            tcx: TyCtxt<'tcx>,
            renamed_args: IndexVec<Local, Local>,
        }
        impl<'tcx> MutVisitor<'tcx> for RenameLocals<'tcx> {
            fn tcx(&self) -> TyCtxt<'tcx> {
                self.tcx
            }
            fn visit_local(&mut self, local: &mut Local, _: PlaceContext, _: Location) {
                if let Some(&new_local) = self.renamed_args.get(*local) {
                    *local = new_local;
                }
            }
        }
    }
}
