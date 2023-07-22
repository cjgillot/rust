//! A pass that propagates the unreachable terminator of a block to its predecessors
//! when all of their successors are unreachable. This is achieved through a
//! post-order traversal of the blocks.

use crate::MirPass;
use rustc_data_structures::fx::FxHashSet;
use rustc_middle::mir::patch::MirPatch;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;

pub struct UnreachablePropagation;

impl MirPass<'_> for UnreachablePropagation {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        // Enable only under -Zmir-opt-level=2 as this can make programs less debuggable.
        sess.mir_opt_level() >= 2
    }

    fn run_pass<'tcx>(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let mut patch = MirPatch::new(body);
        let mut unreachable_blocks = FxHashSet::default();

        for (bb, bb_data) in traversal::postorder(body) {
            let terminator = bb_data.terminator();
            let is_unreachable = match &terminator.kind {
                TerminatorKind::Unreachable => true,
                // This will unconditionally run into an unreachable and is therefore unreachable as well.
                TerminatorKind::Goto { target } if unreachable_blocks.contains(target) => {
                    patch.patch_terminator(bb, TerminatorKind::Unreachable);
                    true
                }
                // Try to remove unreachable targets from the switch.
                TerminatorKind::SwitchInt { .. } => {
                    remove_successors_from_switch(bb, &unreachable_blocks, body, &mut patch)
                }
                _ => false,
            };
            if is_unreachable {
                unreachable_blocks.insert(bb);
            }
        }

        if !tcx
            .consider_optimizing(|| format!("UnreachablePropagation {:?} ", body.source.def_id()))
        {
            return;
        }

        patch.apply(body);

        // We do want do keep some unreachable blocks, but make them empty.
        for bb in unreachable_blocks {
            body.basic_blocks_mut()[bb].statements.clear();
        }
    }
}

/// Return whether the current terminator is fully unreachable.
fn remove_successors_from_switch<'tcx>(
    bb: BasicBlock,
    unreachable_blocks: &FxHashSet<BasicBlock>,
    body: &Body<'tcx>,
    patch: &mut MirPatch<'tcx>,
) -> bool {
    let terminator = body.basic_blocks[bb].terminator();
    let TerminatorKind::SwitchInt { discr, targets } = &terminator.kind else { bug!() };
    let location = body.terminator_loc(bb);

    let is_unreachable = |bb| unreachable_blocks.contains(&bb);

    // If there are multiple targets, we want to keep information about reachability for codegen.
    // For example (see tests/codegen/match-optimizes-away.rs)
    //
    // pub enum Two { A, B }
    // pub fn identity(x: Two) -> Two {
    //     match x {
    //         Two::A => Two::A,
    //         Two::B => Two::B,
    //     }
    // }
    //
    // This generates a `switchInt() -> [0: 0, 1: 1, otherwise: unreachable]`, which allows us or LLVM to
    // turn it into just `x` later. Without the unreachable, such a transformation would be illegal.
    //
    // In order to preserve this information, we record reachable and unreachable targets as
    // `Assume` statements in MIR.

    let mut add_assumption = |binop, value| {
        let assume = NonDivergingIntrinsic::Assume(discr.to_copy(), binop, value);
        patch.add_statement(location, StatementKind::Intrinsic(Box::new(assume)));
    };

    let reachable_iter = targets.iter().filter(|&(value, bb)| {
        let is_unreachable = is_unreachable(bb);
        if is_unreachable {
            // We remove this target from the switch, so record the inequality using `Assume`.
            add_assumption(BinOp::Ne, value);
            false
        } else {
            true
        }
    });

    let otherwise = targets.otherwise();
    let new_targets = SwitchTargets::new(reachable_iter, otherwise);

    let num_targets = new_targets.all_targets().len();
    let otherwise_unreachable = is_unreachable(otherwise);
    let fully_unreachable = num_targets == 1 && otherwise_unreachable;

    let terminator = match (num_targets, otherwise_unreachable) {
        // If all targets are unreachable, we can be unreachable as well.
        (1, true) => TerminatorKind::Unreachable,
        (1, false) => TerminatorKind::Goto { target: otherwise },
        (2, true) => {
            // All targets are unreachable except one. Record the equality, and make it a goto.
            let (value, target) = new_targets.iter().next().unwrap();
            add_assumption(BinOp::Eq, value);
            TerminatorKind::Goto { target }
        }
        _ if num_targets == targets.all_targets().len() => {
            // Nothing has changed.
            return false;
        }
        _ => TerminatorKind::SwitchInt { discr: discr.clone(), targets: new_targets },
    };

    patch.patch_terminator(bb, terminator);
    fully_unreachable
}
