pub mod aggregate;
pub mod borrowck_errors;
pub mod def_use;
pub mod elaborate_drops;
pub mod patch;

mod alignment;
pub mod collect_writes;
mod graphviz;
pub mod liveness;
pub(crate) mod pretty;

pub use self::aggregate::expand_aggregate;
pub use self::alignment::is_disaligned;
pub use self::graphviz::write_node_label as write_graphviz_node_label;
pub use self::graphviz::{graphviz_safe_def_name, write_mir_graphviz};
pub use self::pretty::{dump_enabled, dump_mir, write_mir_pretty, PassWhere};

use rustc::mir::{Body, Local};
use rustc_index::vec::IndexVec;
use rustc_span::Symbol;

pub(crate) fn collect_local_names<'tcx>(
    input_body: &Body<'tcx>,
) -> IndexVec<Local, Option<Symbol>> {
    let mut local_names = IndexVec::from_elem(None, &input_body.local_decls);
    for var_debug_info in &input_body.var_debug_info {
        if let Some(local) = var_debug_info.place.as_local() {
            if let Some(prev_name) = local_names[local] {
                if var_debug_info.name != prev_name {
                    span_bug!(
                        var_debug_info.source_info.span,
                        "local {:?} has many names (`{}` vs `{}`)",
                        local,
                        prev_name,
                        var_debug_info.name
                    );
                }
            }
            local_names[local] = Some(var_debug_info.name);
        }
    }
    local_names
}
