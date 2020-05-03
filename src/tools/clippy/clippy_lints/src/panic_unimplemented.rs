use crate::utils::{is_direct_expn_of, is_expn_of, match_function_call, paths, span_lint};
use if_chain::if_chain;
use rustc_ast::ast::LitKind;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::Span;

declare_clippy_lint! {
    /// **What it does:** Checks for missing parameters in `panic!`.
    ///
    /// **Why is this bad?** Contrary to the `format!` family of macros, there are
    /// two forms of `panic!`: if there are no parameters given, the first argument
    /// is not a format string and used literally. So while `format!("{}")` will
    /// fail to compile, `panic!("{}")` will not.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```no_run
    /// panic!("This `panic!` is probably missing a parameter there: {}");
    /// ```
    pub PANIC_PARAMS,
    style,
    "missing parameters in `panic!` calls"
}

declare_clippy_lint! {
    /// **What it does:** Checks for usage of `panic!`.
    ///
    /// **Why is this bad?** `panic!` will stop the execution of the executable
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```no_run
    /// panic!("even with a good reason");
    /// ```
    pub PANIC,
    restriction,
    "usage of the `panic!` macro"
}

declare_clippy_lint! {
    /// **What it does:** Checks for usage of `unimplemented!`.
    ///
    /// **Why is this bad?** This macro should not be present in production code
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```no_run
    /// unimplemented!();
    /// ```
    pub UNIMPLEMENTED,
    restriction,
    "`unimplemented!` should not be present in production code"
}

declare_clippy_lint! {
    /// **What it does:** Checks for usage of `todo!`.
    ///
    /// **Why is this bad?** This macro should not be present in production code
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```no_run
    /// todo!();
    /// ```
    pub TODO,
    restriction,
    "`todo!` should not be present in production code"
}

declare_clippy_lint! {
    /// **What it does:** Checks for usage of `unreachable!`.
    ///
    /// **Why is this bad?** This macro can cause code to panic
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```no_run
    /// unreachable!();
    /// ```
    pub UNREACHABLE,
    restriction,
    "`unreachable!` should not be present in production code"
}

declare_lint_pass!(PanicUnimplemented => [PANIC_PARAMS, UNIMPLEMENTED, UNREACHABLE, TODO, PANIC]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for PanicUnimplemented {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr<'_>) {
        if_chain! {
            if let ExprKind::Block(ref block, _) = expr.kind;
            if let Some(ref ex) = block.expr;
            if let Some(params) = match_function_call(cx, ex, &paths::BEGIN_PANIC);
            if params.len() == 1;
            then {
                let expr_span = cx.tcx.hir().span(expr.hir_id);
                if is_expn_of(expr_span, "unimplemented").is_some() {
                    let span = get_outer_span(cx, expr);
                    span_lint(cx, UNIMPLEMENTED, span,
                              "`unimplemented` should not be present in production code");
                } else if is_expn_of(expr_span, "todo").is_some() {
                    let span = get_outer_span(cx, expr);
                    span_lint(cx, TODO, span,
                              "`todo` should not be present in production code");
                } else if is_expn_of(expr_span, "unreachable").is_some() {
                    let span = get_outer_span(cx, expr);
                    span_lint(cx, UNREACHABLE, span,
                              "`unreachable` should not be present in production code");
                } else if is_expn_of(expr_span, "panic").is_some() {
                    let span = get_outer_span(cx, expr);
                    span_lint(cx, PANIC, span,
                              "`panic` should not be present in production code");
                    match_panic(params, expr, cx);
                }
            }
        }
    }
}

fn get_outer_span(cx: &LateContext<'_, '_>, expr: &Expr<'_>) -> Span {
    let expr_span = cx.tcx.hir().span(expr.hir_id);
    if_chain! {
        if expr_span.from_expansion();
        let first = expr_span.ctxt().outer_expn_data();
        if first.call_site.from_expansion();
        let second = first.call_site.ctxt().outer_expn_data();
        then {
            second.call_site
        } else {
            expr_span
        }
    }
}

fn match_panic(params: &[Expr<'_>], expr: &Expr<'_>, cx: &LateContext<'_, '_>) {
    let expr_span = cx.tcx.hir().span(expr.hir_id);
    let param0_span = cx.tcx.hir().span(params[0].hir_id);
    if_chain! {
        if let ExprKind::Lit(ref lit) = params[0].kind;
        if is_direct_expn_of(expr_span, "panic").is_some();
        if let LitKind::Str(ref string, _) = lit.node;
        let string = string.as_str().replace("{{", "").replace("}}", "");
        if let Some(par) = string.find('{');
        if string[par..].contains('}');
        if param0_span.source_callee().is_none();
        if param0_span.lo() != param0_span.hi();
        then {
            span_lint(cx, PANIC_PARAMS, param0_span,
                      "you probably are missing some parameter in your format string");
        }
    }
}
