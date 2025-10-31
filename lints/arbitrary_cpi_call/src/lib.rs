#![feature(rustc_private)]
#![warn(unused_extern_crates)]
#![feature(box_patterns)]

extern crate rustc_hir;
extern crate rustc_middle;
extern crate rustc_span;

use clippy_utils::{
    diagnostics::span_lint, fn_has_unsatisfiable_preds, ty::is_type_diagnostic_item,
};

use rustc_hir::{Body as HirBody, FnDecl, def_id::LocalDefId, intravisit::FnKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::{
    mir::{BasicBlock, HasLocalDecls, Local, Operand, TerminatorKind},
    ty::{self as rustc_ty},
};
use rustc_span::{Span, Symbol};

use std::collections::{HashMap, HashSet};

mod models;
mod utils;

use models::{CpiCallsInfo, CpiContextsInfo};
use utils::*;

dylint_linting::declare_late_lint! {
    /// ### What it does
    /// Detects potential **arbitrary Cross-Program Invocations (CPIs)** where the target
    /// program ID appears to be user-controlled without validation.
    ///
    /// ### Why is this bad?
    /// Allowing user-controlled program ID in CPI calls can lead to
    /// **security vulnerabilities**, such as unauthorized fund transfers, privilege
    /// escalation, or unintended external calls. All CPI targets should be strictly
    /// validated or hardcoded to ensure safe execution.
    ///
    pub ARBITRARY_CPI_CALL,
    Warn,
    "arbitrary CPI detected — target program ID may be user-controlled"
}

impl<'tcx> LateLintPass<'tcx> for ArbitraryCpiCall {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        _kind: FnKind<'tcx>,
        _: &FnDecl<'tcx>,
        _body: &HirBody<'tcx>,
        fn_span: Span,
        def_id: LocalDefId,
    ) {
        // skip macro expansions
        if fn_span.from_expansion() {
            return;
        }
        // skip functions with unsatisfiable predicates
        if fn_has_unsatisfiable_preds(cx, def_id.to_def_id()) {
            return;
        }

        let anchor_cpi_sym = Symbol::intern("AnchorCpiContext");
        let cmp_syms = [
            Symbol::intern("cmp_partialeq_ne"),
            Symbol::intern("cmp_partialeq_eq"),
        ];
        let cmp_functions = ["::contains", "::any", "::find", "::binary_search"];

        let mir = cx.tcx.optimized_mir(def_id.to_def_id());

        // build variables assignment, reverse assignment and transitive reverse assignment maps
        let (assignment_map, reverse_assignment_map) = build_assign_and_reverse_assignment_map(mir);
        let transitive_assignment_reverse_map =
            build_transitive_reverse_map(&reverse_assignment_map);

        // Need to identify:
        // A) CPI calls
        // B) CPI contexts with user controllable program id
        // C) Conditional blocks for program id
        // Then we check all CPI contexts where a CPI call is reachable from the context
        // and the program ID is not validated in any conditional blocks

        let mut cpi_calls: HashMap<BasicBlock, CpiCallsInfo> = HashMap::new();
        let mut cpi_contexts: HashMap<BasicBlock, CpiContextsInfo> = HashMap::new();
        let mut program_id_cmp: HashMap<BasicBlock, Vec<Local>> = HashMap::new();

        for (bb, bbdata) in mir.basic_blocks.iter_enumerated() {
            if let TerminatorKind::Call {
                func: Operand::Constant(func_const),
                args,
                fn_span,
                destination,
                ..
            } = &bbdata.terminator().kind
                && let rustc_ty::FnDef(fn_def_id, _) = func_const.ty().kind()
            {
                // check if the function takes a CPI context
                if takes_cpi_context(cx, mir, args)
                    && let Some(instruction) = args.get(0)
                    && let Operand::Copy(place) | Operand::Move(place) = &instruction.node
                    && let Some(local) = place.as_local()
                    && let Some(ty) = mir.local_decls().get(local).map(|d| d.ty.peel_refs())
                    && is_type_diagnostic_item(cx, ty, anchor_cpi_sym)
                {
                    if let Some(cpi_ctx_local) = get_local_from_operand(args.get(0)) {
                        cpi_calls.insert(
                            bb,
                            CpiCallsInfo {
                                span: *fn_span,
                                local: cpi_ctx_local,
                            },
                        );
                    }
                // check if the function returns a CPI context
                } else if let fn_sig = cx.tcx.fn_sig(*fn_def_id).skip_binder()
                    && let fn_sig_unbounded = fn_sig.skip_binder()
                    && let return_ty = fn_sig_unbounded.output()
                    && is_type_diagnostic_item(cx, return_ty, anchor_cpi_sym)
                {
                    // check if CPI context with user controllable program id
                    if let Some(program_id) = args.get(0)
                        && let Operand::Copy(place) | Operand::Move(place) = &program_id.node
                        && let Some(local) = place.as_local()
                        && is_pubkey_type(cx, mir, &local)
                        && let Some(cpi_ctx_return_local) = destination.as_local()
                        && let origin =
                            origin_of_operand(cx, mir, &assignment_map, &program_id.node)
                        && let Origin::Parameter | Origin::Unknown = origin
                    {
                        cpi_contexts.insert(
                            bb,
                            CpiContextsInfo {
                                cpi_ctx_local: cpi_ctx_return_local,
                                program_id_local: local,
                            },
                        );
                    }
                } else {
                    let fn_path = cx.tcx.def_path_str(*fn_def_id);
                    // check if the function is a comparison of pubkey type
                    // TODO: Add better check for comparison functions
                    if let Some(diag_item) = cx
                        .tcx
                        .diagnostic_items(fn_def_id.krate)
                        .id_to_name
                        .get(fn_def_id)
                        && cmp_syms.contains(diag_item)
                    {
                        if let (Some(lhs_local), Some(rhs_local)) =
                            check_function_args_has_pubkey_type(cx, mir, args)
                        {
                            program_id_cmp.insert(bb, vec![lhs_local, rhs_local]);
                        }
                    // check if the function is a comparison function and one argument is pubkey type
                    } else if cmp_functions.iter().any(|suffix| fn_path.ends_with(suffix)) {
                        if let (Some(lhs_local), Some(rhs_local)) =
                            check_function_args_has_pubkey_type(cx, mir, args)
                        {
                            program_id_cmp.insert(bb, vec![lhs_local, rhs_local]);
                        }
                    }
                }
            }
        }

        // check if the CPI call is reachable from a CPI context
        // and the program ID is not validated in conditional blocks
        for (bb, cpi_ctx_info) in cpi_contexts.into_iter() {
            if let Some(cpi_call_bb) =
                cpi_invocation_is_reachable_from_cpi_context(&mir.basic_blocks, bb, &cpi_calls)
                && check_cpi_context_variables_are_same(
                    &cpi_ctx_info.cpi_ctx_local,
                    &cpi_calls[&cpi_call_bb].local,
                    &mut HashSet::new(),
                    &reverse_assignment_map,
                )
                && !check_program_id_included_in_conditional_blocks(
                    &cpi_ctx_info.program_id_local,
                    &program_id_cmp,
                    &transitive_assignment_reverse_map,
                )
            {
                span_lint(
                    cx,
                    ARBITRARY_CPI_CALL,
                    cpi_calls[&cpi_call_bb].span,
                    "arbitrary CPI detected — program id appears user-controlled",
                );
            }
        }
    }
}

#[test]
fn ui() {
    dylint_testing::ui_test(env!("CARGO_PKG_NAME"), "ui");
}
