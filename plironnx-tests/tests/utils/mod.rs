// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! Shared integration-test utilities.

use std::path::Path;

use expect_test::expect_file;
use pliron::{
    builtin::{op_interfaces::SingleBlockRegionInterface, ops::FuncOp as BuiltinFuncOp},
    context::Context,
    irbuild::dialect_conversion::apply_dialect_conversion,
    linked_list::ContainsLinkedList,
    op::{Op, verify_op},
    operation::Operation,
    printable::Printable,
    result::ExpectOk,
};
use pliron_common_dialects::cf::to_llvm::CFToLLVM;
use pliron_llvm::builtin_to_llvm::BuiltinToLLVMConversion;
use pliron_tensor::{
    memref::conversions::MemrefToCF,
    tensor::{bufferize::bufferize, memory_management::MallocFreeTMM},
};
use plironnx::{
    burn_onnx_ir::parse_onnx_file,
    onnx::{
        conversions::OnnxToTensor,
        entrypoint_abi::{add_packed_wrapper, lower_model_outputs},
        from_onnx_graph::build_module,
    },
};

use crate::common::onnx_tests_dir;

/// Snapshot the ONNX, tensor, and final Pliron LLVM-dialect IR for a fixture.
///
/// `input` is relative to the upstream `onnx-tests/tests` directory;
/// `expect_files` are relative to this crate's `tests/resources` directory.
pub fn assert_ir_snapshots(input: impl AsRef<Path>, expect_files: [&str; 3]) {
    let graph =
        parse_onnx_file(onnx_tests_dir().join(input)).expect("failed to parse ONNX test fixture");
    let ctx = &mut Context::new();
    let module = build_module(ctx, &graph, "onnx_module", "model").expect_ok(ctx);
    verify_op(&module, ctx).expect_ok(ctx);

    let resources = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/resources");
    expect_file![resources.join(expect_files[0])]
        .assert_eq(&module.get_operation().disp(ctx).to_string());

    let root = module.get_operation();
    apply_dialect_conversion(ctx, &mut OnnxToTensor, root).expect_ok(ctx);
    expect_file![resources.join(expect_files[1])]
        .assert_eq(&module.get_operation().disp(ctx).to_string());

    let model = module
        .get_body(ctx, 0)
        .deref(ctx)
        .iter(ctx)
        .filter_map(|op| Operation::get_op::<BuiltinFuncOp>(op, ctx))
        .next()
        .expect("ONNX test fixture must define a model function");
    let abi = lower_model_outputs(ctx, model).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut BuiltinToLLVMConversion, root).expect_ok(ctx);
    bufferize(&mut MallocFreeTMM, root, ctx).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut MemrefToCF, root).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut CFToLLVM, root).expect_ok(ctx);
    add_packed_wrapper(ctx, module, "model", "main", abi).expect_ok(ctx);
    verify_op(&module, ctx).expect_ok(ctx);
    expect_file![resources.join(expect_files[2])]
        .assert_eq(&module.get_operation().disp(ctx).to_string());
}
