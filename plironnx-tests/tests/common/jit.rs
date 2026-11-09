// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! Compile an ONNX test model with plironnx and call its packed JIT entry point.

// Each test crate includes this module and uses only some of its helpers.
#![allow(dead_code)]

use burn::tensor::{Tensor, TensorData};
use pliron::{
    builtin::ops::ModuleOp, context::Context, irfmt::parsers::spaced, op::verify_op,
    operation::Operation, parsable::parse_from_str, result::ExpectOk,
};
use pliron_llvm::llvm_sys::lljit::SimpleJIT;
use pliron_tensor::tensor::runtime_utils::TensorDesciptor;
use plironnx::{
    burn_onnx_ir::parse_onnx_file,
    onnx::{entrypoint_abi::compile_packed_entrypoint, from_onnx_graph::build_module},
};

use super::common;

/// Compile `model_path` (relative to [common::onnx_tests_dir]) and
/// call its packed entry point once.
///
/// Each entry of `inputs` points to a tensor descriptor or to a scalar value.
/// Each entry of `outputs` points to storage for a tensor descriptor (see
/// [`build_empty_output_descriptor`]) or for a scalar value.
pub fn run_packed(model_path: &str, inputs: &[*const u8], outputs: &[*mut u8]) {
    let graph = parse_onnx_file(common::onnx_tests_dir().join(model_path))
        .expect("failed to parse ONNX test model");
    let ctx = &mut Context::new();
    let module = build_module(ctx, &graph, "onnx_module", "model").expect_ok(ctx);
    run_packed_module(ctx, module, inputs, outputs);
}

/// Parse `onnx` dialect IR text, compile its function `@model`, and call its
/// packed entry point once.
///
/// Each entry of `inputs` points to a tensor descriptor or to a scalar value.
/// Each entry of `outputs` points to storage for a tensor descriptor (see
/// [`build_empty_output_descriptor`]) or for a scalar value.
pub fn run_packed_ir(ir: &str, inputs: &[*const u8], outputs: &[*mut u8]) {
    let ctx = &mut Context::new();
    let module = parse_from_str(spaced(Operation::top_level_parser()), ctx, ir).expect_ok(ctx);
    let module = Operation::get_op::<ModuleOp>(module, ctx).expect("IR must be a builtin.module");
    run_packed_module(ctx, module, inputs, outputs);
}

/// Verify `module`, compile its function `@model`, and call the packed entry point once.
fn run_packed_module(
    ctx: &mut Context,
    module: ModuleOp,
    inputs: &[*const u8],
    outputs: &[*mut u8],
) {
    verify_op(&module, ctx).expect_ok(ctx);

    let (llvm_ctx, llvm_module, abi) =
        compile_packed_entrypoint(ctx, module, "model", "main").expect_ok(ctx);
    assert_eq!(
        (abi.num_inputs, abi.num_outputs),
        (inputs.len(), outputs.len())
    );
    llvm_module.verify().expect("generated LLVM IR must verify");
    let jit = SimpleJIT::new(llvm_ctx, llvm_module).expect("failed to create JIT");
    let function = unsafe {
        jit.lookup_symbol::<extern "C" fn(*const *const u8, *const *mut u8) -> ()>("main")
    }
    .expect("failed to find compiled main function");
    function(inputs.as_ptr(), outputs.as_ptr());
}

/// Build the input descriptor of a Burn tensor. The descriptor points into
/// `data`, so `data` must stay alive until the call is complete.
pub fn build_input_descriptor<T: burn::tensor::Element>(data: &TensorData) -> Vec<u8> {
    let values = data.as_slice::<T>().expect("unexpected input element type");
    TensorDesciptor::new(
        data.shape().to_vec(),
        size_of::<T>(),
        values.as_ptr().cast(),
    )
    .build_ir_descriptor()
}

/// Zeroed storage for an output descriptor with `rank` dimensions:
/// - three fixed fields: allocated pointer, aligned pointer and offset (in elements)
/// - one size and one stride for each dimension.
pub fn build_empty_output_descriptor(rank: usize) -> Vec<u8> {
    vec![0u8; (3 + 2 * rank) * size_of::<usize>()]
}

/// Read the elements of an output descriptor and check its shape.
pub fn read_output<T: Copy>(descriptor: &[u8], shape: &[usize]) -> Vec<T> {
    unsafe {
        let descriptor =
            TensorDesciptor::from_ir_descriptor(descriptor.as_ptr(), shape.len(), size_of::<T>());
        assert_eq!(descriptor.sizes(), shape);
        // `TensorDesciptor::copy_to_vec` does not support rank 0.
        if shape.is_empty() {
            let element = descriptor
                .aligned_ptr()
                .cast::<T>()
                .add(descriptor.offset());
            return vec![std::ptr::read_unaligned(element)];
        }
        let mut values = Vec::new();
        descriptor.copy_to_vec(&mut values);
        values
    }
}

/// Read an f32 output descriptor as a Burn tensor.
pub fn read_f32_output<const D: usize>(descriptor: &[u8], shape: [usize; D]) -> Tensor<D> {
    let values = read_output::<f32>(descriptor, &shape);
    Tensor::<1>::from_floats(values.as_slice(), &Default::default()).reshape(shape)
}

/// Compile and run a test model with one f32 tensor input and one f32 tensor output.
/// `output_shape` (of any rank) specifies the expected static output dimensions.
pub fn run_single_f32<const INPUT_D: usize, const OUTPUT_D: usize>(
    model_path: &str,
    input: &Tensor<INPUT_D>,
    output_shape: [usize; OUTPUT_D],
) -> Tensor<OUTPUT_D> {
    let input_data = input.to_data();
    let input_descriptor = build_input_descriptor::<f32>(&input_data);
    let mut output = build_empty_output_descriptor(OUTPUT_D);
    run_packed(
        model_path,
        &[input_descriptor.as_ptr()],
        &[output.as_mut_ptr()],
    );
    read_f32_output(&output, output_shape)
}
