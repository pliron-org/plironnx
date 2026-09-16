// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! Execute a single-input f32 ONNX model using the packed JIT entrypoint.

use burn::tensor::Tensor;
use pliron::{context::Context, op::verify_op, result::ExpectOk};
use pliron_llvm::llvm_sys::lljit::SimpleJIT;
use pliron_tensor::tensor::runtime_utils::TensorDesciptor;
use plironnx::{
    burn_onnx_ir::parse_onnx_file,
    onnx::{entrypoint_abi::compile_packed_entrypoint, from_onnx_graph::build_module},
};

use super::{Backend, common};

/// Compile a Burn test fixture with plironnx, run its packed JIT entrypoint,
/// and return the single f32 output as a Burn tensor for comparison.
///
/// `model_path` is relative to Burn's `onnx-tests/tests` directory.
///
/// This helper assumes one input and one output.
/// `output_shape` supplies the expected static dimensions
/// and may have a different rank from the input.
pub fn run_plironnx<const INPUT_D: usize, const OUTPUT_D: usize>(
    model_path: &str,
    input: &Tensor<Backend, INPUT_D>,
    output_shape: [usize; OUTPUT_D],
) -> Tensor<Backend, OUTPUT_D> {
    let input_data = input.to_data();
    let input_values = input_data
        .as_slice::<f32>()
        .expect("input must contain f32 values");
    let shape = input.dims().to_vec();

    let graph = parse_onnx_file(common::onnx_tests_dir().join(model_path))
        .expect("failed to parse ONNX test model");
    let ctx = &mut Context::new();
    let module = build_module(ctx, &graph, "onnx_module", "model").expect_ok(ctx);
    verify_op(&module, ctx).expect_ok(ctx);

    let (llvm_ctx, llvm_module, abi) =
        compile_packed_entrypoint(ctx, module, "model", "main").expect_ok(ctx);
    assert_eq!((abi.num_inputs, abi.num_outputs), (1, 1));
    llvm_module.verify().expect("generated LLVM IR must verify");
    let jit = SimpleJIT::new(llvm_ctx, llvm_module).expect("failed to create JIT");

    let input_descriptor =
        TensorDesciptor::new(shape, size_of::<f32>(), input_values.as_ptr().cast())
            .build_ir_descriptor();
    let mut output_descriptor = vec![0u8; (3 + 2 * OUTPUT_D) * size_of::<usize>()];
    let inputs = [input_descriptor.as_ptr()];
    let outputs = [output_descriptor.as_mut_ptr()];
    let function = unsafe {
        jit.lookup_symbol::<extern "C" fn(*const *const u8, *const *mut u8) -> ()>("main")
    }
    .expect("failed to find compiled main function");
    function(inputs.as_ptr(), outputs.as_ptr());

    let output_values = unsafe {
        let descriptor = TensorDesciptor::from_ir_descriptor(
            output_descriptor.as_ptr(),
            OUTPUT_D,
            size_of::<f32>(),
        );
        assert_eq!(descriptor.sizes(), &output_shape);
        let mut values: Vec<f32> = Vec::new();
        descriptor.copy_to_vec(&mut values);
        values
    };
    Tensor::<Backend, 1>::from_floats(output_values.as_slice(), &Default::default())
        .reshape(output_shape)
}
