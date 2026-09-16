// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! Test from onnx-tests' tests/mul/mod.rs (burn-onnx repo)

#[allow(dead_code)]
mod mul_model {
    include!(concat!(env!("OUT_DIR"), "/model/mul.rs"));
}

use burn::tensor::{Device, Tensor};
use pliron::{context::Context, op::verify_op, result::ExpectOk};
use pliron_llvm::llvm_sys::lljit::SimpleJIT;
use pliron_tensor::tensor::runtime_utils::TensorDesciptor;
use plironnx::{
    burn_onnx_ir::parse_onnx_file,
    onnx::{entrypoint_abi::compile_packed_entrypoint, from_onnx_graph::build_module},
};

mod common;
mod utils;

type Backend = burn::backend::Flex;

fn run_plironnx(input: Tensor<Backend, 4>, scalar: f64) -> Tensor<Backend, 4> {
    let input_data = input.to_data();
    let input_values = input_data
        .as_slice::<f32>()
        .expect("input must contain f32 values");
    let graph = parse_onnx_file(common::onnx_tests_dir().join("mul/mul.onnx"))
        .expect("failed to parse mul.onnx");
    let ctx = &mut Context::new();
    let module = build_module(ctx, &graph, "onnx_module", "model").expect_ok(ctx);
    verify_op(&module, ctx).expect_ok(ctx);

    let (llvm_ctx, llvm_module, abi) =
        compile_packed_entrypoint(ctx, module, "model", "main").expect_ok(ctx);
    assert_eq!((abi.num_inputs, abi.num_outputs), (2, 1));
    llvm_module.verify().expect("generated LLVM IR must verify");
    let jit = SimpleJIT::new(llvm_ctx, llvm_module).expect("failed to create JIT");

    let input_descriptor = TensorDesciptor::new(
        vec![1, 2, 3, 4],
        size_of::<f32>(),
        input_values.as_ptr().cast(),
    )
    .build_ir_descriptor();
    // Descriptor: 3 fixed fields plus size and stride for each of 4 dimensions.
    let mut output_descriptor = vec![0u8; (3 + 2 * 4) * size_of::<usize>()];
    let inputs = [input_descriptor.as_ptr(), (&raw const scalar).cast::<u8>()];
    let outputs = [output_descriptor.as_mut_ptr()];
    let function = unsafe {
        jit.lookup_symbol::<extern "C" fn(*const *const u8, *const *mut u8) -> ()>("main")
    }
    .expect("failed to find compiled main function");
    function(inputs.as_ptr(), outputs.as_ptr());

    let output_values = unsafe {
        let descriptor =
            TensorDesciptor::from_ir_descriptor(output_descriptor.as_ptr(), 4, size_of::<f32>());
        let mut values: Vec<f32> = Vec::new();
        descriptor.copy_to_vec(&mut values);
        values
    };
    Tensor::<Backend, 1>::from_floats(output_values.as_slice(), &Default::default())
        .reshape([1, 2, 3, 4])
}

#[test]
fn mul_ir() {
    utils::assert_ir_snapshots(
        "mul/mul.onnx",
        [
            "mul_onnx.expect.plir",
            "mul_tensor.expect.plir",
            "mul_llvm.expect.plir",
        ],
    );
}

#[test]
fn mul_scalar_with_tensor_and_tensor_with_tensor() {
    let device: Device<Backend> = Default::default();
    let input = Tensor::<Backend, 4>::from_floats(
        [[
            [[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]],
            [
                [13., 14., 15., 16.],
                [17., 18., 19., 20.],
                [21., 22., 23., 24.],
            ],
        ]],
        &device,
    );
    let scalar = 6.0f64;
    let expected = mul_model::Model::<Backend>::default().forward(input.clone(), scalar);
    let actual = run_plironnx(input, scalar);
    actual.to_data().assert_eq(&expected.to_data(), true);
}
