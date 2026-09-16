// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! Tests from onnx-tests' tests/add/mod.rs (burn-onnx repo)

#[allow(dead_code)]
mod add_broadcast_model {
    include!(concat!(env!("OUT_DIR"), "/model/add_broadcast.rs"));
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

fn run_broadcast_plironnx(
    x_3d: &Tensor<Backend, 3>,
    y_2d: &Tensor<Backend, 2>,
    a_2d: &Tensor<Backend, 2>,
    b_3d: &Tensor<Backend, 3>,
) -> (Tensor<Backend, 3>, Tensor<Backend, 3>) {
    let input_data = [
        x_3d.to_data(),
        y_2d.to_data(),
        a_2d.to_data(),
        b_3d.to_data(),
    ];
    let input_shapes = [vec![2, 3, 4], vec![3, 4], vec![3, 4], vec![2, 3, 4]];
    let input_descriptors: Vec<_> = input_data
        .iter()
        .zip(input_shapes)
        .map(|(data, shape)| {
            let values = data
                .as_slice::<f32>()
                .expect("input must contain f32 values");
            TensorDesciptor::new(shape, size_of::<f32>(), values.as_ptr().cast())
                .build_ir_descriptor()
        })
        .collect();

    let graph = parse_onnx_file(common::onnx_tests_dir().join("add/add_broadcast.onnx"))
        .expect("failed to parse add_broadcast.onnx");
    let ctx = &mut Context::new();
    let module = build_module(ctx, &graph, "onnx_module", "model").expect_ok(ctx);
    verify_op(&module, ctx).expect_ok(ctx);
    let (llvm_ctx, llvm_module, abi) =
        compile_packed_entrypoint(ctx, module, "model", "main").expect_ok(ctx);
    assert_eq!((abi.num_inputs, abi.num_outputs), (4, 2));
    llvm_module.verify().expect("generated LLVM IR must verify");
    let jit = SimpleJIT::new(llvm_ctx, llvm_module).expect("failed to create JIT");

    let inputs: Vec<_> = input_descriptors.iter().map(Vec::as_ptr).collect();
    let mut output1_descriptor = vec![0u8; (3 + 2 * 3) * size_of::<usize>()];
    let mut output2_descriptor = vec![0u8; (3 + 2 * 3) * size_of::<usize>()];
    let outputs = [
        output1_descriptor.as_mut_ptr(),
        output2_descriptor.as_mut_ptr(),
    ];
    let function = unsafe {
        jit.lookup_symbol::<extern "C" fn(*const *const u8, *const *mut u8) -> ()>("main")
    }
    .expect("failed to find compiled main function");
    function(inputs.as_ptr(), outputs.as_ptr());

    let read_output = |descriptor: &[u8]| unsafe {
        let descriptor =
            TensorDesciptor::from_ir_descriptor(descriptor.as_ptr(), 3, size_of::<f32>());
        let mut values: Vec<f32> = Vec::new();
        descriptor.copy_to_vec(&mut values);
        Tensor::<Backend, 1>::from_floats(values.as_slice(), &Default::default()).reshape([2, 3, 4])
    };
    (
        read_output(&output1_descriptor),
        read_output(&output2_descriptor),
    )
}

#[test]
fn add_broadcast_ir() {
    utils::assert_ir_snapshots(
        "add/add_broadcast.onnx",
        [
            "add_broadcast_onnx.expect.plir",
            "add_broadcast_tensor.expect.plir",
            "add_broadcast_llvm.expect.plir",
        ],
    );
}

#[test]
fn add_broadcast_tensor_ranks() {
    let device: Device<Backend> = Default::default();
    let x_3d = Tensor::<Backend, 3>::from_floats(
        [
            [[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]],
            [
                [13., 14., 15., 16.],
                [17., 18., 19., 20.],
                [21., 22., 23., 24.],
            ],
        ],
        &device,
    );
    let y_2d = Tensor::<Backend, 2>::from_floats(
        [[1., 1., 1., 1.], [2., 2., 2., 2.], [3., 3., 3., 3.]],
        &device,
    );
    let a_2d = Tensor::<Backend, 2>::from_floats(
        [
            [0.5, 0.5, 0.5, 0.5],
            [1.5, 1.5, 1.5, 1.5],
            [2.5, 2.5, 2.5, 2.5],
        ],
        &device,
    );
    let b_3d = x_3d.clone();
    let expected = add_broadcast_model::Model::<Backend>::default().forward(
        x_3d.clone(),
        y_2d.clone(),
        a_2d.clone(),
        b_3d.clone(),
    );
    let actual = run_broadcast_plironnx(&x_3d, &y_2d, &a_2d, &b_3d);
    actual.0.to_data().assert_eq(&expected.0.to_data(), true);
    actual.1.to_data().assert_eq(&expected.1.to_data(), true);
}
