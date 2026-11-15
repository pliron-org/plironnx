// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! Tests from
//! - onnx-tests' tests/gemm/mod.rs (burn-onnx repo), and
//! - the Gemm nodes in models/gemm (see models/gemm/gemm_variants.py).

#[allow(dead_code)]
mod gemm_model {
    include!(concat!(env!("OUT_DIR"), "/model/gemm.rs"));
}
#[allow(dead_code)]
mod gemm_no_c_model {
    include!(concat!(env!("OUT_DIR"), "/model/gemm_no_c.rs"));
}
#[allow(dead_code)]
mod gemm_non_unit_alpha_beta_model {
    include!(concat!(
        env!("OUT_DIR"),
        "/model/gemm_non_unit_alpha_beta.rs"
    ));
}

use burn::tensor::{Device, Int, Tensor, TensorData, Tolerance, f16};

mod common;
#[path = "common/jit.rs"]
mod jit;
mod utils;

#[test]
fn gemm_ir() {
    utils::assert_ir_snapshots(
        "gemm/gemm_non_unit_alpha_beta.onnx",
        [
            "gemm_onnx.expect.plir",
            "gemm_tensor.expect.plir",
            "gemm_llvm.expect.plir",
        ],
    );
}

/// Run a burn Gemm test model with the matrices `a` and `b`, and the scalar `c` if it is given.
fn run_burn_gemm_model(model: &str, a: &Tensor<2>, b: &Tensor<2>, c: Option<f32>) -> Tensor<2> {
    let (a, b) = (a.to_data(), b.to_data());
    let a_descriptor = jit::build_input_descriptor::<f32>(&a);
    let b_descriptor = jit::build_input_descriptor::<f32>(&b);
    let mut inputs = vec![a_descriptor.as_ptr(), b_descriptor.as_ptr()];
    if let Some(c) = &c {
        inputs.push(std::ptr::from_ref(c).cast());
    }
    let mut output = jit::build_empty_output_descriptor(2);
    jit::run_packed(model, &inputs, &[output.as_mut_ptr()]);
    jit::read_f32_output(&output, [2, 2])
}

/// Check that Gemm with a scalar bias, without a bias, and with `alpha = beta = 0.5`
/// matches Burn.
#[test]
fn gemm() {
    let device: Device = Default::default();
    let a = Tensor::<2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
    let b = Tensor::<2>::from_floats([[5.0, 6.0], [7.0, 8.0]], &device);
    let c = 1.0;

    let expected = gemm_model::Model::new(&device).forward(a.clone(), b.clone(), c);
    let actual = run_burn_gemm_model("gemm/gemm.onnx", &a, &b, Some(c));
    actual.to_data().assert_eq(&expected.to_data(), true);

    let expected = gemm_no_c_model::Model::new(&device).forward(a.clone(), b.clone());
    let actual = run_burn_gemm_model("gemm/gemm_no_c.onnx", &a, &b, None);
    actual.to_data().assert_eq(&expected.to_data(), true);

    let expected =
        gemm_non_unit_alpha_beta_model::Model::new(&device).forward(a.clone(), b.clone(), c);
    let actual = run_burn_gemm_model("gemm/gemm_non_unit_alpha_beta.onnx", &a, &b, Some(c));
    actual.to_data().assert_eq(&expected.to_data(), true);
}

/// The product of the `[2, 3]` matrix `x` and the `[3, 2]` matrix `0, 1, ..., 5`,
/// multiplied by `alpha`.
fn scaled_product(x: &[f64], alpha: f64) -> Vec<f64> {
    let w = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    (0..2)
        .flat_map(|row| (0..2).map(move |col| (row, col)))
        .map(|(row, col)| {
            let dot: f64 = (0..3).map(|k| x[row * 3 + k] * w[k * 2 + col]).sum();
            alpha * dot
        })
        .collect()
}

/// Check that these Gemm nodes match the same computation in Burn or in [scaled_product]:
/// - f32: runtime operands transposed by `transA` and `transB`, a row bias,
///   and `alpha = 2`, `beta = -1`
/// - i32: a constant operand transposed by `transB`, a column bias,
///   and `alpha = 3`, `beta = 2`
/// - f32: a Gemm that burn fuses into Linear, with a runtime weight
/// - f32: `beta = 0`, which must ignore a bias that holds NaN and infinities
/// - f64, f16 and u32: `alpha = 0.5`, `0.5` and `3`
#[test]
fn gemm_variants() {
    let device: Device = Default::default();
    let a = Tensor::<2>::from_floats([[0.5, -1.0], [2.0, 0.25], [-3.0, 1.5]], &device);
    let b = Tensor::<2>::from_floats(
        [
            [1.0, 2.0, 3.0],
            [-1.0, 0.5, 2.0],
            [0.0, 4.0, -2.0],
            [1.5, -0.5, 1.0],
        ],
        &device,
    );
    let c = Tensor::<1>::from_floats([10.0, 20.0, 30.0, 40.0], &device);
    let p = Tensor::<2, Int>::from_ints([[1, -2, 3], [4, 5, -6]], &device);
    let w = Tensor::<1, Int>::arange(-6..6, &device).reshape([4, 3]);
    let q = Tensor::<2, Int>::from_ints([[7], [-8]], &device);
    let x = [1.0, -2.0, 3.0, 4.0, 5.0, 7.0];
    let x_u32 = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0];

    let product = a.clone().transpose().matmul(b.clone().transpose());
    let expected_f32 = product
        .clone()
        .mul_scalar(2.0)
        .sub(c.clone().unsqueeze::<2>());
    let expected_i32 = p
        .clone()
        .matmul(w.transpose())
        .mul_scalar(3)
        .add(q.clone().mul_scalar(2));
    let expected_fused = b
        .clone()
        .matmul(b.clone().transpose())
        .add(c.clone().unsqueeze::<2>());

    let input_data: [TensorData; 8] = [
        a.to_data(),
        b.to_data(),
        c.to_data(),
        p.to_data().convert::<i32>(),
        q.to_data().convert::<i32>(),
        TensorData::new(x.to_vec(), [2, 3]),
        TensorData::new(x.map(f16::from_f64).to_vec(), [2, 3]),
        TensorData::new(x_u32.map(|v| v as u32).to_vec(), [2, 3]),
    ];
    let input_descriptors = [
        jit::build_input_descriptor::<f32>(&input_data[0]),
        jit::build_input_descriptor::<f32>(&input_data[1]),
        jit::build_input_descriptor::<f32>(&input_data[2]),
        jit::build_input_descriptor::<i32>(&input_data[3]),
        jit::build_input_descriptor::<i32>(&input_data[4]),
        jit::build_input_descriptor::<f64>(&input_data[5]),
        jit::build_input_descriptor::<f16>(&input_data[6]),
        jit::build_input_descriptor::<u32>(&input_data[7]),
    ];
    let mut outputs = [(); 7].map(|()| jit::build_empty_output_descriptor(2));
    jit::run_packed(
        common::local_models_dir().join("gemm/gemm_variants.onnx"),
        &input_descriptors
            .each_ref()
            .map(|descriptor| descriptor.as_ptr()),
        &outputs.each_mut().map(|descriptor| descriptor.as_mut_ptr()),
    );

    let tolerance = Tolerance::<f32>::default();
    jit::read_f32_output(&outputs[0], [2, 4])
        .to_data()
        .assert_approx_eq(&expected_f32.to_data(), tolerance);
    assert_eq!(
        jit::read_output::<i32>(&outputs[1], &[2, 4]),
        expected_i32
            .to_data()
            .convert::<i32>()
            .as_slice::<i32>()
            .unwrap(),
    );
    jit::read_f32_output(&outputs[2], [4, 4])
        .to_data()
        .assert_approx_eq(&expected_fused.to_data(), tolerance);
    jit::read_f32_output(&outputs[3], [2, 4])
        .to_data()
        .assert_approx_eq(&product.to_data(), tolerance);
    assert_eq!(
        jit::read_output::<f64>(&outputs[4], &[2, 2]),
        scaled_product(&x, 0.5)
    );
    assert_eq!(
        jit::read_output::<f16>(&outputs[5], &[2, 2]),
        scaled_product(&x, 0.5)
            .into_iter()
            .map(f16::from_f64)
            .collect::<Vec<_>>(),
    );
    assert_eq!(
        jit::read_output::<u32>(&outputs[6], &[2, 2]),
        scaled_product(&x_u32, 3.0)
            .into_iter()
            .map(|v| v as u32)
            .collect::<Vec<_>>(),
    );
}
