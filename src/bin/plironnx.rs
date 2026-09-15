// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

use std::{path::PathBuf, process::ExitCode};

use clap::Parser;

use pliron::{
    context::Context,
    irbuild::dialect_conversion::apply_dialect_conversion,
    op::{Op, verify_op},
    printable::Printable,
    result::Result,
};

use plironnx::{
    burn_onnx_ir::parse_onnx_file,
    onnx::{conversions::OnnxToTensor, from_onnx_graph::build_module},
};

#[derive(Parser)]
#[command(version, about = "Pliron for ONNX", long_about = None)]
struct Cli {
    /// The input ONNX model file
    #[arg(value_name = "FILE", required = true, num_args = 1)]
    input: PathBuf,

    /// Print the burn-onnx graph before the conversion
    #[arg(long)]
    dump_graph: bool,

    /// Convert the `onnx` dialect to the `tensor` dialect
    #[arg(long)]
    to_tensor: bool,
}

fn run(ctx: &mut Context, cli: Cli) -> Result<()> {
    let graph = parse_onnx_file(&cli.input)?;

    if cli.dump_graph {
        println!("Graph inputs:");
        for input in &graph.inputs {
            println!("  {input}");
        }
        println!("Graph outputs:");
        for output in &graph.outputs {
            println!("  {output}");
        }
        for (i, node) in graph.nodes.iter().enumerate() {
            println!("Node#{i}: {node}");
        }
        println!();
    }

    let module = build_module(ctx, &graph, "onnx_module", "main")?;
    verify_op(&module, ctx)?;

    if cli.to_tensor {
        apply_dialect_conversion(ctx, &mut OnnxToTensor, module.get_operation())?;
        verify_op(&module, ctx)?;
    }

    println!("{}", module.get_operation().disp(ctx));

    Ok(())
}

pub fn main() -> ExitCode {
    env_logger::init();
    let cli = Cli::parse();
    let ctx = &mut Context::new();

    match run(ctx, cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{}", e.disp(ctx));
            ExitCode::FAILURE
        }
    }
}
