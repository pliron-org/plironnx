// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

use std::{path::PathBuf, process::ExitCode};

use clap::Parser;

use plironnx::burn_onnx_ir::parse_onnx_file;

#[derive(Parser)]
#[command(version, about = "Pliron for ONNX", long_about = None)]
struct Cli {
    /// Input ONNX model file
    #[arg(value_name = "FILE", required = true, num_args = 1)]
    input: PathBuf,
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    let graph = parse_onnx_file(&cli.input)?;
    for (i, node) in graph.nodes.iter().enumerate() {
        println!("Node#{}: {}", i, node);
    }
    Ok(())
}

pub fn main() -> ExitCode {
    env_logger::init();
    let cli = Cli::parse();

    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
