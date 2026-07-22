// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

use std::{path::PathBuf, process::ExitCode};

use clap::Parser;

use onnx_ir::Node;
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
    for node in &graph.nodes {
        println!("Node: {}", node.name());

        // Pattern match on node type to access operation-specific configuration
        match node {
            Node::Softmax(softmax_node) => {
                println!("  Softmax on axis {}", softmax_node.config.axis);
            }
            Node::Conv2d(conv_node) => {
                println!(
                    "  Conv2d with kernel size {:?}",
                    conv_node.config.kernel_size
                );
            }
            _ => {}
        }
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
