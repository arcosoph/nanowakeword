# ==============================================================================
#  NanoWakeWord: Lightweight, Intelligent Wake Word Detection
#  Copyright 2025 Arcosoph. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Project: https://github.com/arcosoph/nanowakeword
# ==============================================================================

"""
nanowakeword - unified CLI.

The command figures out what to do from the flags you provide.
No subcommands needed.

Training pipeline 
-----------------
OPTION 1: Using CLI flags (explicit control)
  nanowakeword -c config.yaml -G          # generate clips
  nanowakeword -c config.yaml -t          # extract features
  nanowakeword -c config.yaml -T          # train model
  nanowakeword -c config.yaml -d          # distill lite model (standalone)
  nanowakeword -c config.yaml -G -t -T -d # full pipeline in one shot
  nanowakeword -c config.yaml -T --resume ./trained_models/my_model

OPTION 2: Using config file settings (config-driven)
  nanowakeword -c config.yaml             # reads pipeline flags from config file
                                          # (generate_clips, transform_clips, train_model, distill keys)

OPTION 3: Combining both (CLI flags override config file)
  nanowakeword -c config.yaml -T          # ignores config's train_model setting, uses only -T

Server
------
  nanowakeword --model my_model.onnx                          # start server (verifier_only)
  nanowakeword --model my_model.onnx --pipeline full          # full pipeline on server
  nanowakeword --model my_model.onnx --pipeline full --port 8765 --host 0.0.0.0

Model info
----------
  nanowakeword --info my_model.onnx
"""

import os
import sys
import argparse


def _lazy_load_yaml_config(config_path: str) -> dict:
    """
    Lazily loads a YAML config file only when needed.
    This avoids requiring PyYAML as a dependency for basic CLI operations.
    
    Args:
        config_path: Path to the YAML config file.
        
    Returns:
        Dictionary with the parsed YAML content.
        
    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ImportError: If PyYAML is not installed.
        yaml.YAMLError: If the YAML is invalid.
    """
    try:
        import yaml
    except ImportError:
        print("Error: PyYAML is required to load config files.")
        print("Install it with: pip install pyyaml")
        sys.exit(1)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.load(f, yaml.Loader)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML config file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading config file: {e}")
        sys.exit(1)


def _get_pipeline_stages_from_config(config: dict) -> dict:
    """
    Extracts pipeline stage flags from the config dictionary.
    
    Args:
        config: The parsed configuration dictionary.
        
    Returns:
        Dictionary with keys: generate_clips, transform_clips, train_model, distill
        Values are booleans indicating whether each stage is enabled.
    """
    return {
        'generate_clips': config.get('generate_clips', False),
        'transform_clips': config.get('transform_clips', False),
        'train_model': config.get('train_model', False),
        'distill': config.get('distill', False),
    }


def _merge_config_with_cli_args(config_stages: dict, args) -> dict:
    """
    Merges config file pipeline settings with CLI arguments.
    CLI arguments take precedence over config file settings.
    
    Args:
        config_stages: Dictionary of pipeline stages from config file.
        args: Parsed CLI arguments.
        
    Returns:
        Dictionary with merged settings.
    """
    merged = config_stages.copy()
    
    # CLI flags override config file settings
    if args.generate_clips:
        merged['generate_clips'] = True
    if args.transform_clips:
        merged['transform_clips'] = True
    if args.train:
        merged['train_model'] = True
    if args.distill:
        merged['distill'] = True
    
    return merged


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nanowakeword",
        description="NanoWakeWord - lightweight wake word detection engine.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  nanowakeword -c config.yaml -T\n"
            "  nanowakeword -c config.yaml             # Uses pipeline flags from config\n"
            "  nanowakeword -c config.yaml -G -t -T -d\n"
            "  nanowakeword -c config.yaml -T --resume ./trained_models/my_model\n"
            "  nanowakeword -c config.yaml -d\n"
            "  nanowakeword --model my_model.onnx\n"
            "  nanowakeword --model my_model.onnx --pipeline full --port 8765\n"
            "  nanowakeword --info my_model.onnx\n"
        ),
    )

    # Training flags
    train_group = parser.add_argument_group("Training pipeline  (-c required)")

    train_group.add_argument(
        "-c", "--config",
        metavar="PATH",
        default=None,
        help="Path to the training configuration YAML file.",
    )
    train_group.add_argument(
        "-G", "--generate_clips",
        action="store_true",
        help="Generate synthetic audio clips from text (TTS).",
    )
    train_group.add_argument(
        "-t", "--transform_clips",
        action="store_true",
        help="Augment clips and extract features into .npy files.",
    )
    train_group.add_argument(
        "-T", "--train",
        action="store_true",
        help="Train the wake word model.",
    )
    train_group.add_argument(
        "-d", "--distill",
        action="store_true",
        help=(
            "Generate a lite model via knowledge distillation.\n"
            "With -T: runs after training.\n"
            "Without -T: distills from an existing trained ONNX."
        ),
    )
    train_group.add_argument(
        "-f", "--force-verify",
        action="store_true",
        help="Re-verify all data directories, ignoring the cache.",
    )
    train_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing feature files. Use with caution.",
    )
    train_group.add_argument(
        "--resume",
        metavar="PATH",
        default=None,
        help="Resume training from a project directory checkpoint.",
    )

    # Server flags
    server_group = parser.add_argument_group("Server  (--model required)")

    server_group.add_argument(
        "--model",
        metavar="PATH",
        default=None,
        help="Path to the wake word .onnx model. Starts the RemoteVerifier server.",
    )
    server_group.add_argument(
        "--pipeline",
        default="verifier_only",
        choices=["verifier_only", "full"],
        metavar="MODE",
        help=(
            "Server pipeline mode:\n"
            "  verifier_only  Edge sends pre-computed features. Server runs\n"
            "                 only the wake word model. (default)\n"
            "  full           Edge sends raw audio. Server runs the complete\n"
            "                 pipeline: mel + embedding + wake word model."
        ),
    )
    server_group.add_argument(
        "--host",
        default="0.0.0.0",
        metavar="HOST",
        help="Interface to bind to. Default: 0.0.0.0",
    )
    server_group.add_argument(
        "--port",
        default=8765,
        type=int,
        metavar="PORT",
        help="Port number. Default: 8765",
    )
    server_group.add_argument(
        "--log",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        metavar="LEVEL",
        help="Server log verbosity. Default: INFO",
    )
    server_group.add_argument(
        "--api-key",
        dest="api_keys",
        action="append",
        default=[],
        metavar="KEY",
        help="Add an API key for client authentication. Repeat to add multiple keys.",
    )
    server_group.add_argument(
        "--enable-tokens",
        action="store_true",
        help="Allow clients to exchange an API key for a short-lived access token.",
    )
    server_group.add_argument(
        "--token-ttl",
        type=int,
        default=3600,
        metavar="SECONDS",
        help="Token lifetime in seconds. Default: 3600.",
    )
    server_group.add_argument(
        "--token-secret",
        default=None,
        metavar="SECRET",
        help="Secret used to sign tokens. Auto-generated if omitted.",
    )
    server_group.add_argument(
        "--rate-limit",
        type=int,
        default=0,
        metavar="COUNT",
        help="Maximum messages per rate-window per IP. 0 disables rate limiting.",
    )
    server_group.add_argument(
        "--rate-window",
        type=int,
        default=60,
        metavar="SECONDS",
        help="Rate-limit sliding window in seconds. Default: 60.",
    )
    server_group.add_argument(
        "--ip-allowlist",
        action="append",
        default=[],
        metavar="IP_OR_CIDR",
        help="Allow only connections from this IP or CIDR. Repeat for multiple entries.",
    )
    server_group.add_argument(
        "--ssl-certfile",
        default=None,
        metavar="PATH",
        help="Path to PEM certificate file for WSS/TLS.",
    )
    server_group.add_argument(
        "--ssl-keyfile",
        default=None,
        metavar="PATH",
        help="Path to PEM private key file for WSS/TLS.",
    )
    server_group.add_argument(
        "--ssl-ca-certs",
        default=None,
        metavar="PATH",
        help="Optional CA bundle path for mutual TLS.",
    )
    server_group.add_argument(
        "--max-connections",
        type=int,
        default=0,
        metavar="COUNT",
        help="Maximum number of simultaneous client connections. 0 = unlimited.",
    )
    server_group.add_argument(
        "--ban-duration",
        type=int,
        default=300,
        metavar="SECONDS",
        help="Ban duration in seconds after rate-limit breach. 0 = no ban.",
    )

    # Info flags
    parser.add_argument(
        "--info",
        metavar="MODEL",
        default=None,
        help="Show metadata for a .onnx model file and exit.",
    )

    return parser


# Handlers

def _run_training(args, config_stages=None):
    """
    Translate flat args back to the argv list trainer.train() expects.
    
    Args:
        args: Parsed CLI arguments.
        config_stages: Optional dict of pipeline stages from config file.
                      If provided, these are merged with CLI args (CLI takes precedence).
    """
    # Merge config file settings with CLI args if config was provided
    if config_stages:
        stages = _merge_config_with_cli_args(config_stages, args)
    else:
        stages = {
            'generate_clips': args.generate_clips,
            'transform_clips': args.transform_clips,
            'train_model': args.train,
            'distill': args.distill,
        }
    
    argv = ["-c", args.config]

    if stages['generate_clips']:
        argv.append("-G")
    if stages['transform_clips']:
        argv.append("-t")
    if stages['train_model']:
        argv.append("-T")
    if stages['distill']:
        argv.append("-d")
    if args.force_verify:
        argv.append("-f")
    if args.overwrite:
        argv.append("--overwrite")
    if args.resume:
        argv += ["--resume", args.resume]

    from nanowakeword.trainer import train
    train(cli_args=argv)


def _run_server(args):
    from nanowakeword.interpreter.remote_verifier import serve
    from nanowakeword.interpreter.server_security import build_security

    security = build_security(
        api_keys=args.api_keys,
        enable_tokens=args.enable_tokens,
        token_ttl=args.token_ttl,
        token_secret=args.token_secret,
        rate_limit=args.rate_limit,
        rate_window=args.rate_window,
        ip_allowlist=args.ip_allowlist,
        ssl_certfile=args.ssl_certfile,
        ssl_keyfile=args.ssl_keyfile,
        ssl_ca_certs=args.ssl_ca_certs,
        max_connections=args.max_connections,
        ban_duration=args.ban_duration,
    )

    serve(
        model_path=args.model,
        pipeline=args.pipeline,
        host=args.host,
        port=args.port,
        log_level=args.log,
        security=security,
    )


def _run_info(model_path: str):
    if not os.path.exists(model_path):
        print(f"Error: model not found at '{model_path}'")
        sys.exit(1)

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime is required. pip install onnxruntime")
        sys.exit(1)

    import numpy as np

    sess    = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inputs  = sess.get_inputs()
    outputs = sess.get_outputs()
    name    = os.path.splitext(os.path.basename(model_path))[0]
    size_kb = os.path.getsize(model_path) / 1024

    try:
        import onnx
        proto       = onnx.load(model_path)
        total_params = sum(np.prod(list(t.dims)) for t in proto.graph.initializer)
        param_str   = f"{total_params:,}"
    except ImportError:
        param_str = "unknown  (pip install onnx for exact count)"

    is_lite      = name.endswith("_lite")
    input_names  = [i.name for i in inputs]
    is_stateful  = "hidden_in" in input_names

    print(f"\n  Model       {name}")
    print(f"  Path        {model_path}")
    print(f"  Type        {'lite / gate model' if is_lite else 'full / verifier model'}")
    print(f"  File size   {size_kb:.1f} KB")
    print(f"  Parameters  {param_str}")
    print(f"  Architecture  {'stateful (LSTM/GRU)' if is_stateful else 'stateless (DNN/CNN/Transformer)'}")
    print(f"\n  Inputs")
    for inp in inputs:
        print(f"    {inp.name:20s}  shape={inp.shape}")
    print(f"\n  Outputs")
    for out in outputs:
        print(f"    {out.name:20s}  shape={out.shape}")
    print()


# Entry point

def main():
    parser = _build_parser()
    args   = parser.parse_args()

    # --info: inspect a model and exit
    if args.info:
        _run_info(args.info)
        return

    # --model: start the server
    if args.model:
        _run_server(args)
        return

    # -c: training pipeline
    if args.config:
        # Check if any training flags are explicitly provided
        training_flags = (
            args.generate_clips or args.transform_clips or
            args.train or args.distill
        )
        
        config_stages = None
        if not training_flags:
            # No explicit flags provided; try to load config file to get defaults
            try:
                config = _lazy_load_yaml_config(args.config)
                config_stages = _get_pipeline_stages_from_config(config)
                
                # Check if config file specifies any pipeline stages
                if not any(config_stages.values()):
                    parser.error(
                        "No pipeline stages specified!\n"
                        "Provide at least one of these:\n"
                        "  CLI flags: -G, -t, -T, -d\n"
                        "  OR in config file: generate_clips, transform_clips, train_model, distill"
                    )
            except FileNotFoundError as e:
                parser.error(f"Config file not found: {args.config}\n{e}")
        
        _run_training(args, config_stages)
        return

    # Nothing actionable provided
    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
