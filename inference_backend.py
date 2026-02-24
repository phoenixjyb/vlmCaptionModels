#!/usr/bin/env python3
"""
Backend-facing wrapper for caption inference.
Parses the same CLI as inference.py and forwards the call, relaying stdout/stderr and exit code.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Backend caption inference wrapper")
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", default="auto")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--device", default=None)  # optional; passed via env
    args = parser.parse_args()

    repo_dir = Path(__file__).parent
    inference_py = repo_dir / "inference.py"
    if not inference_py.exists():
        print(f"{{\"status\": \"error\", \"message\": \"inference.py not found at {inference_py}\"}}")
        sys.exit(1)

    cmd = [
        sys.executable,
        str(inference_py),
        "--provider", args.provider,
        "--model", args.model,
        "--image", args.image,
    ]
    if args.prompt:
        cmd += ["--prompt", args.prompt]

    env = os.environ.copy()
    
    # Set GPU device preference for RTX 3090
    if args.device:
        env["CAPTION_DEVICE"] = args.device
    
    # Force RTX 3090 usage if available
    env["PYTORCH_CUDA_DEVICE"] = env.get("PYTORCH_CUDA_DEVICE", "1")  # Default to RTX 3090
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # Relay output 1:1 and preserve exit code; avoid Windows codepage issues
    proc = subprocess.run(
        cmd,
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        env=env,
    )

    if proc.stdout:
        sys.stdout.write(proc.stdout)
        sys.stdout.flush()
    if proc.returncode != 0 and proc.stderr:
        sys.stderr.write(proc.stderr)
        sys.stderr.flush()

    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
