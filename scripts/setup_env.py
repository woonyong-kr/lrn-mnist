"""Create or update the Conda environment for this project."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def run(args: list[str], *, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    """Run a command and fail fast with the original exit code."""
    return subprocess.run(
        args,
        check=True,
        capture_output=capture_output,
        text=True,
    )


def conda_exists(conda_command: str) -> bool:
    """Return whether the conda executable is available on PATH."""
    return shutil.which(conda_command) is not None or Path(conda_command).exists()


def env_exists(conda_command: str, env_name: str) -> bool:
    """Check Conda environments using JSON output instead of shell tools."""
    result = run([conda_command, "env", "list", "--json"], capture_output=True)
    env_paths = json.loads(result.stdout).get("envs", [])
    return any(Path(env_path).name == env_name for env_path in env_paths)


def setup_environment(conda_command: str, env_name: str, env_file: Path) -> None:
    """Create a missing environment or update an existing one."""
    if not conda_exists(conda_command):
        print("conda was not found. Install Miniforge or Anaconda first.", file=sys.stderr)
        raise SystemExit(1)

    if not env_file.exists():
        print(f"Environment file not found: {env_file}", file=sys.stderr)
        raise SystemExit(1)

    if env_exists(conda_command, env_name):
        print(f"Updating conda environment: {env_name}")
        run([conda_command, "env", "update", "-n", env_name, "-f", str(env_file)])
    else:
        print(f"Creating conda environment: {env_name}")
        run([conda_command, "env", "create", "-n", env_name, "-f", str(env_file)])

    print(f"Done. Activate it with: conda activate {env_name}")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conda", default="conda", help="Conda executable name or path.")
    parser.add_argument("--env", default="mnist-nn", help="Conda environment name.")
    parser.add_argument("--file", default="environment.yml", help="Environment YAML file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_environment(args.conda, args.env, Path(args.file))


if __name__ == "__main__":
    main()
