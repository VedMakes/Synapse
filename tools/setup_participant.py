"""Create/check the local SYNAPSE participant Python environment.

The script keeps dependencies inside the project `.venv`. It is intentionally
small and explicit so a participant who starts the UI with system Python gets a
clear setup path instead of a raw ``ModuleNotFoundError``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import venv


ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = ROOT / ".venv"
VENV_PYTHON = VENV_DIR / "bin" / "python"
REQUIREMENTS = ROOT / "requirements.lock"


def _python_has_dependencies(python: Path) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            [str(python), "-c", "import numpy, torch; print(f'numpy={numpy.__version__} torch={torch.__version__}')"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    detail = (result.stdout or result.stderr).strip()
    return result.returncode == 0, detail


def _bootstrap_python() -> Path:
    configured = shutil.which("python3.11") or sys.executable
    return Path(configured).resolve()


def setup_environment() -> str:
    if VENV_PYTHON.exists():
        ready, detail = _python_has_dependencies(VENV_PYTHON)
        if ready:
            return f"SYNAPSE environment ready: {detail}"
    bootstrap = _bootstrap_python()
    print(f"Creating project environment with {bootstrap} ...")
    builder = venv.EnvBuilder(with_pip=True, clear=False)
    builder.create(VENV_DIR)
    if not VENV_PYTHON.exists():
        raise RuntimeError(f"virtual environment did not create {VENV_PYTHON}")
    print("Installing pinned SYNAPSE dependencies ...")
    subprocess.run(
        [str(VENV_PYTHON), "-m", "pip", "install", "--requirement", str(REQUIREMENTS)],
        cwd=ROOT,
        check=True,
    )
    ready, detail = _python_has_dependencies(VENV_PYTHON)
    if not ready:
        raise RuntimeError(f"dependency check failed: {detail}")
    return f"SYNAPSE environment ready: {detail}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="only check the project environment")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check:
        if not VENV_PYTHON.exists():
            raise SystemExit(f"SYNAPSE environment is missing. Run: {sys.executable} tools/setup_participant.py")
        ready, detail = _python_has_dependencies(VENV_PYTHON)
        if not ready:
            raise SystemExit(f"SYNAPSE environment is incomplete: {detail}\nRun: {sys.executable} tools/setup_participant.py")
        print(f"SYNAPSE environment ready: {detail}")
        return
    try:
        print(setup_environment())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(
            "SYNAPSE setup failed. Check network access and Python 3.11, then retry "
            f"'{sys.executable} tools/setup_participant.py'. Details: {exc}"
        ) from exc


if __name__ == "__main__":
    main()
