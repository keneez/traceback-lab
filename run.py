"""Cross-platform first-run launcher for the extracted project."""
from pathlib import Path
import argparse
import importlib.util
import os
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Traceback local demo")
    parser.add_argument("--neural", action="store_true", help="Install ML dependencies and load UniXcoder")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--encoder")
    parser.add_argument("--reranker")
    parser.add_argument("--calibrator")
    parser.add_argument("--device", default="cpu")
    # Preview tooling passes this Vite option. Uvicorn already binds a strict port.
    parser.add_argument("--strictPort", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    environment = ROOT / ".venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    inside = Path(sys.prefix).resolve() == environment.resolve()
    if not inside:
        if not python.exists():
            print("Creating the project Python environment…", flush=True)
            venv.create(environment, with_pip=True)
        subprocess.check_call([str(python), str(Path(__file__).resolve()), *sys.argv[1:]], cwd=ROOT)
        return
    if importlib.util.find_spec("traceback_lab") is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r",
                               str(ROOT / "requirements.txt")], cwd=ROOT)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps",
                               "-e", str(ROOT)], cwd=ROOT)
        os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    if args.neural:
        installed = False
        if importlib.util.find_spec("torch") is None:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==2.7.1",
                                   "--index-url", "https://download.pytorch.org/whl/cpu"])
            installed = True
        if importlib.util.find_spec("transformers") is None:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", f"{ROOT}[ml]"])
            installed = True
        if installed:
            os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
        checkpoint = Path(args.encoder) if args.encoder else ROOT / "models" / "unixcoder"
        if not (checkpoint / "config.json").exists() and not (checkpoint / "current.json").exists():
            from traceback_lab.ml.encoders import download_checkpoint
            os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
            download_checkpoint("microsoft/unixcoder-base", checkpoint,
                                "5604afdc964f6c53782a6813140ade5216b99006")
        args.encoder = str(checkpoint)
    from traceback_lab.cli import main as cli
    command = ["serve", "--host", args.host, "--port", str(args.port), "--device", args.device]
    for key in ["encoder", "reranker", "calibrator"]:
        if getattr(args, key):
            command.extend(["--" + key, getattr(args, key)])
    cli(command)


if __name__ == "__main__":
    main()
