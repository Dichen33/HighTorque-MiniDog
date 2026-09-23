from __future__ import annotations

import sys
from pathlib import Path


MJLAB_ROOT = Path(__file__).resolve().parents[1]
if str(MJLAB_ROOT) not in sys.path:
    sys.path.insert(0, str(MJLAB_ROOT))


TASK_TO_RUN_NAME = {
    "Robot-Flat-v0": "flat",
    "Robot-Rough-v0": "rough",
    "Robot-Crawl-v0": "crawl",
}


def _has_arg(args: list[str], name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in args)


def _get_arg_value(args: list[str], name: str) -> str | None:
    for index, arg in enumerate(args):
        if arg == name and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith(f"{name}="):
            return arg.split("=", 1)[1]
    return None


def _default_run_name(task: str) -> str:
    return TASK_TO_RUN_NAME.get(task, task.replace("/", "_").replace(":", "_"))


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print("usage: train_native.py <TASK> [MJLab train options]")
        print("")
        print("Example:")
        print(
            "  uv run python scripts/train_native.py Robot-Rough-v0 "
            "--env.scene.num-envs 2048 --agent.max-iterations 15000 "
            "--agent.run-name rough_v1 --agent.logger tensorboard --agent.upload-model False"
        )
        raise SystemExit(0)

    task = sys.argv[1]
    args = sys.argv[2:]
    run_name = _get_arg_value(args, "--agent.run-name") or _default_run_name(task)

    injected: list[str] = []
    if not _has_arg(args, "--agent.run-name"):
        injected += ["--agent.run-name", run_name]
    if not _has_arg(args, "--log-root"):
        injected += ["--log-root", str(MJLAB_ROOT / "model" / run_name)]
    if not _has_arg(args, "--agent.experiment-name"):
        injected += ["--agent.experiment-name", "."]

    from mjlab.scripts.train import main as mjlab_train_main

    sys.argv = ["train", task, *args, *injected]
    print(f"[MiniDog] model output root: {MJLAB_ROOT / 'model' / run_name}")
    mjlab_train_main()


if __name__ == "__main__":
    main()
