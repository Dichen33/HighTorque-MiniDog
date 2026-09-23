from __future__ import annotations

import importlib
import importlib.metadata
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    print(f"python={sys.executable}")
    try:
        import mjlab  # noqa: F401
    except Exception as exc:
        print("mjlab=missing")
        print(f"error={type(exc).__name__}: {exc}")
        print("hint=run: uv sync --extra native --extra cu128")
        return 1

    try:
        print(f"mjlab={importlib.metadata.version('mjlab')}")
    except importlib.metadata.PackageNotFoundError:
        print("mjlab=importable")

    try:
        importlib.import_module("minidog_native")
    except Exception as exc:
        print("minidog_native=failed")
        print(f"error={type(exc).__name__}: {exc}")
        return 2

    print("minidog_native=ok")
    print("registered_tasks=Robot-Flat-v0, Robot-Rough-v0, Robot-Crawl-v0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
