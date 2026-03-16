"""Register a custom loader in LoaderRegistry."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from asymmetry.core.data import MuonDataset
from asymmetry.core.io.base import BaseLoader, LoaderRegistry


class DemoTextLoader(BaseLoader):
    extensions = [".demo"]
    format_name = "Demo text (.demo)"

    def load(self, filepath: str) -> MuonDataset:
        path = Path(filepath)
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        values = np.array([float(v) for v in lines], dtype=float)
        time = np.arange(len(values), dtype=float) * 0.05
        error = np.full_like(values, 0.2)
        return MuonDataset(
            time=time,
            asymmetry=values,
            error=error,
            metadata={"run_number": 9001, "title": "Custom loader demo", "source_file": str(path)},
        )


def main() -> None:
    LoaderRegistry.register(DemoTextLoader)
    print("extensions:", LoaderRegistry.supported_extensions())

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "synthetic.demo"
        path.write_text("\n".join(["12.0", "11.4", "10.9", "10.1"]), encoding="utf-8")

        loader = LoaderRegistry.get_loader(str(path))
        dataset = loader.load(str(path))
        print(f"loaded {dataset.n_points} points from {path.name}")


if __name__ == "__main__":
    main()
