"""Run all Asymmetry examples."""

from __future__ import annotations

from importlib import import_module

EXAMPLES = [
    "basic_dataset_loading",
    "transform_workflow",
    "single_fit",
    "composite_models",
    "parameter_trending",
    "logbook_usage",
    "project_files",
    "custom_loader",
]


def main() -> None:
    for name in EXAMPLES:
        print(f"\n=== Running {name}.py ===")
        module = import_module(f"examples.{name}")
        module.main()


if __name__ == "__main__":
    main()
