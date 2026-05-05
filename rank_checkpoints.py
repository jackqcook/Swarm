from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple


def checkpoint_key(record: Dict[str, Any]) -> Tuple[float, float, float, float, float, float, float, float]:
    fixed = record["fixed_metrics"]
    general = record["generalization_metrics"]
    heldout = record.get("heldout_metrics", general)
    return (
        float(heldout["success_rate"]),
        float(heldout["completed_fraction"]),
        float(general["success_rate"]),
        float(general["completed_fraction"]),
        float(fixed["success_rate"]),
        float(fixed["completed_fraction"]),
        float(heldout["completion_auc"]),
        float(general["completion_auc"]),
        -float(heldout["remaining_assignment_cost"]),
        -float(general["remaining_assignment_cost"]),
    )


def load_summary(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def collect_runs(results_dir: Path, run_ids: List[str]) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for run_id in run_ids:
        summary_path = results_dir / run_id / "evaluation_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary: {summary_path}")
        summary = load_summary(summary_path)
        runs.append({"run_id": run_id, "summary": summary})
    return runs


def collect_ranked_checkpoints(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for run in runs:
        run_id = run["run_id"]
        for checkpoint in run["summary"].get("top_checkpoints", []):
            ranked.append(
                {
                    "run_id": run_id,
                    "episode": checkpoint["episode"],
                    "phase": checkpoint["phase"],
                    "path": checkpoint["path"],
                    "fixed_metrics": checkpoint["fixed_metrics"],
                    "generalization_metrics": checkpoint["generalization_metrics"],
                    "heldout_metrics": checkpoint.get("heldout_metrics"),
                    "key": checkpoint_key(checkpoint),
                }
            )
    ranked.sort(key=lambda item: item["key"], reverse=True)
    return ranked


def summarize_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    best_general = [run["summary"]["best_generalization_metrics"]["success_rate"] for run in runs]
    final_general = [run["summary"]["final_generalization_metrics"]["success_rate"] for run in runs]
    best_episodes = [run["summary"].get("best_episode", -1) for run in runs]
    return {
        "run_count": len(runs),
        "best_general_success_mean": sum(best_general) / max(len(best_general), 1),
        "final_general_success_mean": sum(final_general) / max(len(final_general), 1),
        "best_episode_mean": sum(best_episodes) / max(len(best_episodes), 1),
        "best_general_success_values": best_general,
        "final_general_success_values": final_general,
        "best_episode_values": best_episodes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank top checkpoints across multiple training runs.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-id", action="append", required=True, help="Run directory name under results/")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--export-best-dir", type=Path, default=None)
    return parser


def export_best_checkpoint(best_checkpoint: Dict[str, Any], export_dir: Path) -> Dict[str, str]:
    export_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(best_checkpoint["path"])
    checkpoint_target = export_dir / "promoted_model.pt"
    manifest_target = export_dir / "promotion_manifest.json"
    shutil.copy2(source_path, checkpoint_target)
    manifest = {
        "source_run_id": best_checkpoint["run_id"],
        "source_episode": best_checkpoint["episode"],
        "source_checkpoint_path": best_checkpoint["path"],
        "fixed_metrics": best_checkpoint["fixed_metrics"],
        "generalization_metrics": best_checkpoint["generalization_metrics"],
        "heldout_metrics": best_checkpoint.get("heldout_metrics"),
        "ranking_key": list(best_checkpoint["key"]),
    }
    manifest_target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "checkpoint": str(checkpoint_target),
        "manifest": str(manifest_target),
    }


def main() -> None:
    args = build_parser().parse_args()
    runs = collect_runs(args.results_dir, args.run_id)
    ranked = collect_ranked_checkpoints(runs)
    summary = summarize_runs(runs)

    output = {
        "runs": summary,
        "top_checkpoints": ranked[: max(args.top_n, 1)],
    }
    if args.export_best_dir is not None and ranked:
        output["promoted_artifact"] = export_best_checkpoint(ranked[0], args.export_best_dir)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
