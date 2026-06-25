from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_MANIFEST = "outputs/limit_hit_research/reports/board_candidate_strategy_manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    result = check_board_live_gate(Path(args.manifest))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(1)


def check_board_live_gate(manifest_path: Path) -> dict[str, object]:
    if not manifest_path.exists():
        return {
            "ok": False,
            "reason": f"missing board candidate manifest: {manifest_path}",
            "status": "missing",
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = str(manifest.get("status", "unknown"))
    if status != "live_candidate":
        blockers = manifest.get("promotion_gate", {}).get("blockers", [])
        return {
            "ok": False,
            "reason": f"board strategy is not live_candidate: {status}",
            "status": status,
            "blockers": blockers,
            "live_sim_policy": manifest.get("live_sim_policy", ""),
        }
    return {
        "ok": True,
        "reason": "board strategy passed live gate",
        "status": status,
        "candidate_variant": manifest.get("candidate_variant", ""),
        "candidate_rule": manifest.get("candidate_rule", {}),
    }


if __name__ == "__main__":
    main()
