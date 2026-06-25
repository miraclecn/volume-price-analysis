from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe xtquant 1-minute K-line access in an isolated output directory.")
    parser.add_argument("--stock-code", default="000001.SZ")
    parser.add_argument("--start-time", default="20240603093000")
    parser.add_argument("--end-time", default="20240603150000")
    parser.add_argument("--period", default="1m")
    parser.add_argument("--out-dir", default="outputs/xuntou_probe")
    parser.add_argument("--skip-download", action="store_true", help="Only query local xtquant data cache.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {
        "ok": False,
        "stock_code": args.stock_code,
        "period": args.period,
        "start_time": args.start_time,
        "end_time": args.end_time,
        "output_dir": str(out_dir),
    }

    try:
        from xtquant import xtdata

        status["xtdata_file"] = getattr(xtdata, "__file__", "")
        if not args.skip_download:
            xtdata.download_history_data(args.stock_code, args.period, args.start_time, args.end_time)
            status["download_attempted"] = True
        else:
            status["download_attempted"] = False

        data = xtdata.get_market_data_ex(
            [],
            [args.stock_code],
            args.period,
            start_time=args.start_time,
            end_time=args.end_time,
            count=-1,
            dividend_type="none",
            fill_data=False,
        )
        frame = data.get(args.stock_code)
        if frame is None:
            raise RuntimeError(f"xtdata returned no frame for {args.stock_code}; keys={list(data.keys())}")

        rows = int(len(frame))
        csv_path = out_dir / f"{args.stock_code.replace('.', '_')}_{args.period}_{args.start_time}_{args.end_time}.csv"
        frame.to_csv(csv_path)
        status.update(
            {
                "ok": rows > 0,
                "rows": rows,
                "columns": list(frame.columns),
                "csv": str(csv_path),
                "head": frame.head(3).reset_index().astype(str).to_dict(orient="records"),
            }
        )
        if rows == 0:
            status["error"] = "query succeeded but returned zero rows"
    except Exception as exc:
        status["error"] = repr(exc)
        status["traceback"] = traceback.format_exc()

    status_path = out_dir / "xtquant_minute_probe_status.json"
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps(status, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if status["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
