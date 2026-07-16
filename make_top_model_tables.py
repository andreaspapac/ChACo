#!/usr/bin/env python3
"""Build per-dataset, per-stem layer tables for the best model at each SG setting.

This script scans the summary CSV files in ``Summaries/ResNet`` and ``Summaries/WAN``
(default) and, for each ``(dataset, stem_arch, num_supergroup_layers)`` group,
selects the top-performing configuration according to the lowest deepest-layer
``test_mean``. It then emits tables that show the per-layer train/test
``mean ± std`` values for each selected configuration, arranged by SG setting.

Outputs:
- one flattened CSV per (dataset, stem_arch)
- one Markdown report collecting all tables

The Markdown layout is intended to resemble the spreadsheet-style example shared
by the user, with a channels column, layer labels, and grouped columns for each
SG setting.
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

LAYER_TEST_MEAN_RE = re.compile(r"^L(\d+)_test_mean$")


def find_input_files(input_roots: Sequence[Path]) -> List[Path]:
    files: List[Path] = []
    for root in input_roots:
        if not root.exists():
            continue
        files.extend(sorted(root.glob("*_summary_metrics_gen.csv")))
    return files



def deepest_layer_index(fieldnames: Iterable[str]) -> int:
    layers = [int(match.group(1)) for name in fieldnames if (match := LAYER_TEST_MEAN_RE.match(name))]
    if not layers:
        raise ValueError("No layer metrics matching 'Lk_test_mean' were found.")
    return max(layers)



def parse_channels(raw: str) -> List[str]:
    try:
        values = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(values, list):
        return []
    return [str(v) for v in values]



def parse_float(raw: str) -> float:
    if raw is None:
        return math.nan
    raw = str(raw).strip()
    if not raw or raw.lower() == "nan":
        return math.nan
    return float(raw)



def sg_label(num_layers: int, sg_layers: int) -> str:
    if sg_layers <= 0:
        return "All CwC"
    if sg_layers >= num_layers:
        return f"L1-L{num_layers}: SG"
    if sg_layers == 1:
        return f"L1: SG; L2-L{num_layers}: CwC"
    return f"L1-L{sg_layers}: SG; L{sg_layers + 1}-L{num_layers}: CwC"



def load_rows(input_files: Sequence[Path]) -> List[dict]:
    rows: List[dict] = []
    for path in input_files:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            deepest = deepest_layer_index(fieldnames)
            for row in reader:
                row = dict(row)
                row["_source_file"] = path.as_posix()
                row["_deepest_layer"] = deepest
                row["_rank_score"] = parse_float(row.get(f"L{deepest}_test_mean", "nan"))
                row["_num_supergroup_layers_int"] = int(parse_float(row.get("num_supergroup_layers", "nan")))
                row["_channels"] = parse_channels(row.get("out_channels_list", ""))
                rows.append(row)
    return rows



def select_best_by_sg(rows: Sequence[dict]) -> Dict[Tuple[str, str, int], dict]:
    best: Dict[Tuple[str, str, int], dict] = {}
    for row in rows:
        key = (
            row.get("dataset", ""),
            row.get("stem_arch", ""),
            row.get("_num_supergroup_layers_int", -1),
        )
        current = best.get(key)
        if current is None or row["_rank_score"] < current["_rank_score"]:
            best[key] = row
    return best



def build_table_rows(selected_rows: Sequence[dict]) -> Tuple[List[str], List[dict]]:
    ordered = sorted(selected_rows, key=lambda row: row["_num_supergroup_layers_int"])
    deepest = max(row["_deepest_layer"] for row in ordered)
    channels = ordered[0]["_channels"]
    body_rows: List[dict] = []
    sg_headers = [sg_label(deepest, row["_num_supergroup_layers_int"]) for row in ordered]

    for layer_idx in range(1, deepest + 1):
        channel_value = channels[layer_idx - 1] if layer_idx - 1 < len(channels) else ""
        record = {
            "Channels": channel_value,
            "Layer": f"L{layer_idx}",
        }
        for row, header in zip(ordered, sg_headers):
            train_mean = parse_float(row.get(f"L{layer_idx}_train_mean", "nan"))
            train_std = parse_float(row.get(f"L{layer_idx}_train_std", "nan"))
            test_mean = parse_float(row.get(f"L{layer_idx}_test_mean", "nan"))
            test_std = parse_float(row.get(f"L{layer_idx}_test_std", "nan"))
            record[f"{header}__Train"] = format_metric(train_mean, train_std)
            record[f"{header}__Test"] = format_metric(test_mean, test_std)
        body_rows.append(record)

    return sg_headers, body_rows



def format_metric(mean: float, std: float, decimals: int = 3) -> str:
    if math.isnan(mean):
        return ""
    if math.isnan(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"



def write_flat_csv(path: Path, headers: List[str], rows: List[dict]) -> None:
    fieldnames = ["Channels", "Layer"]
    for header in headers:
        fieldnames.append(f"{header}__Train")
        fieldnames.append(f"{header}__Test")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def write_markdown_report(path: Path, grouped_tables: Dict[Tuple[str, str], Tuple[List[str], List[dict], List[dict]]]) -> None:
    lines: List[str] = ["# Top-performing models by SuperGroup setting", ""]
    for dataset, stem in sorted(grouped_tables):
        headers, table_rows, selected = grouped_tables[(dataset, stem)]
        lines.append(f"## {dataset} — {stem}")
        lines.append("")
        lines.append("Selected configuration per `num_supergroup_layers` by lowest deepest-layer `test_mean`.")
        lines.append("")
        lines.append("### Selected models")
        lines.append("")
        lines.append("| SG setting | num_supergroup_layers | set_id | deepest-layer test mean | source |")
        lines.append("| --- | ---: | --- | ---: | --- |")
        for row in sorted(selected, key=lambda item: item["_num_supergroup_layers_int"]):
            header = sg_label(row["_deepest_layer"], row["_num_supergroup_layers_int"])
            lines.append(
                "| {header} | {sg} | {set_id} | {score:.3f} | `{source}` |".format(
                    header=header,
                    sg=row["_num_supergroup_layers_int"],
                    set_id=row.get("set_id", ""),
                    score=row["_rank_score"],
                    source=Path(row["_source_file"]).name,
                )
            )
        lines.append("")
        lines.append("### Layer-wise table")
        lines.append("")

        header_row = ["Channels", "Layer"]
        value_keys = ["Channels", "Layer"]
        separator_row = ["---", "---"]
        for header in headers:
            header_row.extend([f"{header} Train", f"{header} Test"])
            value_keys.extend([f"{header}__Train", f"{header}__Test"])
            separator_row.extend(["---", "---"])
        lines.append("| " + " | ".join(header_row) + " |")
        lines.append("| " + " | ".join(separator_row) + " |")
        for row in table_rows:
            values = [row.get(column, "") for column in value_keys]
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        action="append",
        default=None,
        help="Input folder containing *_summary_metrics_gen.csv files. May be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        default="Summaries/top_model_tables",
        help="Directory where the generated CSV and Markdown files will be written.",
    )
    args = parser.parse_args()

    input_roots = [Path(p) for p in (args.input_root or ["Summaries/ResNet", "Summaries/WAN"])]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = find_input_files(input_roots)
    if not input_files:
        raise SystemExit("No summary CSV files were found in the requested input roots.")

    rows = load_rows(input_files)
    best = select_best_by_sg(rows)

    grouped_selected: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for (dataset, stem, _sg), row in best.items():
        grouped_selected[(dataset, stem)].append(row)

    report_payload: Dict[Tuple[str, str], Tuple[List[str], List[dict], List[dict]]] = {}

    for (dataset, stem), selected_rows in sorted(grouped_selected.items()):
        headers, table_rows = build_table_rows(selected_rows)
        csv_name = f"{dataset}_{stem}_top_model_by_supergroup_table.csv"
        write_flat_csv(output_dir / csv_name, headers, table_rows)
        report_payload[(dataset, stem)] = (headers, table_rows, selected_rows)

    write_markdown_report(output_dir / "top_model_tables.md", report_payload)


if __name__ == "__main__":
    main()
