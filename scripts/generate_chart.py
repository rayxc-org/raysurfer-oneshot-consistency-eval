#!/usr/bin/env python3
"""Generate a simple SVG benchmark chart from eval summary JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Generate consistency comparison SVG chart")
    parser.add_argument("--summary", type=Path, required=True, help="Summary JSON from score_eval.py")
    parser.add_argument("--out", type=Path, required=True, help="Output SVG path")
    return parser.parse_args()


def format_pct(value: float) -> str:
    """Format float ratio as percentage string."""
    return f"{value * 100:.1f}%"


def main() -> None:
    """Build chart SVG from summary metrics."""
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))

    raysurfer = summary.get("raysurfer")
    baseline = summary.get("baseline")
    if not isinstance(raysurfer, dict) or not isinstance(baseline, dict):
        raise ValueError("summary JSON must include both 'raysurfer' and 'baseline'")

    rs_consistency = float(raysurfer["overall_consistency"])
    base_consistency = float(baseline["overall_consistency"])
    delta = float(baseline.get("delta", rs_consistency - base_consistency))
    rs_attempts = int(raysurfer["total_attempts"])
    base_attempts = int(baseline["total_attempts"])

    width = 980
    height = 560
    chart_x = 120
    chart_y = 120
    chart_w = 740
    chart_h = 300
    bar_w = 220

    base_bar_h = int(chart_h * base_consistency)
    rs_bar_h = int(chart_h * rs_consistency)

    base_x = chart_x + 110
    rs_x = chart_x + 410
    base_y = chart_y + chart_h - base_bar_h
    rs_y = chart_y + chart_h - rs_bar_h

    title = "Raysurfer One-Shot 3-Minute Consistency"
    subtitle = "20-task eval, 1 trial per task (baseline vs cached Raysurfer run)"

    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>
  <defs>
    <linearGradient id='bg' x1='0' y1='0' x2='1' y2='1'>
      <stop offset='0%' stop-color='#f8fafc'/>
      <stop offset='100%' stop-color='#eef2ff'/>
    </linearGradient>
    <linearGradient id='barRs' x1='0' y1='0' x2='0' y2='1'>
      <stop offset='0%' stop-color='#16a34a'/>
      <stop offset='100%' stop-color='#15803d'/>
    </linearGradient>
    <linearGradient id='barBase' x1='0' y1='0' x2='0' y2='1'>
      <stop offset='0%' stop-color='#94a3b8'/>
      <stop offset='100%' stop-color='#64748b'/>
    </linearGradient>
  </defs>

  <rect x='0' y='0' width='{width}' height='{height}' fill='url(#bg)'/>

  <text x='60' y='58' font-family='Helvetica, Arial, sans-serif' font-size='34' font-weight='700' fill='#0f172a'>{title}</text>
  <text x='60' y='92' font-family='Helvetica, Arial, sans-serif' font-size='18' fill='#334155'>{subtitle}</text>

  <line x1='{chart_x}' y1='{chart_y + chart_h}' x2='{chart_x + chart_w}' y2='{chart_y + chart_h}' stroke='#334155' stroke-width='2'/>
  <line x1='{chart_x}' y1='{chart_y}' x2='{chart_x}' y2='{chart_y + chart_h}' stroke='#334155' stroke-width='2'/>

  <text x='{chart_x - 40}' y='{chart_y + chart_h + 6}' font-family='Helvetica, Arial, sans-serif' font-size='14' fill='#475569'>0%</text>
  <text x='{chart_x - 48}' y='{chart_y + chart_h * 0.75 + 6}' font-family='Helvetica, Arial, sans-serif' font-size='14' fill='#475569'>25%</text>
  <text x='{chart_x - 48}' y='{chart_y + chart_h * 0.5 + 6}' font-family='Helvetica, Arial, sans-serif' font-size='14' fill='#475569'>50%</text>
  <text x='{chart_x - 48}' y='{chart_y + chart_h * 0.25 + 6}' font-family='Helvetica, Arial, sans-serif' font-size='14' fill='#475569'>75%</text>
  <text x='{chart_x - 55}' y='{chart_y + 6}' font-family='Helvetica, Arial, sans-serif' font-size='14' fill='#475569'>100%</text>

  <rect x='{base_x}' y='{base_y}' width='{bar_w}' height='{base_bar_h}' rx='12' fill='url(#barBase)'/>
  <rect x='{rs_x}' y='{rs_y}' width='{bar_w}' height='{rs_bar_h}' rx='12' fill='url(#barRs)'/>

  <text x='{base_x + bar_w / 2}' y='{base_y - 12}' text-anchor='middle' font-family='Helvetica, Arial, sans-serif' font-size='22' font-weight='700' fill='#1e293b'>{format_pct(base_consistency)}</text>
  <text x='{rs_x + bar_w / 2}' y='{rs_y - 12}' text-anchor='middle' font-family='Helvetica, Arial, sans-serif' font-size='22' font-weight='700' fill='#14532d'>{format_pct(rs_consistency)}</text>

  <text x='{base_x + bar_w / 2}' y='{chart_y + chart_h + 34}' text-anchor='middle' font-family='Helvetica, Arial, sans-serif' font-size='18' fill='#334155'>Baseline</text>
  <text x='{rs_x + bar_w / 2}' y='{chart_y + chart_h + 34}' text-anchor='middle' font-family='Helvetica, Arial, sans-serif' font-size='18' fill='#14532d'>With Raysurfer</text>

  <text x='{base_x + bar_w / 2}' y='{chart_y + chart_h + 58}' text-anchor='middle' font-family='Helvetica, Arial, sans-serif' font-size='14' fill='#64748b'>{base_attempts} attempts</text>
  <text x='{rs_x + bar_w / 2}' y='{chart_y + chart_h + 58}' text-anchor='middle' font-family='Helvetica, Arial, sans-serif' font-size='14' fill='#166534'>{rs_attempts} attempts</text>

  <rect x='60' y='470' width='860' height='58' rx='12' fill='#0f172a'/>
  <text x='86' y='506' font-family='Helvetica, Arial, sans-serif' font-size='23' font-weight='700' fill='#f8fafc'>Consistency uplift within 3 minutes: {format_pct(delta)}</text>
</svg>
"""

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(svg, encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
