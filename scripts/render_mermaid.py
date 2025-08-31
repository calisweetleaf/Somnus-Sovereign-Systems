#!/usr/bin/env python3
"""
Render Mermaid code blocks found in Markdown docs to PNG/SVG using mermaid-cli (mmdc).

Usage:
  python scripts/render_mermaid.py --format png --theme default

Requirements:
  - NodeJS
  - npm install -g @mermaid-js/mermaid-cli@10

Behavior:
  - Scans docs/diagrams/*.md for ```mermaid code fences
  - Writes .mmd sources to docs/diagrams/mmd/<basename>__block<N>.mmd
  - Renders images to docs/diagrams/img/<basename>__block<N>.(png|svg)

Notes:
  - If mmdc is not available, prints instructions and exits with non-zero status.
"""

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAGRAMS_DIR = ROOT / "docs" / "diagrams"
MMD_DIR = DIAGRAMS_DIR / "mmd"
IMG_DIR = DIAGRAMS_DIR / "img"
PUPPETEER_CONFIG = ROOT / "scripts" / "puppeteer.config.json"

MERMAID_FENCE_RE = re.compile(r"^```mermaid\s*$")
FENCE_END_RE = re.compile(r"^```\s*$")


def find_md_files() -> list[Path]:
    return sorted([p for p in DIAGRAMS_DIR.glob("*.md") if p.name != "README.md"])


def extract_mermaid_blocks(md_text: str) -> list[str]:
    blocks: list[str] = []
    lines = md_text.splitlines()
    in_block = False
    current: list[str] = []
    for line in lines:
        if not in_block and MERMAID_FENCE_RE.match(line):
            in_block = True
            current = []
            continue
        if in_block and FENCE_END_RE.match(line):
            in_block = False
            blocks.append("\n".join(current).strip() + "\n")
            current = []
            continue
        if in_block:
            current.append(line)
    return blocks


def ensure_dirs() -> None:
    MMD_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)


def check_mmdc() -> str | None:
    return shutil.which("mmdc")


def render_block(src_mmd: Path, out_img: Path, fmt: str, theme: str) -> None:
    cmd = [
        "mmdc",
        "-i", str(src_mmd),
        "-o", str(out_img),
        "-t", theme,
    ]
    # If a Puppeteer config is present, pass it to mermaid-cli (e.g., to disable sandbox)
    if PUPPETEER_CONFIG.exists():
        cmd.extend(["-p", str(PUPPETEER_CONFIG)])
    if fmt == "svg":
        # mmdc infers from extension; caller typically sets .svg already
        out_img = out_img.with_suffix(".svg")
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["png", "svg"], default="png")
    parser.add_argument("--theme", default="default", help="mmdc theme: default|dark|forest|neutral")
    args = parser.parse_args()

    ensure_dirs()
    mmdc = check_mmdc()
    if not mmdc:
        print("ERROR: 'mmdc' not found. Install with: npm install -g @mermaid-js/mermaid-cli@10")
        return 2

    md_files = find_md_files()
    total_blocks = 0
    failed_blocks = 0
    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8")
        blocks = extract_mermaid_blocks(text)
        if not blocks:
            continue
        base = md_path.stem
        for idx, code in enumerate(blocks, start=1):
            src = MMD_DIR / f"{base}__block{idx:02d}.mmd"
            out = IMG_DIR / f"{base}__block{idx:02d}.{args.format}"
            src.write_text(code, encoding="utf-8")
            try:
                render_block(src, out, args.format, args.theme)
                print(f"Rendered {out}")
                total_blocks += 1
            except subprocess.CalledProcessError as e:
                print(f"ERROR rendering {src} -> {out}: {e}")
                failed_blocks += 1
                continue
    print(f"Done. Rendered {total_blocks} diagram(s), {failed_blocks} failed. Output -> {IMG_DIR}")
    return 1 if failed_blocks > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())

