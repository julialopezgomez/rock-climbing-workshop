#!/usr/bin/env python3
"""Write an A4 landscape calibration board as metric vector SVG."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--square-mm", type=float, default=20.0)
    parser.add_argument("--columns", type=int, default=9, help="Internal corners")
    parser.add_argument("--rows", type=int, default=6, help="Internal corners")
    args = parser.parse_args()
    width, height = (args.columns + 1) * args.square_mm, (args.rows + 1) * args.square_mm
    if args.square_mm <= 0 or width > 277 or height > 170:
        parser.error("Board must fit A4 landscape with margins")
    x0, y0 = (297 - width) / 2, (210 - height) / 2
    parts = [
        (
            '<svg xmlns="http://www.w3.org/2000/svg" width="297mm" '
            'height="210mm" viewBox="0 0 297 210">'
        ),
        '<rect width="297" height="210" fill="white"/>',
    ]
    for row in range(args.rows + 1):
        for col in range(args.columns + 1):
            if (row + col) % 2 == 0:
                parts.append(
                    f'<rect x="{x0 + col * args.square_mm}" y="{y0 + row * args.square_mm}" '
                    f'width="{args.square_mm}" height="{args.square_mm}" fill="black"/>'
                )
    parts.append(
        f'<text x="10" y="200" font-size="4">{args.columns} x {args.rows} internal corners; '
        f"{args.square_mm} mm squares. Print at 100% and measure the result.</text>"
    )
    parts.append("</svg>")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(parts) + "\n")


if __name__ == "__main__":
    main()
