#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build report.tex > build/compile.log
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build report.tex >> build/compile.log
cp build/report.pdf perception_report.pdf
printf '%s\n' "Built perception_report.pdf"
