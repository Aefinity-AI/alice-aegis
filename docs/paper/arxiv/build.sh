#!/usr/bin/env bash
# Build the CIS-1 arXiv PDF. Requires TeX Live (pdflatex, bibtex, latexmk).
set -euo pipefail
cd "$(dirname "$0")"
latexmk -pdf -bibtex -interaction=nonstopmode main.tex
tar czf cis1-arxiv.tar.gz main.tex refs.bib main.bbl
echo "built main.pdf ($(pdfinfo main.pdf | awk '/Pages/{print $2}') pages) and cis1-arxiv.tar.gz"
