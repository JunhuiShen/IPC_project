#!/usr/bin/env bash
# Build main.tex (and supplementary.tex).
# Prefers `tectonic` (single-binary, auto-downloads LaTeX packages — easiest
# cross-platform setup). Falls back to a system `latexmk` + TeX Live install.
#
# Usage:
#   ./build.sh              # build both
#   ./build.sh main         # build only main
#   ./build.sh supp         # build only supplementary
#   ./build.sh clean        # remove build artifacts (keeps PDFs)
#   ./build.sh distclean    # remove build artifacts AND PDFs

set -euo pipefail
cd "$(dirname "$0")"

if command -v tectonic >/dev/null 2>&1; then
    ENGINE=tectonic
elif command -v latexmk >/dev/null 2>&1; then
    ENGINE=latexmk
else
    cat >&2 <<'EOF'
error: neither tectonic nor latexmk found.

Recommended: install Tectonic (one binary, no TeX Live needed):
  macOS:   brew install tectonic
  Windows: scoop install tectonic
  Linux:   cargo install tectonic  (or see https://tectonic-typesetting.github.io/install.html)

Alternative: install TeX Live + latexmk for your platform.
EOF
    exit 1
fi

build() {
    echo ">> building $1.tex with $ENGINE"
    case "$ENGINE" in
        tectonic) tectonic "$1.tex" ;;
        latexmk)  latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error "$1.tex" ;;
    esac
}

clean_aux() {
    local exts=(aux bbl blg fdb_latexmk fls log out toc lof lot nav snm vrb synctex.gz run.xml bcf)
    for stem in main supplementary; do
        for ext in "${exts[@]}"; do rm -f "$stem.$ext"; done
    done
}

case "${1:-all}" in
    all)        build main; build supplementary ;;
    main)       build main ;;
    supp|supplementary) build supplementary ;;
    clean)      clean_aux ;;
    distclean)  clean_aux; rm -f main.pdf supplementary.pdf ;;
    *) echo "usage: $0 [all|main|supp|clean|distclean]" >&2; exit 2 ;;
esac
