# Paper Sources

LaTeX sources for the paper and its supplementary material.

## Files

- `main.tex` — paper (uses the `acmart` document class)
- `supplementary.tex` — supplementary material
- `refs.bib` — bibliography
- `build.sh` — build script (see below)

## Building

```sh
./build.sh              # builds main.pdf and supplementary.pdf
./build.sh main         # only main.pdf
./build.sh supp         # only supplementary.pdf
./build.sh clean        # remove intermediate files
./build.sh distclean    # remove intermediates AND the PDFs
```

The script auto-detects which LaTeX engine to use. Install **one** of:

### Tectonic (recommended — easiest cross-platform setup)

Single ~30 MB binary. Auto-downloads any LaTeX package the document needs on
first build, so there is no `texlive-*` package hunt.

| OS      | Install                                                                  |
| ------- | ------------------------------------------------------------------------ |
| macOS   | `brew install tectonic`                                                  |
| Windows | `scoop install tectonic` (or download from the [releases page][rel])     |
| Linux   | `cargo install tectonic` (or see the [install guide][install])           |

[rel]: https://github.com/tectonic-typesetting/tectonic/releases
[install]: https://tectonic-typesetting.github.io/install.html

### TeX Live + latexmk (alternative)

If you already have a TeX distribution, the script will use `latexmk`. You will
need these collections (names below are for Debian/Ubuntu — MacTeX and MiKTeX
typically include them out of the box):

```sh
sudo apt install latexmk \
    texlive-latex-extra \
    texlive-publishers   \
    texlive-science      \
    texlive-fonts-extra  \
    texlive-bibtex-extra
```
