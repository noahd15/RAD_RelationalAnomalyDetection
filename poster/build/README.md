# Rebuilding the poster

`RAD_CIKM2026_poster.pptx` is a single A0-portrait slide (33.1 in x 46.8 in) generated with pptxgenjs.
Everything except the three figure images is native PowerPoint (text boxes, tables, line charts), so the
simplest way to change it is to open the .pptx and edit directly.

To regenerate from source:

```bash
cd poster/build
npm install pptxgenjs          # once
node build_poster.js           # writes ../RAD_CIKM2026_poster.pptx
```

`build_poster.js` requires `apply_theme.js` from the Claude pptx skill (hard-coded path at the top of the file);
if that path is not available, delete the `applyTheme` require and call, and the deck still builds with
hex colors but without theme-color metadata.

Assets: `fig1_hi-04.png` (paper Fig. 1), `fig-000.png` (paper Fig. 2), `fig3_vec-1.png` (ablation bars rendered
from `results/shared/combined_ablation_barplots_broken_axis.pdf`), `qr_paper.png` (arXiv), `qr_code.png` (GitHub).
