#!/usr/bin/env bash
# Build the print PDF from poster.html and a set of preview crops.
set -e
cd "$(dirname "$0")"
{ printf '<!doctype html>\n<html lang="en"><head><meta charset="utf-8"></head><body>\n'; cat poster.html; printf '\n</body></html>\n'; } > poster_print.html
timeout 180 google-chrome --headless=new --no-sandbox --disable-gpu --no-pdf-header-footer \
  --virtual-time-budget=20000 --run-all-compositor-stages-before-draw \
  --print-to-pdf="$PWD/RAD_CIKM26_poster_A0.pdf" "file://$PWD/poster_print.html" 2>/dev/null | grep -i "bytes written" || true
pdfinfo RAD_CIKM26_poster_A0.pdf | grep -E "Pages|Page size"
rm -f mid-*.png prev-*.png
pdftoppm -r 72 -png RAD_CIKM26_poster_A0.pdf mid
pdftoppm -r 28 -png RAD_CIKM26_poster_A0.pdf prev
python3 - <<'EOF'
from PIL import Image
im=Image.open('mid-1.png'); w,h=im.size
for i,name in enumerate(['mid_top','mid_mid','mid_bot']):
    im.crop((0,i*h//3,w,(i+1)*h//3)).save(name+'.png')
print(im.size)
EOF
