#!/bin/sh

cnt=0

while true; do
  cp deepdream.py deepdream-copy.py
  for i in {1..8}
  do

    read template
    # Forgive me for such a horrible hack...
    sed -i "s/\\\$template$i\\\$/${template}/" "deepdream-copy.py"

  done
  python deepdream-copy.py --input Christmas.jpg
  for FILENAME in front back
  do
    inkscape --export-pdf "$(printf "%05d\n" $cnt)-$FILENAME.pdf" "${FILENAME}.svg"
  done

  cnt=$((cnt + 1))

done;
