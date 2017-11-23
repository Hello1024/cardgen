#!/bin/bash

cnt=0
coollayers=( 4b 4c 4d 4e 5a )

while true ; do
  cp deepdream.py deepdream-copy.py
  for i in $(seq 1 1 8)
  do

    read template || exit
    # Escape string to make it regex safe.
    template=$(printf '%s' "$template" | sed -e 's/[\/&]/\\&/g')
    # Forgive me for such a horrible hack...
    sed -i "s/\\\$template${i}\\\$/${template}/" "deepdream-copy.py"

  done
  #continue
  python deepdream-copy.py --input Christmas.jpg --layer=import/mixed${coollayers[$(($cnt % 5))]}
  for FILENAME in front back
  do
    inkscape --export-pdf "$(printf "%05d\n" $cnt)-$FILENAME.pdf" "${FILENAME}.svg"
  done

  cnt=$((cnt + 1))

done;
