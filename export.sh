#!/bin/sh

INPUT=template.svg

FILENAME=1

  cp "$INPUT" "$FILENAME.svg"
  for i in {1..5}
  do

    read template
    sed -i "s/\\\$template$i\\\$/${template}/" "${FILENAME}.svg"

  done

  inkscape --export-text-to-path -i=g2990 -j --export-eps "${FILENAME}.eps" "${FILENAME}.svg"


  pstoedit -f "dxf: -mm -ctl" "${FILENAME}.eps" "${FILENAME}.dxf"
