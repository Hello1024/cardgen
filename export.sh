#!/bin/sh

INPUT=design.svg

echo "%!" > master-layer1.eps
echo "%!" > master-layer2.eps

xoffset=0
yoffset=0

for FILENAME in {1..2}
do
  cp "$INPUT" "$FILENAME.svg"
  for i in {1..8}
  do

    read template
    sed -i "s/\\\$template$i\\\$/${template}/" "${FILENAME}.svg"

  done


  for l in layer1 layer2; do
    inkscape --export-text-to-path -i=${l} -j --export-eps "${FILENAME}-${l}.eps" "${FILENAME}.svg"
    
    reflectedxoffset=$xoffset
    if [ $l = layer2 ]; then
      reflectedxoffset=$((600-xoffset))
    fi
    # create eps masterfile
    echo "gsave" >> master-${l}.eps
    echo "$((reflectedxoffset * 720 / 254 )) $((yoffset * 720 / 254 )) translate" >> master-${l}.eps
#    echo "30 rotate" >> master-${l}.eps
    echo "save" >> master-${l}.eps
    echo "/showpage {} bind def" >> master-${l}.eps
    echo "(${FILENAME}-${l}.eps) run" >> master-${l}.eps
    echo "restore" >> master-${l}.eps
    echo "grestore" >> master-${l}.eps
  done;
  
  xoffset=$((xoffset + 112))
  if (( xoffset > 500 )); then
    if (( ( yoffset / 99 ) % 2 )); then
      xoffset=0
      yoffset=$((yoffset + 99))
    else
      xoffset=40
      yoffset=$((yoffset + 99))    
    fi
  fi
done

for l in layer1 layer2; do
  echo "showpage" >> master-${l}.eps

  pstoedit -f "dxf: -mm -ctl" "master-${l}.eps" "${l}.dxf"
done;
