#!/bin/sh

INPUT=test.svg

FILENAME=1

read template1
cat $INPUT | sed "s/\\\$template1\\\$/${template1}/" > ${FILENAME}.svg

inkscape --export-text-to-path -i=g2990 -j --export-eps ${FILENAME}.eps ${FILENAME}.svg


pstoedit -f "dxf: -mm -ctl" ${FILENAME}.eps ${FILENAME}.dxf
