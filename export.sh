#!/bin/sh

INPUT=test.svg

read template1
cat $INPUT | sed "s/\\\$template1\\\$/${template1}/"

inkscape --export-text-to-path -i=g2990 -j -P test.ps test.svg


