#!/bin/bash
# Télécharge et décompresse FGVC-Aircraft dans data/ (2,75 Go à télécharger). À lancer depuis projet1_avions/.
set -e
mkdir -p data
curl -L -o data/fgvc-aircraft-2013b.tar.gz https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
tar -xzf data/fgvc-aircraft-2013b.tar.gz -C data
rm data/fgvc-aircraft-2013b.tar.gz
