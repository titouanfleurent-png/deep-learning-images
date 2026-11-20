#!/bin/bash
# Campagne du projet 2 : n et s, à 640 et 1280 px. Reprend après une coupure (fichier .done).
cd "$(dirname "$0")"
while pgrep -f "src.train --model n --imgsz 640" >/dev/null; do sleep 30; done
run() { [ -f "runs/yolo11$1_$2_seed0/.done" ] || caffeinate -i ../.venv/bin/python -m src.train --model $1 --imgsz $2 --batch $3 > "runs/yolo11$1_$2.log" 2>&1; }
run n 640 16
run s 640 16
run n 1280 8
run s 1280 8
