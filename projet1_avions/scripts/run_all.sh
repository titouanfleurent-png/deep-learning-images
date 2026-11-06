#!/bin/bash
# Refait les 9 entraînements : 3 modèles x 3 graines. À lancer depuis projet1_avions/, environnement activé.
# Sur Mac, caffeinate empêche la mise en veille pendant les calculs.
KEEP_AWAKE=$(command -v caffeinate >/dev/null && echo "caffeinate -i")
for seed in 0 1 2; do
  for model in flat hier soft; do
    [ -f "runs/${model}_efficientnet_b0_224_seed${seed}/.done" ] && continue
    $KEEP_AWAKE python -m src.train --model $model --seed $seed --num-workers 6
  done
done
