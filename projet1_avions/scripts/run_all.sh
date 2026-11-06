#!/bin/zsh
# Campagne d'entraînement : 3 modèles x 3 graines. À lancer depuis projet1_avions/.
# caffeinate empêche le Mac de se mettre en veille pendant les calculs.
PY=../.venv/bin/python
for seed in 0 1 2; do
  for model in flat hier soft; do
    [ -f "runs/${model}_efficientnet_b0_224_seed${seed}/.done" ] && continue
    caffeinate -i $PY -m src.train --model $model --seed $seed --num-workers 6 \
      && touch "runs/${model}_efficientnet_b0_224_seed${seed}/.done"
  done
done
