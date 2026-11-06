# Protocole figé avant l'évaluation sur le test (2026-11-06)

Tous les choix ci-dessous ont été faits sur la validation. Le test n'a été lancé qu'après l'écriture de ce fichier.

- Modèles comparés : flat, hier, soft, EfficientNet-B0, 224 px, graines 0, 1 et 2, sans nouvel entraînement.
- Checkpoint de chaque run : meilleure accuracy variante en validation.
- Hyperparamètres inchangés : w_coarse = 0,5 ; beta = 10. Aucun réglage n'a été fait sur ces deux valeurs.
- Critères : accuracy variante, gravité moyenne sur toutes les images, taux d'erreurs graves, cohérence.
- Tests : différences par paires (hier - flat, soft - flat, soft - hier), bootstrap apparié sur les images, 2 000 tirages, moyenne des 3 graines par image.
- Modèle retenu pour la démo : soft, graine 2 (meilleure accuracy variante en validation parmi les soft : 0,701).
