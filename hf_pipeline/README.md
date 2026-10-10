---
base_model: google/embeddinggemma-2
library_name: transformers
pipeline_tag: text-classification
tags:
- custom-code
- decision-making
---
# EmbeddingGemma 2 — pipeline décisionnelle

Pipeline Hugging Face locale utilisant les poids Google d’origine, sans quantification. CPU et `dtype="auto"` par défaut. La révision du checkpoint est fixée dans `config.json`. Aucun dépôt n’a été publié.

```bash
python -m pip install -r requirements.txt
```

Depuis ce dossier :

```python
from transformers import pipeline

pipe = pipeline(
    "decision-making", model=".",
    trust_remote_code=True, device="cpu", dtype="auto",
)
result = pipe({
    "state": "Mon paiement a été débité deux fois. Je souhaite un remboursement.",
    "questions": {
        "service": {
            "type": "choice", "instructions": "Quel service ?",
            "criteria": {"facturation": "Paiements et factures", "livraison": "Colis et livraison"},
        },
        "urgence": {
            "type": "score", "instructions": "Quel degré d’urgence ?",
            "criteria": ["Faible", "Modéré", "Élevé"],
        },
        "remboursement": {
            "type": "noul", "instructions": "Le client demande un remboursement.",
        },
    },
})
print(result)
```

Depuis un autre dossier, remplacer `model="."` par le chemin absolu de ce projet. Les poids sont téléchargés depuis `google/embeddinggemma-2` ou lus dans le cache HF ; ce dépôt contient uniquement du code et la configuration. Installer les dépendances avant le chargement. L’accès au checkpoint et ses conditions Google restent applicables.

## Fichiers

- `config.json` : révision Google, paramètres décisionnels et déclaration de la pipeline personnalisée.
- `modeling_decision.py` : chargement du modèle et intégration `AutoModel`.
- `decision_pipeline.py` : interface `transformers.pipeline`.
- `engine.py` : embeddings, centrage, similarités et agrégation.
- `compat.py` : requêtes et réponses au format commun Jev/Laya.
- `requirements.txt` : dépendances.

## Portée

`decision-making` est une tâche personnalisée, pas une tâche native de Transformers. Une requête dictionnaire par appel. Les types `choice`, `score`, `noul` sont pris en charge, sans serveur HTTP ni extensions de routage Laya.

Traitements inspirés de MediaPipe : préfixes de classification, centrage des candidats, normalisation L2, similarités cosinus et agrégation LogSumExp de fenêtres de texte. Ce n’est pas une reproduction exacte de son moteur natif. Aucun entraînement décisionnel ajouté ; probabilités non calibrées. La confiance suit la formule Jev, différente de celle de Laya. `usage.input_tokens` compte les entrées logiques, descriptions en cache comprises.

Le chargement réel sur CPU avec `auto` (BF16 pour la révision fixée), les trois types et leur égalité avec le moteur ont été vérifiés. Cela ne constitue pas un benchmark de qualité. Le téléchargement depuis un dépôt publié et l’installation dans un environnement vierge restent à vérifier.
