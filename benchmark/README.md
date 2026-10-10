# ARBITER Benchmark Suite

Ce module fournit le harnais d'évaluation et les jeux de données figés permettant de reproduire l'intégralité des benchmarks empiriques pour **ARBITER-272M**, **EmbeddingGemma-2 base** (`hf_pipeline/`), **Laya** (ConvAI) et **GLiNER**.

## Reproduction Immédiate

### 1. Suites Publiques (23 suites, 9 252 décisions : SST-2, TREC, Banking77, Fast Decisions)

```bash
# Évaluation d'ARBITER-272M sur GPU :
python benchmark/run.py --model arbiter-272m \
  --data benchmark/data/public_requested_v1/cases.jsonl \
  --device cuda:0 --threads 16 --output results/public_benchmark

# Évaluation d'EmbeddingGemma-2 base pipeline sur GPU :
python benchmark/run.py --model embeddinggemma2 \
  --data benchmark/data/public_requested_v1/cases.jsonl \
  --device cuda:0 --threads 16 --output results/public_benchmark
```

### 2. Suites Historiques (60 suites, 21 915 décisions : MASSIVE, Typed Decisions, etc.)

```bash
python benchmark/run.py --model arbiter-272m \
  --data benchmark/data/cases.jsonl \
  --device cuda:0 --threads 16 --output results/historical_benchmark
```

## Ce qui est repris de Laya

Les deux scripts officiels sont conservés sans modification dans `upstream/`, depuis le commit `3cf26cbcb18725dbc2d127bb8bb2c4c43243ae63`. La préparation exécute seulement leurs blocs de construction des données, pas leur chargement de modèles ni leurs cellules GPU. Elle échoue si une suite manque ; aucune substitution de dataset ou de prompt n'est faite. Pour `PolyAI/banking77`, le script de chargement obsolète est remplacé par la lecture directe du CSV original PolyAI, au commit `57ec275d8078af65b7731c2a98be812d844a6d6b`. L'ordre des lignes et le `ClassLabel` du script HF sont conservés, sans exécuter ce script ; les empreintes des sources sont enregistrées.

| Groupe | Suites et échantillonnage officiels |
|---|---|
| `typed` | LocalLLaMA/typed-decisions, test complet (annoncé à 400 cas, 2 000 décisions) |
| `multilingual` | MASSIVE intent et scenario, 14 langues dont français ; XNLI, 15 langues dont français ; 300 premiers cas par suite |
| `english` | SST-5 (600), Emotion (600), prompt-injections (test complet), Banking77 (500), AG News (600), BoolQ (600) |
| `apps` | 10 suites : AG News, Emotion, Banking77, support, spam, phishing, jailbreak, toxicité, pertinence RAG, routage de domaine ; N=400 selon les filtres officiels |

MASSIVE utilise 20 candidats : le vrai label + 19 distracteurs (seed 13). Cela ne mesure pas une classification parmi toutes les intentions. Les suites applicatives comprennent aussi des splits `train` et des tâches connues de Laya : leurs chiffres ne représentent pas tous une généralisation hors entraînement. Les sous-ensembles en tête de dataset ne sont pas nécessairement représentatifs. Les groupes anglais/applicatifs se recoupent : ne pas les additionner en un unique score global.

## Comparaison

- Entrées et questions identiques, construites une fois en JSONL puis vérifiées par SHA256.
- Quatre exécutions séparées : notre pipeline HF, Laya English, Laya Multilingual, Laya Typed Decisions. Pas de sélection a posteriori du meilleur checkpoint par exemple.
- Checkpoints Laya fixés aux révisions recommandées par `laya/revisions.py` du dépôt consulté. Code Laya fixé au commit ci-dessus. Révision Google dans `hf_pipeline/config.json`.
- Device commun CPU ou CUDA (`--device cpu`, `cuda`, `cuda:0`), 8 threads CPU par défaut, lot de 1, trois requêtes de chauffe. EG2 conserve `dtype="auto"`, Laya sa politique de précision native sur le device choisi. La précision réelle est enregistrée : les latences ne constituent donc pas une ablation à précision identique.
- Aucun réglage de température sur le test. Laya garde ses températures livrées, EG2 ses paramètres actuels.
- Prédictions, erreurs, troncatures éventuelles signalées par Laya, versions, paramètres et temps sauvegardés. Pas de remplacement silencieux d’un modèle en erreur.

## Mesures et limites

Accuracy argmax, ECE 15 bins, Brier **multiclasse**, NLL, MAE des scores, accord avec les cibles souples pour typed-decisions. La calibration utilise `max(probabilités)` pour tous, jamais `confidence` (formules différentes entre Jev et Laya). Les petites erreurs de somme dues à l’arrondi sont renormalisées.

Les échecs figurent dans le dénominateur de `accuracy_including_failures` ; les autres métriques sont calculées sur les réponses valides et accompagnées de leur couverture. Le bootstrap est apparié par requête (toutes ses questions restent ensemble), 2 000 tirages, seed 13. Les IC sont exploratoires, sans correction des comparaisons multiples.

p50/p95 mesurent l'appel complet, tokenisation et post-traitement compris, après chauffe ; le cache de descriptions EG2 reste actif. Pic RSS du processus séparé : inclut imports, chargement et inférence ; ce n'est pas la seule taille des poids. Les chiffres GPU historiques de Laya ne servent pas de baseline de latence. Les budgets de contexte restent propres aux architectures, et Banking77 peut excéder le budget d'options Laya : ces échecs sont conservés.

Différences explicites avec les scripts originaux : inférence via les API publiques, CPU ou GPU en lot de 1 plutôt que lots GPU/forward pass seuls ; échecs comptés plutôt qu'exclus ; couverture à seuil respectant les égalités. Les jeux de données et prompts sont ceux des scripts, les temps ne reproduisent pas les tableaux historiques. Le protocole ne relance pas encore les expériences dédiées de permutation d'options, de contexte long ni de recalibration.

Laya Typed Decisions est entraîné sur le train de typed-decisions : sa comparaison avec EG2 est une comparaison de systèmes, pas deux modèles évalués sans entraînement spécifique sur cette tâche.

## Vérification locale

```bash
.venv/bin/python -m unittest discover -s benchmark/tests -v
```

Ces tests vérifient les métriques et le couplage des résultats, pas la qualité des modèles. Les exécutions réelles nécessitent les dépendances, datasets et poids.

Sources : [scripts et protocoles officiels](https://github.com/NandhaKishorM/laya/tree/3cf26cbcb18725dbc2d127bb8bb2c4c43243ae63/research), [limites publiées](https://github.com/NandhaKishorM/laya/blob/3cf26cbcb18725dbc2d127bb8bb2c4c43243ae63/docs/benchmarks.md). Code amont Apache-2.0 ; licence jointe dans `upstream/LICENSE`.

## GPU depuis le notebook ou le terminal

Dans `notebook/benchmark_laya.ipynb`, choisir `DEVICE = "cuda:0"` (GPU) ou `DEVICE = "cpu"`. Le notebook distingue automatiquement les dossiers CPU/GPU.

```bash
.venv/bin/python benchmark/run.py --model all --device cuda:0 --threads 8 --output results/laya_comparison_cuda_0
.venv/bin/python benchmark/compare.py results/laya_comparison_cuda_0
```

Chaque modèle est chargé dans un processus séparé sur le device demandé. Aucun repli silencieux vers CPU n'est accepté. Les temps CUDA sont synchronisés avant/après chaque appel réussi. Les métadonnées incluent le GPU, la version CUDA et les pics de mémoire PyTorch allouée/réservée (chargement et chauffe inclus), distincts de la RAM RSS. La mémoire d'autres processus GPU n'est pas comprise. La précision EG2 reste `auto` ; la précision effective de Laya est enregistrée pour les poids et dépend de sa politique d'inférence. Les rapports CPU et GPU ne sont pas mélangés.
