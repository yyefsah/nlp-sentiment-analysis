# Analyse de Sentiments — Critiques de Films (SST-2)
### NLP · TF-IDF, Régression Logistique, Random Forest & XGBoost

## Contexte et Problématique

L'analyse de sentiment est une application centrale du traitement automatique du langage (NLP), utilisée dans de nombreux contextes industriels : modération de contenu, analyse de réputation de marque, veille médiatique, ou encore recommandation de produits.

**Objectif :** À partir de phrases extraites de critiques de films, construire un modèle capable de classer automatiquement le sentiment (positif / négatif) et comparer trois algorithmes de classification sur cette tâche.

---

## Données

**Dataset :** Stanford Sentiment Treebank v2 (SST-2) — Socher et al., 2013

Dataset académique de référence en NLP, issu de critiques de films réelles annotées manuellement. Utilisé comme benchmark dans les publications scientifiques et les compétitions NLP (GLUE Benchmark).

| Split | Taille | Taux positif |
|---|---|---|
| Train + Dev | 7 792 phrases | 52.03% |
| Test | 1 821 phrases | 49.92% |

**Classes :** 0 = sentiment négatif · 1 = sentiment positif

**Exemples :**
- ✅ *"a stirring, funny and finally transporting re-imagining of beauty and the beast"*
- ❌ *"no movement, no yuks, not much of anything"*

---

## Méthodologie

```
Données SST-2 → Nettoyage → TF-IDF (unigrammes + bigrammes)
→ Régression Logistique → Random Forest → XGBoost
→ Comparaison (AUC, F1, Accuracy) → Analyse des termes discriminants
```

### Vectorisation TF-IDF

| Paramètre | Valeur | Justification |
|---|---|---|
| `max_features` | 20 000 | Vocabulaire suffisant sans surcharge mémoire |
| `ngram_range` | (1, 2) | Capture "not good", "very bad" — crucial pour la négation |
| `sublinear_tf` | True | log(1 + tf) pour atténuer les termes très fréquents |
| `stop_words` | english | Suppression des mots vides non discriminants |

---

## Résultats

| Modèle | Accuracy | F1-Score | AUC |
|---|---|---|---|
| **Régression Logistique** | **0.7996** | **0.7994** | **0.8874** |
| Random Forest | 0.7430 | 0.7428 | 0.8276 |
| XGBoost | 0.7287 | 0.7283 | 0.8156 |

La Régression Logistique domine — résultat cohérent avec la littérature : TF-IDF + LR est difficile à battre sur des tâches de sentiment binaire avec des phrases courtes et structurées.

---

## Technologies

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)

- **Python** — numpy, pandas, matplotlib, seaborn
- **scikit-learn** — TfidfVectorizer, LogisticRegression, RandomForestClassifier
- **XGBoost** — gradient boosting

---

## Structure du Projet

```
04_nlp_sentiment_sst2/
│
├── 04_nlp_sentiment_sst2.ipynb   # Notebook principal
└── README.md
```

> Les données sont téléchargées automatiquement à l'exécution du notebook — aucun fichier à uploader.

---

*Master 2 Probabilités, Statistiques et Nouvelles Données — HETIC 2026–2027*
