# Lab 1 – Deep Learning avec PyTorch (Partie 2 : classification multi-classes)

**Université Abdelmalek Essaâdi – FST Tanger – Département Génie Informatique**
**Master SBD – Module Deep Learning – Pr. Lotfi Elaachak**

## Objectif
Se familiariser avec PyTorch pour construire des réseaux DNN/MLP. Cette partie traite une **classification multi-classes** de pannes de machines (maintenance prédictive).

- **Dataset** : [Machine Predictive Maintenance Classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) (10 000 lignes, AI4I 2020)
- **Cible** : `Failure Type` (6 classes : No Failure, Heat Dissipation Failure, Power Failure, Overstrain Failure, Tool Wear Failure, Random Failures)
- **Notebook** : `Lab1_Part2_Classification.ipynb`

## Contenu du dépôt
```
.
├── Lab1_Part1_Regression.ipynb        # Partie 1 (régression, NYSE)
├── Lab1_Part2_Classification.ipynb    # Partie 2 (ce rapport)
└── README.md
```

## Exécution
1. Ouvrir le notebook dans **Google Colab** ou **Kaggle**.
2. Installer si besoin : `pip install imbalanced-learn kagglehub`.
3. Le CSV `predictive_maintenance.csv` est cherché dans le dossier courant, puis téléchargé via `kagglehub` ; sinon l'uploader manuellement.
4. *Run all*.

## Méthodologie

### 1. Pré-traitement
- Renommage des colonnes, vérification des valeurs manquantes et des doublons.
- Suppression des lignes **incohérentes** (`Target = 1` alors que `Failure Type = No Failure`).
- Suppression de `UDI`, `Product ID` (identifiants) et de `Target` (fuite d'information : dérivée de la cible).
- `Type` (L/M/H) encodé de façon ordinale ; ajout de `Temp_diff` (écart de température process − air) et `Power` (couple × vitesse angulaire).
- **Standardisation** (`StandardScaler`) ajustée sur le train uniquement. Valeurs aberrantes conservées (elles portent le signal des pannes).

### 2. Analyse exploratoire
- Dataset **très déséquilibré** : ~96,5 % de « No Failure », seulement quelques dizaines d'exemples pour *Tool Wear* et *Random Failures*.
- Les boxplots par classe montrent des profils distincts :
  - **Heat Dissipation** : vitesse de rotation faible et très concentrée, température de l'air plus élevée.
  - **Power Failure** : très grande dispersion du couple, de la vitesse et de la puissance (valeurs extrêmes hautes et basses).
  - **Overstrain** : couple élevé et usure d'outil élevée.
  - **Tool Wear** : usure d'outil concentrée entre ~200 et 250 min.
  - **Random Failures** : aucun profil distinctif (pannes aléatoires par nature).
- Les variables dérivées `Power` et `Temp_diff` séparent bien certaines classes.

### 3. Augmentation de données
Split stratifié **train / validation / test** (68 % / 12 % / 20 %), puis sur le **train uniquement** : **SMOTE** (classes minoritaires) + **sous-échantillonnage** de la classe majoritaire pour amener chaque classe à 3 000 exemples. Le test garde la distribution réelle (1 999 exemples, dont 1 929 « No Failure »).

### 4. Architecture
MLP PyTorch configurable (couches cachées, activation, Dropout, BatchNorm), perte `CrossEntropyLoss`, encapsulé dans une classe compatible scikit-learn (`TorchMLPClassifier`) qui enregistre l'historique loss/accuracy et gère l'early stopping (sur la loss de validation).

### 5. GridSearch
`GridSearchCV` (3-fold stratifié, score = **F1 macro**) sur un pipeline `imblearn` (SMOTE à l'intérieur de chaque fold pour éviter les fuites). Espace exploré : architecture `(64,32)` / `(128,64,32)`, learning rate `1e-3` / `1e-2`, optimiseur `adam` / `sgd`, epochs `30` / `60`.

**Meilleurs hyper-paramètres** (F1 macro CV = **0,5465** ± 0,020) :

| Paramètre | Valeur |
|---|---|
| Architecture | (128, 64, 32) |
| Learning rate | 0,001 |
| Optimiseur | Adam |
| Epochs | 60 |

Les 4 meilleures configurations ont un F1 macro CV entre 0,537 et 0,546 : les résultats sont proches et stables (écart-type ≤ 0,02), donc le choix n'est pas très sensible à la grille.

### 6. Régularisation
Comparaison au modèle de référence de : Dropout (0,3), L2 (1e-3), BatchNorm, early stopping (patience 10), et une combinaison. Le meilleur modèle est choisi sur le **F1 macro de validation** (le test n'est pas utilisé pour choisir). **Le meilleur modèle est « + Early stopping »** (arrêt à ~37 epochs).

## Résultats

### Comparaison référence vs meilleur modèle (test)

| Modèle | Accuracy | Sensibilité (macro) | F1 macro | F1 pondéré |
|---|---|---|---|---|
| Référence (60 epochs) | 0,953 | 0,567 | 0,508 | 0,961 |
| **+ Early stopping** | 0,948 | 0,590 | **0,536** | 0,960 |

> Les autres techniques (Dropout, L2, BatchNorm, combinaison) : compléter avec le tableau `comparison` du notebook.

### Métriques du meilleur modèle (+ Early stopping)

| Métrique | Train | Test |
|---|---|---|
| Accuracy | 0,9657 | 0,9480 |
| Balanced accuracy | 0,9915 | 0,5901 |
| Précision (macro) | 0,6216 | 0,4995 |
| Sensibilité (macro) | 0,9915 | 0,5901 |
| Spécificité (macro) | 0,9936 | 0,9476 |
| F1 macro | 0,7052 | 0,5356 |
| F1 pondéré | 0,9750 | 0,9602 |
| MCC | 0,7010 | 0,5029 |

### Résultats par classe (test)

| Classe | Précision | Rappel | F1 | Support |
|---|---|---|---|---|
| No Failure | 0,990 | 0,957 | 0,973 | 1929 |
| Overstrain Failure | 0,682 | 0,938 | **0,789** | 16 |
| Power Failure | 0,737 | 0,737 | 0,737 | 19 |
| Heat Dissipation Failure | 0,588 | 0,909 | 0,714 | 22 |
| Tool Wear Failure | 0,000 | 0,000 | 0,000 | 9 |
| Random Failures | 0,000 | 0,000 | 0,000 | 4 |

## Interprétation

### Ce que le modèle fait bien
- **Détection des pannes « physiques »** : Overstrain (rappel 94 %, F1 0,79), Heat Dissipation (rappel 91 %) et Power Failure (F1 0,74) sont bien reconnues, alors qu'elles représentent moins de 1 % des données. L'augmentation SMOTE et le choix du F1 macro comme critère ont donc été efficaces.
- **Classe majoritaire** : « No Failure » est prédite avec une très bonne précision (0,99) et un rappel de 0,96 → peu de fausses alarmes sur les machines saines, avec une spécificité de 0,95.
- **Early stopping utile** : par rapport à la référence, le F1 macro test passe de 0,508 à 0,536, la sensibilité de 0,567 à 0,590, et Overstrain passe de F1 0,688 à 0,789.
- **Apprentissage sain** : les courbes de loss et d'accuracy convergent sans divergence ; les courbes validation et test sont quasi superposées (≈ 95 % d'accuracy), ce qui confirme l'absence de fuite entre validation et test.
- **GridSearch cohérent** : résultats proches et stables (écart-type ≤ 0,02).

### Points à améliorer
- **Écart entre entraînement et test** : la loss d'entraînement atteint ~0,02 tandis que celle de validation/test se stabilise vers ~0,2–0,3 ; F1 macro de 0,705 en train contre 0,536 en test. L'early stopping limite cet écart, qui concerne surtout les classes minoritaires peu représentées.
- **Tool Wear Failure et Random Failures : F1 = 0**. Aucune des 9 pannes Tool Wear ni des 4 Random Failures n'est détectée (elles sont prédites « No Failure »). Pire, 35 + 24 machines saines sont classées à tort dans ces deux classes : les exemples SMOTE générés à partir de ~30 et ~12 échantillons réels ne représentent pas bien la classe. Pour *Random Failures*, c'est attendu : ces pannes sont aléatoires par définition et donc non prédictibles.
- **Accuracy trompeuse** : 94,8 % d'accuracy alors que le F1 macro n'est que de 0,54. Les supports de test très faibles (4 et 9) rendent aussi ces métriques instables.

### Pistes d'amélioration
- Ajouter une variable `Tool_wear × Torque` (surcharge) et des seuils liés à la physique des pannes (d'après la description du dataset AI4I).
- Pondération des classes ou *focal loss* à la place (ou en complément) de SMOTE ; réglage des seuils de décision.
- Régularisation plus forte (Dropout + L2 combinés), réseau plus petit.
- Comparer à un modèle d'arbres (Random Forest / XGBoost), souvent plus performant sur ces données tabulaires.
- Regrouper ou retirer *Random Failures*, qui n'est pas prédictible.

## Synthèse personnelle
Ce lab m'a permis de construire de bout en bout un pipeline de classification avec PyTorch, de mesurer l'impact du déséquilibre de classes (l'accuracy de 95 % cache un F1 macro de 0,54, d'où l'intérêt de la sensibilité, de la spécificité et du F1), d'appliquer SMOTE sans fuite de données (train uniquement, et à l'intérieur de la validation croisée), d'utiliser `GridSearchCV` avec un modèle PyTorch via un wrapper scikit-learn, et de comparer des techniques de régularisation : l'early stopping a donné le meilleur compromis. J'ai aussi vu les limites de l'augmentation synthétique quand une classe n'a que quelques dizaines d'exemples réels.

## Outils
Python, PyTorch, scikit-learn, imbalanced-learn, pandas, seaborn/matplotlib – Google Colab / Kaggle, GitHub.
