# Architecture-Transformer

Ce dépôt contient les travaux pratiques réalisés autour des architectures RNN, LSTM, Transformer, GPT et BERT.

Les notebooks combinent les explications du cours, les implémentations, les entraînements, les tests et les visualisations des résultats.

## Structure du dépôt

| Fichier | Contenu |
|---|---|
| [`architecture_transformer.ipynb`](./architecture_transformer.ipynb) | Notebook principal contenant les explications sur les RNN, LSTM, Transformer, GPT et BERT, ainsi que les trois exercices de la deuxième partie |
| [`rnn_lstm_transformer_gpt_bert.ipynb`](./rnn_lstm_transformer_gpt_bert.ipynb) | Première partie : série sinusoidale, RNN, LSTM, Transformer, GPT et BERT |
| [`exercice_1_consommation_transformer.ipynb`](./exercice_1_consommation_transformer.ipynb) | Deuxième partie, exercice 1 : prédiction de la consommation électrique |
| [`exercice_2_mini_gpt.ipynb`](./exercice_2_mini_gpt.ipynb) | Deuxième partie, exercice 2 : mini-GPT autorégressif |
| [`exercice_3_mini_bert.ipynb`](./exercice_3_mini_bert.ipynb) | Deuxième partie, exercice 3 : mini-BERT pour les mots masqués |

# Première partie

La première partie est réalisée dans [`rnn_lstm_transformer_gpt_bert.ipynb`](./rnn_lstm_transformer_gpt_bert.ipynb).

## Travaux réalisés

- Générer une série temporelle `y = sin(x)`.
- Créer une fenêtre glissante contenant 10 observations consécutives.
- Séparer les exemples en données d'entraînement et de test.
- Créer, entraîner et tester un modèle RNN capable de prédire la valeur suivante.
- Créer, entraîner et tester un modèle LSTM capable de prédire la valeur suivante.
- Créer, entraîner et tester un modèle Transformer capable de prédire la valeur suivante.
- Créer et tester un exemple de modèle GPT pour prédire la suite d'un texte.
- Créer et tester un exemple de modèle BERT pour prédire un mot masqué dans une séquence.

## Modèles utilisés

### RNN

Le RNN lit les 10 valeurs successives de la séquence et conserve un état caché représentant la mémoire temporaire. La dernière représentation est transformée en une valeur prédite par une couche linéaire.

### LSTM

Le LSTM reprend le fonctionnement récurrent du RNN avec des portes d'entrée, d'oubli et de sortie. Ces mécanismes permettent de conserver ou d'oublier certaines informations plus efficacement.

### Transformer

Le Transformer utilise une projection linéaire, des encodages de position et un encodeur avec auto-attention multi-têtes. Il compare directement les positions de la séquence sans traiter les données uniquement de façon récurrente.

### GPT

GPT utilise une attention causale : chaque position ne peut consulter que les positions précédentes. Cette propriété permet de générer un texte token par token.

### BERT

BERT utilise une attention bidirectionnelle : une position masquée peut prendre en compte le contexte situé à sa gauche et à sa droite. Ce modèle est adapté aux tâches de reconstruction de mots masqués.

## Comparaison des modèles temporels

Les entrées des modèles RNN, LSTM et Transformer ont la forme `(batch, 10, 1)`.

Les sorties ont la forme `(batch, 1)`, une seule valeur étant prédite pour chaque séquence.

La fonction de perte utilisée est l'erreur quadratique moyenne MSE. Les prédictions sont également visualisées avec les valeurs réelles de la série temporelle.

# Deuxième partie

Les exercices de la deuxième partie se trouvent à la fin de [`architecture_transformer.ipynb`](./architecture_transformer.ipynb). Des notebooks dédiés sont disponibles pour leur correction.

## Exercice 1 — Prédiction de consommation électrique

Notebook : [`exercice_1_consommation_transformer.ipynb`](./exercice_1_consommation_transformer.ipynb)

### Objectif

Utiliser les 24 dernières heures de consommation électrique pour prédire la consommation moyenne de l'heure suivante.

### Travaux réalisés

- Charger le jeu de données UCI `Individual Household Electric Power Consumption`.
- Lire la variable `Global_active_power`.
- Agréger les mesures en puissance moyenne horaire.
- Écarter les fenêtres contenant des valeurs manquantes.
- Normaliser les données à partir de l'ensemble d'entraînement uniquement.
- Construire un Transformer avec une projection 1 vers 32, des encodages de position et deux couches encodeur.
- Séparer les données chronologiquement en entraînement, validation et test.
- Conserver le modèle ayant obtenu la meilleure perte de validation.
- Calculer les métriques MAE et RMSE en kW.
- Comparer le Transformer avec une référence de persistance et une référence journalière.
- Visualiser les prédictions sur sept jours du jeu de test.

La préparation utilise `household_power_consumption.txt`, qui doit être placé dans le dossier du projet.

## Exercice 2 — Mini-GPT

Notebook : [`exercice_2_mini_gpt.ipynb`](./exercice_2_mini_gpt.ipynb)

### Objectif

Construire un Transformer autorégressif entraîné à prédire le prochain caractère d'un texte.

### Travaux réalisés

- Préparer un corpus texte et construire un vocabulaire caractère par caractère.
- Décaler les cibles d'une position pour apprendre à prédire le caractère suivant.
- Implémenter un bloc GPT avec LayerNorm, attention causale multi-têtes, résidus et MLP avec GELU.
- Construire un modèle contenant deux blocs, quatre têtes et des embeddings de caractères et de positions.
- Entraîner le modèle avec l'entropie croisée.
- Générer 200 caractères à partir d'une amorce.
- Comparer les résultats avec les températures `0.5`, `1.0` et `1.5`.
- Vérifier que les caractères futurs ne modifient pas les logits des positions précédentes.

## Exercice 3 — Mini-BERT

Notebook : [`exercice_3_mini_bert.ipynb`](./exercice_3_mini_bert.ipynb)

### Objectif

Construire un Transformer bidirectionnel entraîné à retrouver des mots masqués dans une phrase.

### Travaux réalisés

- Construire un vocabulaire par mots à partir d'un corpus d'entraînement.
- Encoder les phrases avec les identifiants `[CLS]`, `[SEP]`, `[PAD]`, `[UNK]` et `[MASK]`.
- Compléter les séquences jusqu'à une longueur de 32 tokens.
- Sélectionner aléatoirement 15 % des tokens ordinaires.
- Appliquer la stratégie BERT : 80 % `[MASK]`, 10 % token aléatoire et 10 % token conservé.
- Construire un encodeur Transformer avec embeddings de tokens et de positions.
- Utiliser une attention bidirectionnelle et un masque de padding.
- Calculer la perte uniquement sur les positions sélectionnées.
- Mesurer l'exactitude sur le jeu de validation.
- Afficher les trois mots les plus probables pour des positions masquées.
- Vérifier que les valeurs du padding n'affectent pas les prédictions utiles.

# Installation

Depuis Windows PowerShell, à la racine du dépôt :

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

python -m pip install torch matplotlib numpy pandas transformers jupyter ipykernel
python -m ipykernel install --user `
  --name architecture-transformer `
  --display-name "Python (Architecture Transformer)"
```

Si l'exécution de scripts est bloquée par PowerShell :

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

# Exécution des notebooks

Ouvrir Jupyter ou VS Code, puis sélectionner le kernel `Python (Architecture Transformer)`.

```powershell
jupyter notebook
```

L'ordre recommandé est :

1. `rnn_lstm_transformer_gpt_bert.ipynb`
2. `architecture_transformer.ipynb`
3. `exercice_1_consommation_transformer.ipynb`
4. `exercice_2_mini_gpt.ipynb`
5. `exercice_3_mini_bert.ipynb`

# Remarques

- Les modèles GPT-2 et BERT préentraînés nécessitent une connexion Internet lors du premier téléchargement.
- Le jeu de données de l'exercice 1 ne doit pas être versionné avec le dépôt si sa taille est importante.
- Toutes les cellules et fichiers peuvent être régénérés ou modifiés avant un commit manuel.