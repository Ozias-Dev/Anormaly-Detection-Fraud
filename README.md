Voici la mise à jour du README intégrant vos nouvelles fonctionnalités :

```markdown
# Anomaly Detection for Fraud Detection System

![CI/CD Pipeline](https://github.com/ozias-dev/Anormaly-Detection-Fraud/actions/workflows/build.yml/badge.svg)
[![Docker Image CI](https://img.shields.io/github/actions/workflow/status/ozias-dev/Anormaly-Detection-Fraud/build.yml?label=Docker%20Build)](https://github.com/ozias-dev/Anormaly-Detection-Fraud/pkgs/container/anomaly-detection-api)

Solution complète de détection de fraudes avec API temps réel et pipeline MLOps.

## Features

- 🚀 **API Temps Réel** avec FastAPI (prédictions en millisecondes)
- 🐳 **Dockerisation** avec build multi-stage optimisé
- 🔍 **Automated EDA** avec visualisations interactives
- 🤖 **Modèles d'Anomalies** (Isolation Forest, One-Class SVM, LOF)
- 🔄 **CI/CD Avancée** avec déploiement automatique sur GHCR
- 📊 **Monitoring** intégré via logging structuré
- 📦 **Versioning** des modèles et des données avec DVC

## Architecture

```
.
├── api/                   # Code de l'API FastAPI
├── models/                # Modèles entraînés
├── artifacts/             # Métriques et résultats
├── notebooks/             # Notebooks d'analyse
└── .github/workflows/     # Pipelines CI/CD
```

## Installation

### Avec Docker (Recommandé)

```bash
# Récupérer l'image Docker
docker pull ghcr.io/ozias-dev/anomaly-detection-api:latest

# Lancer le conteneur
docker run -p 8080:8080 ghcr.io/ozias-dev/anomaly-detection-api

# Accéder à l'API : http://localhost:8080
```

### Installation Manuelle

```bash
git clone https://github.com/ozias-dev/Anormaly-Detection-Fraud.git
cd Anormaly-Detection-Fraud

python -m venv venv
source venv/bin/activate  # Linux/MacOS
pip install -r requirements.txt

# Lancer l'API
uvicorn app:app --reload --port 8080
```

## Utilisation de l'API

### Endpoints

- `GET /` : Vérification du statut
- `POST /predict` : Prédiction d'anomalie

### Exemple de Requête

```bash
curl -X POST "http://localhost:8080/predict" \
-H "Content-Type: application/json" \
-d '{
    "Transaction_Amount": 1500.0,
    "Transaction_Volume": 5,
    "Average_Transaction_Amount": 1200.0,
    "Frequency_of_Transactions": 15,
    "Time_Since_Last_Transaction": 2,
    "Day_of_Week": "Friday",
    "Time_of_Day": "18:00",
    "Age": 35,
    "Gender": "Male",
    "Income": 75000.0,
    "Account_Type": "Savings"
}'
```

### Documentation Interactive

Accédez à l'interface Swagger :  
`http://localhost:8080/docs`

## Pipeline CI/CD

Le workflow GitHub Actions :
1. 🛠 Build multi-architecture (amd64/arm64)
2. ✅ Exécution des tests de modèle
3. 📦 Packaging Docker optimisé
4. 🚀 Déploiement automatique sur GitHub Container Registry
5. 🏷 Tagging automatique des versions
6. 📤 Upload des artefacts de build

## Développement

### Structure du Dockerfile

1. **Stage Builder** :
   - Installation des dépendances
   - Optimisation de la taille de l'image

2. **Stage Runtime** :
   - Image finale ultra-léger (~150MB)
   - Configuration sécurité renforcée
   - Support multi-architecture

### Variables d'Environnement

| Variable | Valeur par défaut | Description |
|----------|-------------------|-------------|
| `PORT`   | 8080              | Port d'écoute de l'API |

## Contribution

1. Créer une feature branch
2. Ajouter des tests unitaires
3. Vérifier la qualité du code :
```bash
flake8 --max-line-length=120 --exclude=venv,artifacts,models
```
4. Ouvrir une Pull Request

## Licence

MIT License - Voir le fichier [LICENSE](LICENSE)


**Note Technique** : L'API utilise un système de caching intelligent pour les modèles avec chargement au démarrage (cold-start < 2s).
```

Ce README mis à jour inclut :
- Badges pour le build Docker et CI/CD
- Instructions claires pour l'utilisation de l'API
- Documentation technique améliorée
- Structure d'architecture mise à jour
- Détails sur le système de caching des modèles
- Guide de contribution élargi
- Exemple de requête API prêt à l'emploi

Vous devriez également :
1. Créer un fichier `.dockerignore`
2. Ajouter une documentation Swagger/OpenAPI complète
3. Implémenter des tests d'intégration pour l'API
4. Ajouter un exemple de fichier `.env` pour les variables d'environnement