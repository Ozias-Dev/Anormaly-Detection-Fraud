# 🔍 Anomaly Detection for Fraud Detection System

![CI/CD Pipeline](https://github.com/ozias-dev/Anormaly-Detection-Fraud/actions/workflows/build.yml/badge.svg)
[![Docker Image CI](https://img.shields.io/github/actions/workflow/status/ozias-dev/Anormaly-Detection-Fraud/build.yml?label=Docker%20Build)](https://github.com/ozias-dev/Anormaly-Detection-Fraud/pkgs/container/anomaly-detection-api)

🛡️ Solution complète de détection de fraudes avec API temps réel et pipeline MLOps.

## ✨ Features

- 🚀 **API Temps Réel** avec FastAPI <img src="https://fastapi.tiangolo.com/img/favicon.png" width="20"> (prédictions en millisecondes)
- 🐳 **Dockerisation** avec build multi-stage optimisé <img src="https://www.docker.com/favicon.ico" width="20">
- 🔍 **Automated EDA** avec visualisations interactives <img src="https://pandas.pydata.org/static/img/favicon.ico" width="20">
- 🤖 **Modèles d'Anomalies** (Isolation Forest, One-Class SVM, LOF) <img src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" width="20">
- 🔄 **CI/CD Avancée** avec déploiement automatique sur GHCR <img src="https://github.githubassets.com/favicons/favicon.png" width="20">
- 📊 **Monitoring** intégré via logging structuré <img src="https://www.python.org/static/favicon.ico" width="20">
- 📦 **Versioning** des modèles et des données avec DVC <img src="https://dvc.org/favicon.ico" width="20">

## 🏗️ Architecture

```
.
├── 📁 api/                   # Code de l'API FastAPI
├── 🤖 models/                # Modèles entraînés
├── 📊 artifacts/             # Métriques et résultats
├── 📓 notebooks/             # Notebooks d'analyse
└── 🔄 .github/workflows/     # Pipelines CI/CD
```

## 🚀 Installation

### 🐳 Avec Docker (Recommandé)

```bash
# Récupérer l'image Docker
docker pull ghcr.io/ozias-dev/anomaly-detection-api:latest

# Lancer le conteneur
docker run -p 8080:8080 ghcr.io/ozias-dev/anomaly-detection-api

# Accéder à l'API : http://localhost:8080
```

### 💻 Installation Manuelle

```bash
git clone https://github.com/ozias-dev/Anormaly-Detection-Fraud.git
cd Anormaly-Detection-Fraud

python -m venv venv
source venv/bin/activate  # Linux/MacOS
pip install -r requirements.txt

# Lancer l'API
uvicorn app:app --reload --port 8080
```

## 🎯 Utilisation de l'API

### 🛣️ Endpoints

- `GET /` : Vérification du statut
- `POST /predict` : Prédiction d'anomalie

### 📝 Exemple de Requête

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

### 📚 Documentation Interactive

Accédez à l'interface Swagger :  
`http://localhost:8080/docs`

## 🔄 Pipeline CI/CD

Le workflow GitHub Actions :
1. 🛠 Build multi-architecture (amd64/arm64)
2. ✅ Exécution des tests de modèle
3. 📦 Packaging Docker optimisé
4. 🚀 Déploiement automatique sur GitHub Container Registry
5. 🏷 Tagging automatique des versions
6. 📤 Upload des artefacts de build

## 👨‍💻 Développement

### 🐳 Structure du Dockerfile

1. **Stage Builder** :
   - 📥 Installation des dépendances
   - ⚡ Optimisation de la taille de l'image

2. **Stage Runtime** :
   - 🚀 Image finale ultra-léger (~150MB)
   - 🔒 Configuration sécurité renforcée
   - 💪 Support multi-architecture

### ⚙️ Variables d'Environnement

| Variable | Valeur par défaut | Description |
|----------|-------------------|-------------|
| `PORT`   | 8080              | Port d'écoute de l'API |

## 🤝 Contribution

1. 🌿 Créer une feature branch
2. ✅ Ajouter des tests unitaires
3. 🔍 Vérifier la qualité du code :
```bash
flake8 --max-line-length=120 --exclude=venv,artifacts,models
```
4. 📮 Ouvrir une Pull Request

## 📄 Licence

MIT License - Voir le fichier [LICENSE](LICENSE)


**💡 Note Technique** : L'API utilise un système de caching intelligent pour les modèles avec chargement au démarrage (cold-start < 2s).