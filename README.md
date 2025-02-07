# 🔍 Anomaly Detection for Fraud Detection System

![CI/CD Pipeline](https://github.com/your-username/Anormaly-Detection-Fraud/actions/workflows/build.yml/badge.svg)

A machine learning pipeline for detecting fraudulent transactions using anomaly detection techniques. 🚀

## ✨ Features

- **Automated EDA** with visualizations 📊 (histograms, box plots, correlation matrices)
- **Multiple Anomaly Detection Algorithms** 🤖:
    - Isolation Forest
    - One-Class SVM
    - Local Outlier Factor (LOF)
- **Model Evaluation & Selection** with metrics tracking 📈
- **CI/CD Pipeline** with automated model deployment 🔄
- **DVC Integration** for data versioning 📦
- **Logging & Artifact Tracking** 📝

## 🛠️ Tech Stack
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/github%20actions-%232671E5.svg?style=for-the-badge&logo=githubactions&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-945DD6?style=for-the-badge&logo=dvc&logoColor=white)

## 💻 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/Anormaly-Detection-Fraud.git
cd Anormaly-Detection-Fraud
```

2. **Install dependencies**:
```bash
python -m venv venv
source venv/bin/activate  # Linux/MacOS
# venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

3. **Set up DVC** (if using data versioning):
```bash
dvc init
dvc remote add -d myremote gs://your-bucket-name
```

## 🚀 Usage

1. **Prepare your data** 📋:
     - Place your transaction data in `data.csv`
     - Expected format:
         ```
         Transaction_ID,Transaction_Amount,Transaction_Volume,...,Account_Type
         ```

2. **Run the pipeline**:
```bash
python train.py
```

**Output Structure** 📁:
```
├── artifacts/            # Analysis results and metrics
├── figures/             # Generated visualizations
├── models/              # Saved models (including best_model.pkl)
├── logs/                # Execution logs
```

## 🐳 Docker Support (TODO)

```bash
# Build the image
docker build -t fraud-detection .

# Run the container
docker run -v $(pwd)/data:/app/data fraud-detection
```

## 🔄 CI/CD Pipeline

The GitHub Actions workflow:
1. Runs on push to `main` branch or pull requests
2. Sets up Python 3.10 environment
3. Installs dependencies
4. Executes the training pipeline
5. Creates release with artifacts if successful
6. Uploads:
     - Trained models
     - Visualizations
     - Evaluation metrics
     - Log files

## 📊 Model Evaluation

The pipeline automatically:
- Compares model performance using anomaly detection rates
- Selects the best performing model
- Saves all models with versioning
- Generates evaluation reports in JSON format

## 🤝 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

