from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import logging
import joblib
import uvicorn

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Création des dossiers nécessaires
ARTIFACTS_DIR = "artifacts"
FIGURES_DIR = "figures"
MODELS_DIR = "models"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Chemins pour sauvegarder le scaler et le meilleur modèle
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")

app = FastAPI(title="API de Détection d'Anomalies")

# Modèle de données pour la prédiction
class TransactionData(BaseModel):
    Transaction_Amount: float
    Transaction_Volume: float
    Average_Transaction_Amount: float
    Frequency_of_Transactions: float
    Time_Since_Last_Transaction: float
    Age: float
    Income: float
    # Pour simplifier cet exemple, nous considérons uniquement les features numériques.
    # Les variables catégorielles (Day_of_Week, Time_of_Day, Gender, Account_Type) devraient
    # être pré-traitées ou encodées côté client.

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de Détection d'Anomalies."}

@app.post("/train")
def train_pipeline():
    """
    Entraîne le pipeline complet et sauvegarde le meilleur modèle et le scaler.
    Nécessite la présence d'un fichier 'data.csv' à la racine.
    """
    try:
        # Chargement des données
        df = pd.read_csv("data.csv")
        logging.info("Données chargées pour l'entraînement.")

        # Prétraitement des données
        if "Transaction_ID" in df.columns:
            df = df.drop("Transaction_ID", axis=1)
            logging.info("Colonne Transaction_ID supprimée.")

        # Transformation des variables catégorielles en variables indicatrices
        categorical_cols = ["Day_of_Week", "Time_of_Day", "Gender", "Account_Type"]
        for col in categorical_cols:
            if col in df.columns:
                df = pd.get_dummies(df, columns=[col], drop_first=True)
                logging.info(f"Transformation de la colonne {col} effectuée.")

        X = df.copy()

        # Mise à l'échelle
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        joblib.dump(scaler, SCALER_PATH)
        logging.info(f"Scaler sauvegardé dans {SCALER_PATH}.")

        # Division en ensembles d'entraînement et de test
        X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)
        logging.info(f"Ensembles d'entraînement et de test créés: {X_train.shape}, {X_test.shape}.")

        # Entraînement de trois modèles
        models = {}
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        iso_forest.fit(X_train)
        models["IsolationForest"] = iso_forest
        logging.info("IsolationForest entraîné.")

        # One-Class SVM
        oc_svm = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale")
        oc_svm.fit(X_train)
        models["OneClassSVM"] = oc_svm
        logging.info("OneClassSVM entraîné.")

        # Local Outlier Factor (avec novelty=True pour permettre la prédiction)
        lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.05)
        lof.fit(X_train)
        models["LocalOutlierFactor"] = lof
        logging.info("LocalOutlierFactor entraîné.")

        # Évaluation des modèles sur l'ensemble de test
        evaluations = {}
        for name, model in models.items():
            try:
                y_pred = model.predict(X_test)
                prop_outliers = np.mean(y_pred == -1)
                evaluations[name] = prop_outliers
                logging.info(f"{name} détecte {prop_outliers*100:.2f}% d'anomalies.")
            except Exception as e:
                evaluations[name] = None
                logging.error(f"Erreur lors de l'évaluation du modèle {name}: {e}")

        # Sélection du meilleur modèle (celui avec la plus faible proportion d'anomalies)
        best_model_name = min(evaluations, key=lambda k: evaluations[k] if evaluations[k] is not None else float('inf'))
        best_model = models[best_model_name]
        joblib.dump(best_model, BEST_MODEL_PATH)
        logging.info(f"Meilleur modèle {best_model_name} sauvegardé dans {BEST_MODEL_PATH}.")

        return {
            "status": "Entraînement terminé",
            "best_model": best_model_name,
            "evaluations": evaluations
        }
    except Exception as e:
        logging.exception("Erreur lors de l'entraînement du pipeline.")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict(transaction: TransactionData):
    """
    Prédit si une transaction est anormale ou non à l'aide du meilleur modèle entraîné.
    """
    try:
        # Vérification de la présence du modèle et du scaler
        if not os.path.exists(BEST_MODEL_PATH) or not os.path.exists(SCALER_PATH):
            raise HTTPException(
                status_code=400,
                detail="Le modèle ou le scaler n'est pas disponible. Veuillez lancer /train d'abord."
            )
        best_model = joblib.load(BEST_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        # Préparation des données pour la prédiction
        data_dict = transaction.dict()
        df = pd.DataFrame([data_dict])
        # Mise à l'échelle des features
        X_scaled = scaler.transform(df)
        prediction = best_model.predict(X_scaled)

        result = "Anomalie détectée" if prediction[0] == -1 else "Transaction normale"
        return {"prediction": int(prediction[0]), "result": result}
    except Exception as e:
        logging.exception("Erreur lors de la prédiction.")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
