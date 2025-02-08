from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import json
import logging

app = FastAPI(title="API de Détection d'Anomalies")

# Chemins vers les fichiers du modèle et du scaler
MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# Variables globales pour le modèle et le scaler
model = None
scaler = None

# Liste des features attendues après transformation (one-hot des variables catégorielles)
# Cette liste doit correspondre aux colonnes obtenues lors de l'entraînement.
EXPECTED_FEATURES = [
    "Transaction_ID",
    "Transaction_Amount",
    "Transaction_Volume","Average_Transaction_Amount",
    "Frequency_of_Transactions","Time_Since_Last_Transaction",
    "Day_of_Week","Time_of_Day","Age","Gender","Income","Account_Type"
]

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

@app.on_event("startup")
def load_model():
    """
    Charge le modèle et le scaler dès le démarrage de l'API.
    """
    global model, scaler
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        logging.error("Le modèle ou le scaler n'existe pas. Vérifiez vos fichiers.")
        raise FileNotFoundError("Modèle ou scaler non trouvé.")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logging.info("Modèle et scaler chargés avec succès.")

class TransactionData(BaseModel):
    Transaction_ID: str = None
    Transaction_Amount: float = 0.0
    Transaction_Volume: float = 0.0
    Average_Transaction_Amount: float = 0.0
    Frequency_of_Transactions: float = 0.0
    Time_Since_Last_Transaction: float = 0.0
    Day_of_Week: str = "Monday"
    Time_of_Day: str = "00:00"
    Age: int = 30
    Gender: str = "Female"
    Income: float = 50000.0
    Account_Type: str = "Checking"

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de détection d'anomalies."}

# Chemins vers les fichiers
MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/feature_names.json"

# Variables globales
model = None
scaler = None
EXPECTED_FEATURES = None

@app.on_event("startup")
def load_model():
    """
    Charge le modèle, le scaler et les noms des features au démarrage de l'API.
    """
    global model, scaler, EXPECTED_FEATURES
    
    # Vérification de l'existence des fichiers
    for path in [MODEL_PATH, SCALER_PATH, FEATURES_PATH]:
        if not os.path.exists(path):
            logging.error(f"Le fichier {path} n'existe pas.")
            raise FileNotFoundError(f"Fichier {path} non trouvé.")
    
    # Chargement des modèles et features
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURES_PATH, 'r') as f:
        EXPECTED_FEATURES = json.load(f)
    
    logging.info("Modèle, scaler et features chargés avec succès.")

@app.post("/predict")
def predict(transaction: TransactionData):
    try:
        # Conversion des données d'entrée en DataFrame
        data = pd.DataFrame([transaction.model_dump()])
        
        # Suppression de Transaction_ID s'il existe
        if "Transaction_ID" in data.columns:
            data = data.drop("Transaction_ID", axis=1)
        
        # Encodage one-hot des variables catégorielles
        categorical_cols = ["Day_of_Week", "Time_of_Day", "Gender", "Account_Type"]
        data = pd.get_dummies(data, columns=categorical_cols)
        
        # Alignement des colonnes sur les features attendues
        for col in EXPECTED_FEATURES:
            if col not in data.columns:
                data[col] = 0
        data = data[EXPECTED_FEATURES]
        
        # Application du scaler et prédiction
        data_scaled = scaler.transform(data)
        prediction = model.predict(data_scaled)
        result = "Anomalie détectée" if prediction[0] == -1 else "Transaction normale"
        
        return {"prediction": int(prediction[0]), "result": result}
    except Exception as e:
        logging.exception("Erreur lors de la prédiction.")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)