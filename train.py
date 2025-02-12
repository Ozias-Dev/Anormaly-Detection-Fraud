import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
from typing import Dict, Tuple, List, Any
import json
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def setup_directories() -> None:
    """
    Create necessary directories for the project.
    
    Args:
        None
        
    Returns:
        None
    """
    directories = ['artifacts', 'figures', 'models','logs']
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        logging.info(f"Dossier '{dir_name}' créé ou déjà existant.")

def setup_logging() -> None:
    """
    Configure logging to record all steps in a file.
    """
    # Create logs directory first
    os.makedirs('logs', exist_ok=True)
    
    log_file = os.path.join('logs', 'process.log')
    
    # Configure logging with both file and console output
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # Test that logging is working
    logging.info("=== Nouvelle exécution du script ===")
    logging.info("Configuration du logging effectuée.")

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame
    """
    logging.info(f"Chargement des données depuis {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.drop('Transaction_ID', axis=1, errors='ignore')
    logging.info("Données chargées avec succès.")
    return df

def perform_eda(df: pd.DataFrame) -> None:
    """
    Perform exploratory data analysis and save figures and summaries.
    
    Args:
        df (pd.DataFrame): Input DataFrame to analyze
        
    Returns:
        None
    """
    logging.info("Début de l'EDA.")
    
    # Sauvegarde du résumé descriptif
    eda_summary = df.describe(include='all')
    summary_path = os.path.join('artifacts', 'eda_summary.csv')
    eda_summary.to_csv(summary_path)
    logging.info(f"Résumé descriptif sauvegardé dans {summary_path}.")

    # Separate numerical and categorical columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    # Analysis for numerical variables
    # Histograms
    df[numeric_cols].hist(bins=30, figsize=(20, 15))
    plt.tight_layout()
    hist_path = os.path.join('figures', 'numeric_histograms.png')
    plt.savefig(hist_path)
    plt.close()
    logging.info(f"Histogrammes sauvegardés dans {hist_path}.")

    # Box plots for each numerical variable
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        df[col].plot(kind='box')
        plt.title(f'Box Plot of {col}')
        plt.xticks(rotation=45)
        box_path = os.path.join('figures', f'boxplot_{col}.png')
        plt.savefig(box_path)
        plt.close()
        logging.info(f"Box plot for {col} saved in {box_path}.")

    # Combined box plot for all numerical variables
    plt.figure(figsize=(20, 15))
    df[numeric_cols].boxplot()
    plt.title('Combined Box Plot of All Numerical Variables')
    plt.xticks(rotation=45)
    box_path = os.path.join('figures', 'numeric_boxplots_combined.png')
    plt.savefig(box_path)
    plt.close()
    logging.info(f"Combined box plots saved in {box_path}.")

    # Correlation heatmap for numerical variables
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
    corr_path = os.path.join('figures', 'correlation_heatmap.png')
    plt.savefig(corr_path)
    plt.close()
    logging.info(f"Heatmap de corrélation sauvegardé dans {corr_path}.")

    # Analysis for categorical variables
    for col in categorical_cols:
        # Bar plots for value counts
        plt.figure(figsize=(10, 6))
        df[col].value_counts().plot(kind='bar')
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        cat_path = os.path.join('figures', f'categorical_{col}_distribution.png')
        plt.savefig(cat_path)
        plt.close()
        logging.info(f"Distribution de {col} sauvegardée dans {cat_path}.")

    # Pairplot for numerical variables
    try:
        pairplot = sns.pairplot(df[numeric_cols])
        pairplot_path = os.path.join('figures', 'pairplot.png')
        pairplot.savefig(pairplot_path)
        plt.close()
        logging.info(f"Pairplot sauvegardé dans {pairplot_path}.")
    except Exception as e:
        logging.error(f"Échec du pairplot : {e}")

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Preprocess data by removing identifiers, transforming categorical variables, and scaling.
    
    Args:
        df (pd.DataFrame): Input DataFrame to preprocess
        
    Returns:
        Tuple[pd.DataFrame, StandardScaler]: 
            - Scaled DataFrame
            - Fitted StandardScaler object
    """
    logging.info("Début du pré-traitement des données.")
    
    # Suppression de la colonne Transaction_ID si présente
    if 'Transaction_ID' in df.columns:
        df = df.drop('Transaction_ID', axis=1)
        logging.info("Colonne Transaction_ID supprimée.")

    # Transformation des variables catégorielles en variables indicatrices
    categorical_cols = ['Day_of_Week', 'Time_of_Day', 'Gender', 'Account_Type']
    df = pd.get_dummies(df, columns=categorical_cols)
    logging.info("Variables catégorielles transformées en variables indicatrices.")

    # Sauvegarde des noms de colonnes
    columns_path = os.path.join('models', 'feature_names.json')
    with open(columns_path, 'w') as f:
        json.dump(list(df.columns), f)
    logging.info(f"Noms des features sauvegardés dans {columns_path}")

    # Mise à l'échelle
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    X_scaled = pd.DataFrame(X_scaled, columns=df.columns)
    logging.info("Mise à l'échelle effectuée.")
    
    return X_scaled, scaler

def train_test_data(X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and test sets.
    
    Args:
        X (pd.DataFrame): Input features DataFrame
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - Training set
            - Test set
    """
    logging.info("Division des données en train et test.")
    # Ensure we keep the DataFrame format and column names
    indices = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)
    X_train = X.iloc[indices[0]]
    X_test = X.iloc[indices[1]]
    logging.info(f"Ensemble d'entraînement : {X_train.shape}, ensemble de test : {X_test.shape}.")
    return X_train, X_test

def train_models(X_train: pd.DataFrame) -> Dict[str, Any]:
    """
    Train three anomaly detection models.
    
    Args:
        X_train (pd.DataFrame): Training data
        
    Returns:
        Dict[str, Any]: Dictionary containing trained models with model names as keys
    """
    logging.info("Entraînement des modèles de détection d'anomalies.")
    models = {}

    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_forest.fit(X_train)
    models['IsolationForest'] = iso_forest
    logging.info("IsolationForest entraîné.")

    # One-Class SVM
    oc_svm = OneClassSVM(nu=0.05, kernel="rbf", gamma='scale')
    oc_svm.fit(X_train)
    models['OneClassSVM'] = oc_svm
    logging.info("OneClassSVM entraîné.")

    # Local Outlier Factor (avec novelty=True pour permettre la prédiction sur de nouvelles données)
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.05)
    lof.fit(X_train)
    models['LocalOutlierFactor'] = lof
    logging.info("LocalOutlierFactor entraîné.")

    return models

def evaluate_models(models: Dict[str, Any], X_test: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate each model on the test set by calculating the proportion of detected anomalies.
    
    Args:
        models (Dict[str, Any]): Dictionary of trained models
        X_test (pd.DataFrame): Test data
        
    Returns:
        Dict[str, float]: Dictionary containing evaluation results for each model
    """
    logging.info("Évaluation des modèles sur l'ensemble de test.")
    evaluations = {}
    
    # Create evaluation directory if it doesn't exist
    eval_dir = os.path.join('artifacts', 'evaluations')
    os.makedirs(eval_dir, exist_ok=True)
    
    for name, model in models.items():
        try:
            y_pred = model.predict(X_test)
            # Calculate proportion of anomalies (anomalies are marked as -1)
            prop_outliers = np.mean(y_pred == -1)
            evaluations[name] = prop_outliers
            
            # Create detailed evaluation metrics
            eval_metrics = {
                'model_name': name,
                'anomaly_proportion': float(prop_outliers),  # Convert to native Python float
                'total_samples': int(len(X_test)),  # Convert to native Python int
                'detected_anomalies': int(np.sum(y_pred == -1)),  # Convert numpy values to native Python types
                'normal_samples': int(np.sum(y_pred == 1)),
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save evaluation metrics to JSON file
            eval_file = os.path.join(eval_dir, f'{name}_evaluation.json')
            with open(eval_file, 'w') as f:
                json.dump(eval_metrics, f, indent=4)
            
            # Create and save distribution plot of predictions
            plt.figure(figsize=(10, 6))
            plt.hist(y_pred, bins=2, alpha=0.7)
            plt.title(f'Distribution of Predictions - {name}')
            plt.xlabel('Prediction (-1: Anomaly, 1: Normal)')
            plt.ylabel('Count')
            plt.savefig(os.path.join(eval_dir, f'{name}_prediction_dist.png'))
            plt.close()
            
            logging.info(f"{name} détecte {prop_outliers*100:.2f}% d'anomalies.")
            logging.info(f"Evaluation metrics saved to {eval_file}")
            
        except Exception as e:
            logging.error(f"Échec de l'évaluation pour {name} : {e}")
            evaluations[name] = None
            
            # Save error information
            error_info = {
                'model_name': name,
                'error_message': str(e),
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            error_file = os.path.join(eval_dir, f'{name}_error.json')
            with open(error_file, 'w') as f:
                json.dump(error_info, f, indent=4)
    
    # Save summary of all evaluations
    summary = {
        'evaluation_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_results': {
            name: float(value) if value is not None else None 
            for name, value in evaluations.items()
        }
    }
    with open(os.path.join(eval_dir, 'evaluation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    return evaluations

def select_best_model(models: Dict[str, Any], evaluations: Dict[str, float]) -> Tuple[Any, str]:
    logging.info("Comparaison des modèles pour sélectionner le meilleur.")
    
    # Add validation for empty or None evaluations
    valid_evaluations = {k: v for k, v in evaluations.items() if v is not None}
    if not valid_evaluations:
        raise ValueError("No valid model evaluations available. All models failed.")
    
    best_model_name = min(valid_evaluations, key=lambda k: valid_evaluations[k])
    best_model = models[best_model_name]
    logging.info(f"Modèle sélectionné : {best_model_name} avec {valid_evaluations[best_model_name]*100:.2f}% d'anomalies détectées.")
    
    # Save model selection results to artifacts
    selection_results = {
        'best_model_name': best_model_name,
        'anomaly_percentage': float(valid_evaluations[best_model_name]*100),  # Convert to native Python float
        'all_models_results': {
            name: float(evaluations[name]*100) if evaluations[name] is not None else None 
            for name in evaluations
        },
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Create artifacts directory if it doesn't exist
    artifacts_dir = 'artifacts'
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Save selection results to JSON file
    selection_file = os.path.join(artifacts_dir, 'model_selection_results.json')
    with open(selection_file, 'w') as f:
        json.dump(selection_results, f, indent=4)
    
    logging.info(f"Model selection results saved to {selection_file}")
    
    return best_model, best_model_name

def save_models(models: Dict[str, Any], best_model: Any, best_model_name: str, scaler: StandardScaler) -> None:
    """
    Save all models, mark the best model as 'best_model.pkl', and save the scaler.
    
    Args:
        models (Dict[str, Any]): Dictionary of trained models
        best_model (Any): Best performing model
        best_model_name (str): Name of the best model
        scaler (StandardScaler): Fitted scaler used for data preprocessing
        
    Returns:
        None
    """
    logging.info("Sauvegarde de tous les modèles et du scaler.")
    
    # Save all models
    for name, model in models.items():
        model_path = os.path.join('models', f"{name}.pkl")
        joblib.dump(model, model_path)
        logging.info(f"Modèle {name} sauvegardé dans {model_path}.")

    # Save best model
    best_model_path = os.path.join('models', "best_model.pkl")
    joblib.dump(best_model, best_model_path)
    logging.info(f"Meilleur modèle ({best_model_name}) sauvegardé dans {best_model_path}.")
    
    # Save scaler
    scaler_path = os.path.join('models', "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logging.info(f"Scaler sauvegardé dans {scaler_path}.")


def copy_model_to_api(api_dir: str = 'api') -> None:
        """
        Copy the best model and scaler to the API directory.
        
        Args:
            api_dir (str): Path to the API directory
            
        Returns:
            None
        """
        logging.info("Copying model and scaler to API directory")
        
        # Create API model directory if it doesn't exist
        api_models_dir = os.path.join(api_dir, 'models')
        os.makedirs(api_models_dir, exist_ok=True)
        
        try:
            # Copy best model
            source_model = os.path.join('models', 'best_model.pkl')
            dest_model = os.path.join(api_models_dir, 'best_model.pkl')
            joblib.dump(joblib.load(source_model), dest_model)
            
            # Copy scaler
            source_scaler = os.path.join('models', 'scaler.pkl')
            dest_scaler = os.path.join(api_models_dir, 'scaler.pkl')
            joblib.dump(joblib.load(source_scaler), dest_scaler)
            
            # Copy feature names
            source_features = os.path.join('models', 'feature_names.json')
            dest_features = os.path.join(api_models_dir, 'feature_names.json')
            with open(source_features, 'r') as f_src, open(dest_features, 'w') as f_dst:
                json.dump(json.load(f_src), f_dst)
                
            logging.info("Model, scaler and features copied successfully to API directory")
        except Exception as e:
            logging.error(f"Error copying files to API directory: {e}")
            raise

def main() -> None:
    """
    Main execution function that orchestrates the entire anomaly detection process.
    
    Args:
        None
        
    Returns:
        None
    """
    try:
        # Création des dossiers et configuration du logging
        setup_directories()
        setup_logging()
        logging.info("Démarrage du script.")

        # Chargement des données (adapter le chemin vers votre fichier CSV)
        csv_path = 'data.csv'
        df = load_data(csv_path)

        # Analyse exploratoire des données
        perform_eda(df)

        # Pré-traitement des données
        X_scaled, scaler = preprocess_data(df)

        # Division en ensembles d'entraînement et de test
        X_train, X_test = train_test_data(X_scaled)

        # Entraînement des modèles
        models = train_models(X_train)

        # Évaluation des modèles sur l'ensemble de test
        evaluations = evaluate_models(models, X_test)

        # Sélection du meilleur modèle
        best_model, best_model_name = select_best_model(models, evaluations)

        # Sauvegarde de tous les modèles et du meilleur modèle
        save_models(models, best_model, best_model_name, scaler)

        logging.info("Script terminé avec succès.")
        
        # Copy the best model and scaler to the API directory
        copy_model_to_api()
        
    except Exception as e:
        logging.exception("Une erreur est survenue pendant le processus.")

if __name__ == "__main__":
    main()
