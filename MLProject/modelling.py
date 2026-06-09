# MLProject/modelling.py
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

def main():
    # Load data
    data_path = "adult_preprocessing.csv"
    df = pd.read_csv(data_path)
    X = df.drop(columns=['income'])
    y = df['income']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label='>50K')
    rec = recall_score(y_test, y_pred, pos_label='>50K')
    f1 = f1_score(y_test, y_pred, pos_label='>50K')
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"Accuracy: {acc:.4f}, AUC: {auc:.4f}")
    
    # Save model locally
    os.makedirs("models", exist_ok=True)
    model_path = "models/model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # MLflow logging (opsional, bisa juga diabaikan jika tidak ingin setup MLflow di CI)
    # Tapi agar memenuhi, kita akan log ke file lokal
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("CI_Retraining")
    with mlflow.start_run(run_name="Retrain_CI"):
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)
        mlflow.sklearn.log_model(model, "random_forest_model")
        mlflow.log_artifact(model_path)
    
    print("Training selesai. MLflow logs saved in ./mlruns")

if __name__ == "__main__":
    main()