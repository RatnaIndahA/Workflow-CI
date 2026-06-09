import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Izinkan file store (meskipun deprecated)
mlflow.set_tracking_uri("./mlruns")

df = pd.read_csv("adult_preprocessing.csv")
X = df.drop("income", axis=1)
y = df["income"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

with mlflow.start_run(run_name="Retrain_CI"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("roc_auc", auc)
    
    # Simpan model
    joblib.dump(model, "model.pkl")
    mlflow.log_artifact("model.pkl")
    
    # Artefak tambahan untuk advance (confusion matrix)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    
    print(f"Accuracy: {acc:.4f}, AUC: {auc:.4f}")