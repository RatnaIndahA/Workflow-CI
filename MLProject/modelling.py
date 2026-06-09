import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Gunakan path relatif untuk tracking
mlflow.set_tracking_uri("./mlruns")

# Load dataset
df = pd.read_csv("adult_preprocessing.csv")
X = df.drop("income", axis=1)
y = df["income"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("roc_auc", auc)
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    # Simpan model lokal (optional, tapi tidak perlu di-log lagi)
    joblib.dump(model, "model.pkl")
    
    print(f"Accuracy: {acc:.4f}, AUC: {auc:.4f}")