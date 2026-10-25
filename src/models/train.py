import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.lightgbm
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

def train_model(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, stratify=y
    )

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        eval_metric = "logloss"
    )

    with mlflow.start_run():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds)

        mlflow.log_param("n_estimators", 300)
        mlflow.log_metric("accurancy", acc)
        mlflow.log_metric("recall", rec)
        mlflow.lightgbm.log_model(model, "model")

        train_ds = mlflow.data.from_pandas(df, source = "training_data")
        mlflow.log_input(train_ds, context = "training")

        print(f"Model trained. Accurancy: {acc:.4f}, Recall: {rec:.4f}")

