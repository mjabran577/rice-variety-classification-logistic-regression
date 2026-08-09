"""
Rice Variety Classification Using Logistic Regression

Dataset:
UCI Rice (Cammeo and Osmancik) dataset
https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik

This script reproduces the main analysis used in this portfolio project.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.io import arff
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, train_test_split


RANDOM_STATE = 732084
DATA_PATH = Path("data/Rice_Cammeo_Osmancik.arff")
FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    """Load the UCI ARFF dataset into a pandas DataFrame."""
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    df["Class"] = df["Class"].str.decode("utf-8")
    return df


def create_eda_figures(df: pd.DataFrame) -> None:
    """Create class distribution, area boxplot and correlation heatmap."""
    df["Class"].value_counts().plot(kind="bar")
    plt.title("Rice Variety Distribution")
    plt.xlabel("Rice Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "class_distribution.png", dpi=300)
    plt.close()

    df.boxplot(column="Area", by="Class", figsize=(8, 5))
    plt.title("Area by Rice Type")
    plt.suptitle("")
    plt.ylabel("Area")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "area_boxplot.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(df.drop(columns="Class").corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "correlation_heatmap.png", dpi=300)
    plt.close()


def train_and_evaluate(df: pd.DataFrame) -> None:
    """Train Logistic Regression and print evaluation metrics."""
    X = df.drop(columns="Class")
    y = df["Class"].map({"Cammeo": 0, "Osmancik": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cv_scores = cross_val_score(
        LogisticRegression(max_iter=1000),
        X,
        y,
        cv=5,
        scoring="accuracy",
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Mean 5-fold CV accuracy: {cv_scores.mean():.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "roc_curve.png", dpi=300)
    plt.close()


def main() -> None:
    df = load_data(DATA_PATH)
    create_eda_figures(df)
    train_and_evaluate(df)


if __name__ == "__main__":
    main()
