import pandas as pd
import numpy as np
import os

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from lightgbm import LGBMClassifier

import joblib

os.makedirs("outputs/graphs", exist_ok=True)
os.makedirs("model", exist_ok=True)

df = pd.read_csv("data/sleep_health_dataset.csv")

print("Dataset Shape:", df.shape)

df.drop(columns=["Person ID", "headache type"], inplace=True)

df = df.drop_duplicates()

df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")

df["BMI Category"] = df["BMI Category"].replace("Normal Weight", "Normal")

df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True)

df['Systolic'] = df['Systolic'].astype(int)
df['Diastolic'] = df['Diastolic'].astype(int)

df.drop(columns=["Blood Pressure"], inplace=True)

target_encoder = LabelEncoder()

df["Headache"] = target_encoder.fit_transform(df["headache"])

categorical_cols = [
    "Gender",
    "Occupation",
    "BMI Category",
    "Sleep Disorder"
]

le = LabelEncoder()

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

df["Sleep_Stress_Index"] = df["Stress Level"] / df["Sleep Duration"]

features = [
    "Age",
    "Sleep Duration",
    "Quality of Sleep",
    "Physical Activity Level",
    "Stress Level",
    "Heart Rate",
    "Daily Steps",
    "Systolic",
    "Diastolic",
    "Sleep_Stress_Index"
]

X = df[features]
y = df["Headache"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")

scores = cross_val_score(svm, X, y, cv=5)

print("\nCross Validation Accuracy:", scores.mean())

param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", 0.1, 0.01],
    "kernel": ["rbf"]
}

grid = GridSearchCV(SVC(), param_grid, cv=5)

grid.fit(X_train, y_train)

print("\nBest Parameters:", grid.best_params_)

models = {
    "SVM": SVC(kernel="rbf", C=10, gamma="scale"),
    "Random Forest": RandomForestClassifier(n_estimators=200),
    "KNN": KNeighborsClassifier(),
    "LightGBM": LGBMClassifier()
}

results = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    results[name] = acc

    print("\n==============================")

    print(name)

    print("Accuracy:", acc)

    print(classification_report(y_test, pred))

best_model_name = max(results, key=results.get)

best_model = models[best_model_name]

print("\nBest Model:", best_model_name)

print("Accuracy:", results[best_model_name])

pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(6,5))

sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")

plt.title("Confusion Matrix")

plt.savefig("outputs/graphs/confusion_matrix.png")

plt.close()

joblib.dump(best_model, "model/headache_model.pkl")

print("\nModel saved successfully")

results_df = pd.DataFrame(list(results.items()), columns=["Model","Accuracy"])

results_df.to_csv("outputs/model_results.csv", index=False)

print("\nResults saved")
