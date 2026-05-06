import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("Telco-Customer-Churn.csv")

# Corrigir tipos inconsistentes
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Regressão linear para preencher valores nulos de TotalCharges
reg_df = df[['tenure', 'MonthlyCharges', 'TotalCharges']].copy()
predict_df = reg_df[reg_df['TotalCharges'].isnull()]
train_df = reg_df[reg_df['TotalCharges'].notnull()]

if not predict_df.empty and not train_df.empty:
    X_train_reg = train_df[['tenure', 'MonthlyCharges']].fillna(0)
    y_train_reg = train_df['TotalCharges']
    X_pred = predict_df[['tenure', 'MonthlyCharges']].fillna(0)

    reg = LinearRegression()
    reg.fit(X_train_reg, y_train_reg)
    predicted_values = reg.predict(X_pred)

    df.loc[df['TotalCharges'].isnull(), 'TotalCharges'] = predicted_values

# Remover colunas irrelevantes
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

# Codificar target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Separar features e alvo
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Identificar colunas por tipo
numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

#  Pipeline final com modelo
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Treinar e avaliar
model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
