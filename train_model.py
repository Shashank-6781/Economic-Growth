from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "world.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

TARGET = "GDP ($ per capita)"

def impute_missing_values(df):
    df = df.copy()
    numeric = df.select_dtypes(include=np.number).columns
    categorical = df.select_dtypes(exclude=np.number).columns

    for col in numeric:
        if col == TARGET:
            continue
        if df[col].isna().any():
            df[col] = df.groupby("Region")[col].transform(
                lambda s: s.fillna(s.median())
            )
            df[col] = df[col].fillna(df[col].median())

    if "Climate" in df.columns and df["Climate"].isna().any():
        df["Climate"] = df.groupby("Region")["Climate"].transform(
            lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else np.nan)
        )
        if df["Climate"].isna().any():
            df["Climate"] = df["Climate"].fillna(df["Climate"].mode().iloc[0])

    for col in categorical:
        if df[col].isna().any():
            mode = df[col].mode()
            if not mode.empty:
                df[col] = df[col].fillna(mode.iloc[0])
    return df

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing dataset: {DATA_PATH}. Add the original world.csv first."
        )

    data = pd.read_csv(DATA_PATH, decimal=",")
    data = impute_missing_values(data).dropna(subset=[TARGET])

    X = data.drop(columns=[TARGET, "Country"], errors="ignore")
    y = data[TARGET]

    encoders = {}
    for col in X.select_dtypes(exclude=np.number).columns:
        enc = LabelEncoder()
        X[col] = enc.fit_transform(X[col].astype(str))
        encoders[col] = enc

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    candidates = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=300, random_state=42
        ),
    }

    rows = []
    fitted = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        rows.append({
            "Model": name,
            "Train_R2": r2_score(y_train, train_pred),
            "Test_R2": r2_score(y_test, test_pred),
            "RMSE": mean_squared_error(y_test, test_pred, squared=False)
        })
        fitted[name] = model

    metrics = pd.DataFrame(rows).sort_values("RMSE")
    metrics.to_csv(REPORT_DIR / "model_metrics.csv", index=False)

    # The assigned project specifies Random Forest as the deployment model.
    artifact = {
        "model": fitted["Random Forest Regressor"],
        "feature_columns": list(X.columns),
        "encoders": encoders,
        "target": TARGET,
    }
    joblib.dump(artifact, MODEL_DIR / "gdp.pkl")

    print(metrics.to_string(index=False))
    print("Saved:", MODEL_DIR / "gdp.pkl")

if __name__ == "__main__":
    main()
