import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow


def filtrar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """Seleciona colunas relevantes e remove linhas com valores nulos."""
    colunas = [
        "lat", "lon", "minutes_remaining", "period",
        "playoffs", "shot_distance", "shot_made_flag"
    ]
    df_filtrado = df[colunas].dropna(subset=["shot_made_flag"])
    return df_filtrado


def separar_treino_teste(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa os dados em treino e teste de forma e loga no MLflow."""
    X = df.drop(columns=["shot_made_flag"])
    y = df["shot_made_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Log no MLflow
    mlflow.log_param("test_size", test_size)
    mlflow.log_metric("train_size", len(X_train))
    mlflow.log_metric("test_size", len(X_test))

    # Junta novamente X + y pra salvar como parquet
    train_df = X_train.copy()
    train_df["shot_made_flag"] = y_train

    test_df = X_test.copy()
    test_df["shot_made_flag"] = y_test

    return train_df, test_df