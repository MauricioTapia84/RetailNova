"""Entrenamiento y evaluación: preprocesamiento ajustado dentro del pipeline."""
import json
import numpy as np
import pandas as pd
from paths import (
    DATA_QUALITY_PATH,
    GRAPHS_DIR,
    MODEL_COMPARISON_PATH,
    MODEL_COEFFICIENTS_PATH,
    MODEL_DIR,
    MODEL_PATH,
    PARTITIONS_PATH,
    PROCESSED_DIR,
    PROCESSED_TEST_DIR,
    PROCESSED_TRAIN_DIR,
    RAW_TEST_DIR,
    RAW_TRAIN_DIR,
    ROOT,
    SCORED_TEST_PATH,
    REGIONAL_AUDIT_PATH,
    TEST_METRICS_PATH,
    TEST_PROCESSED_PATH,
    TEST_RAW_PATH,
    TRAIN_PROCESSED_PATH,
    TRAIN_RAW_PATH,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
import joblib

FEATURES = [
    "edad", "canal_compra", "antiguedad_meses", "cantidad_compras_12m",
    "monto_promedio_compra", "categoria_principal", "proporcion_compras_promocion",
    "devoluciones_12m", "reclamos_12m", "interacciones_campanas_12m",
    "recencia_dias", "tasa_devoluciones",
]
PALETTE = {"Sin abandono": "#0072B2", "Abandono": "#D55E00", "Neutro": "#009E73"}


def _preprocessing(numeric_features, categorical_features):
    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", RobustScaler()),
        ]), numeric_features),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_features),
    ])


def _add_engineered_features(frame):
    result = frame.copy()
    result["tasa_devoluciones"] = result["devoluciones_12m"] / result["cantidad_compras_12m"].clip(lower=1)
    return result


def _save_figure(figure, path):
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run():
    for directory in (
        PROCESSED_DIR, PROCESSED_TRAIN_DIR, PROCESSED_TEST_DIR,
        RAW_TRAIN_DIR, RAW_TEST_DIR, MODEL_DIR, GRAPHS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    if not TRAIN_RAW_PATH.is_file() or not TEST_RAW_PATH.is_file():
        raise FileNotFoundError(
            "Ejecute primero notebooks/02_analisis_exploratorio.ipynb para "
            "crear data/raw/train/train.csv y data/raw/test/test.csv."
        )

    train_df = _add_engineered_features(pd.read_csv(TRAIN_RAW_PATH))
    test_df = _add_engineered_features(pd.read_csv(TEST_RAW_PATH))
    missing_train = set(FEATURES + ["churn_6m", "id_cliente", "region"]) - set(train_df.columns)
    missing_test = set(FEATURES + ["churn_6m", "id_cliente", "region"]) - set(test_df.columns)
    if missing_train or missing_test:
        raise ValueError(f"Faltan columnas requeridas; train={sorted(missing_train)}, test={sorted(missing_test)}")

    X = train_df[FEATURES]
    y = train_df["churn_6m"].astype(int)
    train_idx, valid_idx = train_test_split(
        train_df.index, test_size=0.2, stratify=y, random_state=42
    )
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = [column for column in FEATURES if column not in numeric_features]
    models = {
        "Baseline": DummyClassifier(strategy="prior"),
        "Logistica": LogisticRegression(max_iter=500, random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=60, max_depth=10, min_samples_leaf=40, n_jobs=2, random_state=42
        ),
    }

    fitted = {}
    rows = []
    for name, estimator in models.items():
        candidate = Pipeline([
            ("preprocess", _preprocessing(numeric_features, categorical_features)),
            ("model", estimator),
        ])
        candidate.fit(X.loc[train_idx], y.loc[train_idx])
        validation_probabilities = candidate.predict_proba(X.loc[valid_idx])[:, 1]
        rows.append({
            "modelo": name,
            "validacion_roc_auc": roc_auc_score(y.loc[valid_idx], validation_probabilities),
            "validacion_average_precision": average_precision_score(y.loc[valid_idx], validation_probabilities),
        })
        fitted[name] = candidate

    comparison = pd.DataFrame(rows).sort_values("validacion_average_precision", ascending=False)
    best_name = comparison.iloc[0].modelo
    validation_pipeline = fitted[best_name]
    validation_probabilities = validation_pipeline.predict_proba(X.loc[valid_idx])[:, 1]
    thresholds = np.arange(0.1, 0.81, 0.01)
    threshold = float(max(
        thresholds,
        key=lambda value: f1_score(y.loc[valid_idx], validation_probabilities >= value),
    ))

    # Fit the selected pipeline on full train; use holdout only for final evaluation.
    pipeline = Pipeline([
        ("preprocess", _preprocessing(numeric_features, categorical_features)),
        ("model", models[best_name]),
    ])
    pipeline.fit(X, y)
    test_X = test_df[FEATURES]
    test_probabilities = pipeline.predict_proba(test_X)[:, 1]
    test_predictions = test_probabilities >= threshold
    test_truth = test_df["churn_6m"].astype(int)

    metrics = {
        "modelo_seleccionado": best_name,
        "umbral_elegido_en_validacion": threshold,
        "train": len(train_idx),
        "validacion": len(valid_idx),
        "train_completo_refit": len(train_df),
        "test": len(test_df),
        "accuracy": accuracy_score(test_truth, test_predictions),
        "precision": precision_score(test_truth, test_predictions, zero_division=0),
        "recall": recall_score(test_truth, test_predictions),
        "f1": f1_score(test_truth, test_predictions),
        "roc_auc": roc_auc_score(test_truth, test_probabilities),
        "average_precision": average_precision_score(test_truth, test_probabilities),
        "matriz_confusion_tn_fp_fn_tp": confusion_matrix(test_truth, test_predictions).tolist(),
        "advertencia": "Evaluación sobre datos sintéticos, sin validación en clientes reales.",
    }

    comparison.to_csv(MODEL_COMPARISON_PATH, index=False)
    TEST_METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    partitions = pd.DataFrame({
        "id_cliente": train_df["id_cliente"],
        "particion": np.where(train_df.index.isin(valid_idx), "validacion", "train"),
    })
    holdout_partitions = pd.DataFrame({"id_cliente": test_df["id_cliente"], "particion": "test"})
    pd.concat([partitions, holdout_partitions], ignore_index=True).to_csv(
        PARTITIONS_PATH, index=False
    )

    scored = pd.DataFrame({
        "id_cliente": test_df["id_cliente"],
        "region": test_df["region"],
        "probabilidad_abandono": test_probabilities,
        "abandono_observado": test_truth,
        "prediccion": test_predictions.astype(int),
    })
    scored["nivel_riesgo"] = pd.cut(
        test_probabilities, [-0.001, 0.2, 0.5, 1.001], labels=["Bajo", "Medio", "Alto"]
    )
    scored.to_csv(SCORED_TEST_PATH, index=False)
    regional_audit = []
    for region, group in scored.groupby("region"):
        regional_audit.append({
            "region": region,
            "clientes_test": len(group),
            "tasa_abandono": group.abandono_observado.mean(),
            "recall": recall_score(group.abandono_observado, group.prediccion),
            "precision": precision_score(group.abandono_observado, group.prediccion, zero_division=0),
            "roc_auc": roc_auc_score(group.abandono_observado, group.probabilidad_abandono),
        })
    regional_df = pd.DataFrame(regional_audit)
    regional_df.to_csv(REGIONAL_AUDIT_PATH, index=False)

    fitted_preprocessor = pipeline.named_steps["preprocess"]
    feature_names = fitted_preprocessor.get_feature_names_out()
    train_matrix = fitted_preprocessor.transform(X)
    test_matrix = fitted_preprocessor.transform(test_X)
    if hasattr(train_matrix, "toarray"):
        train_matrix = train_matrix.toarray()
        test_matrix = test_matrix.toarray()
    train_processed = pd.DataFrame(train_matrix, columns=feature_names, index=train_df.index)
    train_processed.insert(0, "id_cliente", train_df["id_cliente"].to_numpy())
    train_processed["churn_6m"] = y.to_numpy()
    test_processed = pd.DataFrame(test_matrix, columns=feature_names, index=test_df.index)
    test_processed.insert(0, "id_cliente", test_df["id_cliente"].to_numpy())
    test_processed["churn_6m"] = test_truth.to_numpy()
    train_processed.to_csv(TRAIN_PROCESSED_PATH, index=False)
    test_processed.to_csv(TEST_PROCESSED_PATH, index=False)

    joblib.dump(
        {"pipeline": pipeline, "features": FEATURES, "threshold": threshold, "nivel_riesgo_cortes": [0.2, 0.5]},
        MODEL_PATH,
    )

    class_counts = train_df["churn_6m"].value_counts().reindex([0, 1], fill_value=0)
    class_labels = ["Sin abandono", "Abandono"]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    bars = axis.bar(class_labels, class_counts.to_numpy(), color=[PALETTE["Sin abandono"], PALETTE["Abandono"]], label="Clientes train")
    axis.bar_label(bars, labels=[f"{value:,.0f} ({value / len(train_df):.1%})" for value in class_counts], padding=4, fontsize=9)
    axis.set(title="Distribución de abandono en entrenamiento", xlabel="Resultado a seis meses", ylabel="Clientes")
    axis.legend(frameon=False)
    _save_figure(figure, GRAPHS_DIR / "distribucion_objetivo.png")

    region_counts = train_df["region"].value_counts().sort_index()
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    region_bars = axes[0].bar(region_counts.index, region_counts.to_numpy(), color="#009E73", label="Clientes train")
    axes[0].bar_label(region_bars, labels=[f"{v:,.0f}\n{v / len(train_df):.1%}" for v in region_counts], padding=3, fontsize=8)
    axes[0].set(title="Composición de la cohorte", xlabel="Región", ylabel="Clientes")
    axes[0].legend(frameon=False)
    rates = train_df.groupby("region")["churn_6m"].mean().reindex(region_counts.index)
    rate_bars = axes[1].bar(rates.index, rates.to_numpy() * 100, color="#D55E00", label="Tasa de abandono")
    axes[1].bar_label(rate_bars, labels=[f"{value:.2f}%" for value in rates * 100], padding=3, fontsize=8)
    axes[1].set(title="Abandono observado por región", xlabel="Región", ylabel="Clientes sin compras (%)")
    axes[1].legend(frameon=False)
    _save_figure(figure, GRAPHS_DIR / "resumen_regional.png")

    matrix = confusion_matrix(test_truth, test_predictions)
    row_percentages = matrix / matrix.sum(axis=1, keepdims=True) * 100
    figure, axis = plt.subplots(figsize=(6.6, 5.2))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, label="Clientes")
    for row in range(2):
        for column in range(2):
            axis.text(
                column, row,
                f"{matrix[row, column]:,}\n{row_percentages[row, column]:.1f}% de la clase real",
                ha="center", va="center", color="black", fontsize=9,
            )
    axis.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["Sin abandono", "Abandono"],
        yticklabels=["Sin abandono", "Abandono"],
        xlabel="Predicción", ylabel="Resultado real",
        title="Matriz de confusión — holdout",
    )
    _save_figure(figure, GRAPHS_DIR / "matriz_confusion.png")

    if best_name == "Logistica":
        coefficients = pipeline.named_steps["model"].coef_[0]
        pd.DataFrame({
            "variable_transformada": feature_names,
            "coeficiente": coefficients,
        }).sort_values("coeficiente").to_csv(MODEL_COEFFICIENTS_PATH, index=False)

    quality = {
        "filas_train": len(train_df),
        "columnas_train": len(train_df.columns),
        "ids_unicos": bool(train_df.id_cliente.is_unique),
        "duplicados": int(train_df.duplicated().sum()),
        "nulos_por_columna": train_df.isna().sum().to_dict(),
    }
    DATA_QUALITY_PATH.write_text(
        json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(comparison.to_string(index=False))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Gráficos guardados en: {GRAPHS_DIR.relative_to(ROOT)}")
    print(f"Artefactos del modelo guardados en: {MODEL_DIR.relative_to(ROOT)}")
    print(f"Datos procesados guardados en: {PROCESSED_DIR.relative_to(ROOT)}")
    return metrics


if __name__ == "__main__":
    run()
