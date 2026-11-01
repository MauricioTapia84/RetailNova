"""Experimentos de clustering y modelado supervisado para RetailNova."""
import json
import time
import warnings

import joblib
import matplotlib
import numpy as np
import optuna
import pandas as pd
import shap
from imblearn.ensemble import BalancedRandomForestClassifier
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from paths import (
    ADVANCED_MODEL_PATH,
    ADVANCED_TEST_METRICS_PATH,
    CLASS_BALANCE_RESULTS_PATH,
    CLUSTERING_RESULTS_PATH,
    CLUSTER_MODEL_PATH,
    FINAL_MODEL_SCORED_TEST_PATH,
    GAIN_LIFT_TEST_PATH,
    GRAPHS_DIR,
    MODEL_DIR,
    OPTUNA_RESULTS_PATH,
    PARTITIONS_PATH,
    PROCESSED_DIR,
    PROCESSED_TEST_DIR,
    PROCESSED_TRAIN_DIR,
    SUPERVISED_COMPARISON_PATH,
    TEST_RAW_PATH,
    TRAIN_RAW_PATH,
)

matplotlib.use("Agg")
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

RANDOM_STATE = 42
TARGET = "churn_6m"
FEATURES = [
    "edad",
    "canal_compra",
    "antiguedad_meses",
    "cantidad_compras_12m",
    "monto_promedio_compra",
    "categoria_principal",
    "proporcion_compras_promocion",
    "devoluciones_12m",
    "reclamos_12m",
    "interacciones_campanas_12m",
    "recencia_dias",
    "tasa_devoluciones",
]
FORBIDDEN_FEATURES = {
    "id_cliente",
    "prioridad_atencion",
    TARGET,
    "compras_periodo_objetivo",
    "fecha_ultima_compra_al_cierre",
}


def _ensure_output_directories():
    for directory in (
        PROCESSED_DIR, PROCESSED_TRAIN_DIR, PROCESSED_TEST_DIR,
        MODEL_DIR, GRAPHS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def _add_features(frame):
    result = frame.copy()
    result["tasa_devoluciones"] = (
        result["devoluciones_12m"]
        / result["cantidad_compras_12m"].clip(lower=1)
    )
    return result


def load_training_data():
    """Carga solo el split de entrenamiento para experimentación y selección."""
    if not TRAIN_RAW_PATH.is_file():
        raise FileNotFoundError(
            "Ejecute primero notebooks/02_analisis_exploratorio.ipynb "
            "para crear data/raw/train/train.csv."
        )
    train = pd.read_csv(TRAIN_RAW_PATH)
    _validate_modeling_columns(train, "train")
    return _add_features(train)


def load_test_data():
    """Abre el holdout reservado solo al comenzar la evaluación final."""
    if not TEST_RAW_PATH.is_file():
        raise FileNotFoundError(
            "No se encontró data/raw/test/test.csv; vuelva a ejecutar el split del EDA."
        )
    test = pd.read_csv(TEST_RAW_PATH)
    _validate_modeling_columns(test, "test")
    return _add_features(test)


def _validate_modeling_columns(frame, partition):
    required = set(
        [feature for feature in FEATURES if feature != "tasa_devoluciones"]
        + [TARGET, "id_cliente"]
    )
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas requeridas en {partition}: {sorted(missing)}"
        )
    if FORBIDDEN_FEATURES.intersection(FEATURES):
        raise ValueError("La lista de predictores contiene columnas con riesgo de fuga.")
    if not set(frame[TARGET].dropna().unique()).issubset({0, 1}):
        raise ValueError("La variable objetivo debe estar codificada como 0/1.")


def _make_preprocessor():
    numeric = [column for column in FEATURES if column not in {
        "canal_compra", "categoria_principal",
    }]
    categorical = ["canal_compra", "categoria_principal"]
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", RobustScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def prepare_validation(train):
    """Ajusta transformaciones dentro del subgrupo de entrenamiento interno."""
    X = train[FEATURES]
    y = train[TARGET].astype(int)
    train_idx, valid_idx = train_test_split(
        np.arange(len(train)),
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    preprocessor = _make_preprocessor()
    X_fit = preprocessor.fit_transform(X.iloc[train_idx])
    X_valid = preprocessor.transform(X.iloc[valid_idx])
    return {
        "preprocessor": preprocessor,
        "X_fit": X_fit,
        "X_valid": X_valid,
        "y_fit": y.iloc[train_idx].reset_index(drop=True),
        "y_valid": y.iloc[valid_idx].reset_index(drop=True),
        "X_fit_raw": X.iloc[train_idx].reset_index(drop=True),
        "X_valid_raw": X.iloc[valid_idx].reset_index(drop=True),
        "train_idx": train_idx,
        "valid_idx": valid_idx,
        "feature_names": preprocessor.get_feature_names_out(),
    }


def _balance_estimators():
    return {
        "Logística sin balanceo": LogisticRegression(
            C=1.0, max_iter=500, solver="liblinear", random_state=RANDOM_STATE
        ),
        "Logística con pesos de clase": LogisticRegression(
            C=1.0, max_iter=500, solver="liblinear",
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest con pesos": RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_leaf=20,
            class_weight="balanced_subsample", n_jobs=2, random_state=RANDOM_STATE
        ),
        "Random Forest balanceado": BalancedRandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_leaf=20,
            sampling_strategy="all", replacement=True, n_jobs=2,
            random_state=RANDOM_STATE
        ),
    }


def compare_balance_methods(prepared):
    """Compara balanceos en validación interna, sin sobremuestrear datos."""
    _ensure_output_directories()
    rows = []
    for name, estimator in _balance_estimators().items():
        start = time.perf_counter()
        estimator.fit(prepared["X_fit"], prepared["y_fit"])
        probabilities = estimator.predict_proba(prepared["X_valid"])[:, 1]
        predictions = probabilities >= 0.5
        rows.append(
            {
                "metodo": name,
                "average_precision": average_precision_score(
                    prepared["y_valid"], probabilities
                ),
                "precision_umbral_0_5": precision_score(
                    prepared["y_valid"], predictions, zero_division=0
                ),
                "recall_umbral_0_5": recall_score(
                    prepared["y_valid"], predictions, zero_division=0
                ),
                "tiempo_segundos": time.perf_counter() - start,
            }
        )
    result = pd.DataFrame(rows).sort_values(
        "average_precision", ascending=False
    ).reset_index(drop=True)
    result.to_csv(CLASS_BALANCE_RESULTS_PATH, index=False)
    return result


def _cluster_sample(train, limit=2500):
    if len(train) <= limit:
        return train[FEATURES].copy()
    sample = train.sample(n=limit, random_state=RANDOM_STATE)
    return sample[FEATURES].copy()


def evaluate_clustering(train):
    """Compara clustering sin usar churn y persiste el mejor algoritmo."""
    _ensure_output_directories()
    sample = _cluster_sample(train)
    preprocessor = _make_preprocessor()
    matrix = preprocessor.fit_transform(sample)
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coordinates = pca.fit_transform(matrix)
    rows = []
    fitted_models = {}

    for k in range(2, 9):
        candidate = KMeans(
            n_clusters=k, n_init=10, random_state=RANDOM_STATE
        ).fit(matrix)
        fitted_models[f"KMeans k={k}"] = candidate
        rows.append(_cluster_metrics("KMeans", k, matrix, candidate.labels_))
        rows[-1]["inercia"] = float(candidate.inertia_)

    for k in range(2, 7):
        candidate = GaussianMixture(
            n_components=k, covariance_type="diag", reg_covar=1e-5,
            random_state=RANDOM_STATE,
        ).fit(matrix)
        labels = candidate.predict(matrix)
        fitted_models[f"Gaussian Mixture k={k}"] = candidate
        rows.append(_cluster_metrics("Gaussian Mixture", k, matrix, labels))

    for k in range(2, 7):
        candidate = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = candidate.fit_predict(matrix)
        rows.append(
            _cluster_metrics("Agglomerative", k, matrix, labels)
        )

    result = pd.DataFrame(rows).sort_values(
        ["silhouette", "davies_bouldin"], ascending=[False, True]
    ).reset_index(drop=True)
    result.to_csv(CLUSTERING_RESULTS_PATH, index=False)
    selected = result.iloc[0]
    selected_name = f"{selected['algoritmo']} k={int(selected['k'])}"
    if selected_name not in fitted_models:
        chosen = AgglomerativeClustering(
            n_clusters=int(selected["k"]), linkage="ward"
        ).fit(matrix)
        cluster_labels = chosen.labels_
    else:
        chosen = fitted_models[selected_name]
        cluster_labels = chosen.predict(matrix)
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "pca": pca,
            "model": chosen,
            "algoritmo": selected_name,
            "variables": FEATURES,
        },
        CLUSTER_MODEL_PATH,
    )

    kmeans_rows = result[result["algoritmo"] == "KMeans"]
    best_kmeans = kmeans_rows.sort_values(
        ["silhouette", "davies_bouldin"], ascending=[False, True]
    ).iloc[0]
    kmeans_k = int(best_kmeans["k"])
    kmeans_model = fitted_models[f"KMeans k={kmeans_k}"]
    kmeans_labels = kmeans_model.predict(matrix)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    elbow = result[result["algoritmo"] == "KMeans"].sort_values("k")
    axes[0].plot(elbow["k"], elbow["inercia"], marker="o", color="#0072B2")
    axes[0].set(
        title="Método del codo — K-Means",
        xlabel="Número de clústeres (k)",
        ylabel="Inercia",
    )
    axes[1].scatter(
        coordinates[:, 0], coordinates[:, 1],
        c=kmeans_labels, cmap="tab10", s=12, alpha=0.65,
    )
    axes[1].legend(
        handles=[
            plt.Line2D(
                [], [], marker="o", linestyle="", color=plt.get_cmap("tab10")(cluster),
                label=f"Clúster {cluster}",
            )
            for cluster in range(kmeans_k)
        ],
        title="K-Means",
        frameon=False,
    )
    axes[1].set(
        title=f"K-Means en PCA — k={kmeans_k}",
        xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
        ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
    )
    figure.tight_layout()
    _save_figure(figure, "clustering_codo_pca.png")
    return result, selected_name, kmeans_k


def _cluster_metrics(algorithm, k, values, labels):
    unique_count = len(np.unique(labels))
    if unique_count < 2 or unique_count >= len(labels):
        silhouette = np.nan
        davies_bouldin = np.nan
        calinski_harabasz = np.nan
    else:
        silhouette = silhouette_score(values, labels)
        davies_bouldin = davies_bouldin_score(values, labels)
        calinski_harabasz = calinski_harabasz_score(values, labels)
    return {
        "algoritmo": algorithm,
        "k": k,
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski_harabasz,
    }


def _estimator_for_trial(trial, algorithm):
    if algorithm == "LogisticRegression":
        return LogisticRegression(
            C=trial.suggest_float("C", 0.01, 10.0, log=True),
            class_weight=trial.suggest_categorical(
                "class_weight", [None, "balanced"]
            ),
            solver="liblinear",
            max_iter=500,
            random_state=RANDOM_STATE,
        )
    common = {
        "n_estimators": trial.suggest_int("n_estimators", 60, 140, step=40),
        "max_depth": trial.suggest_categorical("max_depth", [6, 10, 14, None]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 50, step=20),
        "n_jobs": 2,
        "random_state": RANDOM_STATE,
    }
    if algorithm == "RandomForest":
        return RandomForestClassifier(
            **common,
            class_weight=trial.suggest_categorical(
                "class_weight", [None, "balanced", "balanced_subsample"]
            ),
        )
    if algorithm == "ExtraTrees":
        return ExtraTreesClassifier(
            **common,
            class_weight=trial.suggest_categorical(
                "class_weight", [None, "balanced", "balanced_subsample"]
            ),
        )
    if algorithm == "BalancedRandomForest":
        return BalancedRandomForestClassifier(
            **common,
            sampling_strategy="all",
            replacement=True,
        )
    raise ValueError(f"Algoritmo de Optuna desconocido: {algorithm}")


def optimize_models(prepared, n_trials=5):
    """Optimiza familias de modelos usando únicamente train/validación."""
    _ensure_output_directories()
    records = []
    study_objects = {}
    for algorithm in (
        "LogisticRegression",
        "RandomForest",
        "ExtraTrees",
        "BalancedRandomForest",
    ):
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
            study_name=f"RetailNova_{algorithm}",
        )

        def objective(trial):
            estimator = _estimator_for_trial(trial, algorithm)
            estimator.fit(prepared["X_fit"], prepared["y_fit"])
            probabilities = estimator.predict_proba(prepared["X_valid"])[:, 1]
            return average_precision_score(prepared["y_valid"], probabilities)

        study.optimize(objective, n_trials=n_trials, n_jobs=1)
        study_objects[algorithm] = study
        records.append(
            {
                "algoritmo": algorithm,
                "average_precision_validacion": study.best_value,
                "mejores_parametros": json.dumps(
                    study.best_params, ensure_ascii=False
                ),
                "n_trials": len(study.trials),
            }
        )
    result = pd.DataFrame(records).sort_values(
        "average_precision_validacion", ascending=False
    ).reset_index(drop=True)
    result.to_csv(OPTUNA_RESULTS_PATH, index=False)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    histories = [
        study.best_value for study in study_objects.values()
    ]
    axis.bar(
        list(study_objects),
        histories,
        color=["#0072B2", "#009E73", "#E69F00", "#D55E00"],
        label="Mejor Average Precision en validación",
    )
    axis.set(
        title="Mejor resultado de Optuna por familia de modelo",
        ylabel="Average Precision (validación)",
        xlabel="Familia de modelo",
    )
    axis.tick_params(axis="x", rotation=15)
    axis.legend(frameon=False)
    _save_figure(figure, "optuna_comparacion.png")
    return result


def evaluate_supervised_models(optuna_results, prepared):
    """Compara en validación los mejores hiperparámetros de cada familia."""
    rows = []
    for _, model_row in optuna_results.iterrows():
        algorithm = model_row["algoritmo"]
        parameters = json.loads(model_row["mejores_parametros"])
        estimator = _estimator_from_params(algorithm, parameters)
        start = time.perf_counter()
        estimator.fit(prepared["X_fit"], prepared["y_fit"])
        fit_seconds = time.perf_counter() - start
        probabilities = estimator.predict_proba(prepared["X_valid"])[:, 1]
        predictions = probabilities >= 0.5
        y_valid = prepared["y_valid"]
        rows.append(
            {
                "modelo": algorithm,
                "average_precision": average_precision_score(
                    y_valid, probabilities
                ),
                "roc_auc": roc_auc_score(y_valid, probabilities),
                "precision_umbral_0_5": precision_score(
                    y_valid, predictions, zero_division=0
                ),
                "recall_umbral_0_5": recall_score(
                    y_valid, predictions, zero_division=0
                ),
                "f1_umbral_0_5": f1_score(
                    y_valid, predictions, zero_division=0
                ),
                "tiempo_ajuste_segundos": fit_seconds,
                "parametros": model_row["mejores_parametros"],
            }
        )
    result = pd.DataFrame(rows).sort_values(
        "average_precision", ascending=False
    ).reset_index(drop=True)
    result.to_csv(SUPERVISED_COMPARISON_PATH, index=False)
    return result


def select_model(optuna_results, prepared):
    """Selecciona algoritmo, parámetros y umbral con validación interna."""
    winner = optuna_results.iloc[0]
    algorithm = winner["algoritmo"]
    parameters = json.loads(winner["mejores_parametros"])
    estimator = _estimator_from_params(algorithm, parameters)
    estimator.fit(prepared["X_fit"], prepared["y_fit"])
    probabilities = estimator.predict_proba(prepared["X_valid"])[:, 1]
    precision, recall, thresholds = precision_recall_curve(
        prepared["y_valid"], probabilities
    )
    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    )
    threshold = float(thresholds[int(np.argmax(f1_scores))]) if len(thresholds) else 0.5
    return {
        "algoritmo": algorithm,
        "parametros": parameters,
        "umbral": threshold,
        "average_precision_validacion": float(winner["average_precision_validacion"]),
        "f1_validacion": float(np.max(f1_scores)) if len(f1_scores) else np.nan,
    }


def _estimator_from_params(algorithm, parameters):
    params = dict(parameters)
    params["random_state"] = RANDOM_STATE
    if algorithm == "LogisticRegression":
        params.update(solver="liblinear", max_iter=500)
        return LogisticRegression(**params)
    params["n_jobs"] = 2
    if algorithm == "RandomForest":
        return RandomForestClassifier(**params)
    if algorithm == "ExtraTrees":
        return ExtraTreesClassifier(**params)
    if algorithm == "BalancedRandomForest":
        params.update(sampling_strategy="all", replacement=True)
        return BalancedRandomForestClassifier(**params)
    raise ValueError(f"Algoritmo seleccionado desconocido: {algorithm}")


def evaluate_final_test(train, test, selection, prepared):
    """Calibra con CV solo en fit; fija umbral en validación y evalúa test.

    Test ya fue inspeccionado en versiones anteriores: esta es una reevaluación
    del prototipo, no un nuevo holdout independiente.
    """
    _ensure_output_directories()
    estimator = _estimator_from_params(
        selection["algoritmo"], selection["parametros"]
    )
    pipeline = Pipeline(
        [("preprocess", _make_preprocessor()), ("model", estimator)]
    )
    X_train = train[FEATURES]
    y_train = train[TARGET].astype(int)
    X_test = test[FEATURES]
    y_test = test[TARGET].astype(int)
    # Ensemble=False: calibrador con predicciones out-of-fold; pipeline base
    # reajustado solo en los 118400 clientes de fit. Validación no se incorpora.
    fit_idx, valid_idx = prepared["train_idx"], prepared["valid_idx"]
    pipeline = CalibratedClassifierCV(
        estimator=pipeline, method="sigmoid", ensemble=False,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=1,
    )
    pipeline.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
    base_pipeline = pipeline.calibrated_classifiers_[0].estimator
    vp = pipeline.predict_proba(X_train.iloc[valid_idx])[:, 1]
    vy = y_train.iloc[valid_idx]
    v_precision, v_recall, v_thresholds = precision_recall_curve(vy, vp)
    vf1 = 2 * v_precision[:-1] * v_recall[:-1] / np.maximum(v_precision[:-1] + v_recall[:-1], 1e-12)
    selection = dict(selection)
    selection["umbral"] = float(v_thresholds[int(np.argmax(vf1))])
    selection["average_precision_validacion"] = float(average_precision_score(vy, vp))
    selection["f1_validacion"] = float(np.max(vf1))
    # Comparación de calibración fijada en validación, sin elegir con test.
    vb = base_pipeline.predict_proba(X_train.iloc[valid_idx])[:, 1]
    calibration_report = []
    for name, values in [("sin_calibrar", vb), ("sigmoide_cv3", vp)]:
        calibration_report.append({"version": name, "brier_validacion": brier_score_loss(vy, values),
            "log_loss_validacion": log_loss(vy, values), "promedio_predicho_validacion": float(values.mean()),
            "abandono_observado_validacion": float(vy.mean())})
    pd.DataFrame(calibration_report).to_csv(PROCESSED_TRAIN_DIR / "comparacion_calibracion.csv", index=False)
    # El método y el umbral ya están fijados antes de evaluar la nueva versión en test.
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = probabilities >= selection["umbral"]
    train_partition = pd.DataFrame(
        {
            "id_cliente": train["id_cliente"].to_numpy(),
            "particion": np.where(
                np.isin(np.arange(len(train)), prepared["valid_idx"]),
                "validacion",
                "train",
            ),
        }
    )
    test_partition = pd.DataFrame(
        {"id_cliente": test["id_cliente"].to_numpy(), "particion": "test"}
    )
    pd.concat([train_partition, test_partition], ignore_index=True).to_csv(
        PARTITIONS_PATH, index=False
    )
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    metrics = {
        "modelo_seleccionado": selection["algoritmo"],
        "parametros": selection["parametros"],
        "umbral_elegido_en_validacion": selection["umbral"],
        "average_precision_validacion": selection["average_precision_validacion"],
        "f1_validacion": selection["f1_validacion"],
        "train_ajuste_y_calibracion_cv": int(len(fit_idx)),
        "validacion_umbral": int(len(valid_idx)),
        "calibracion": "sigmoide CV3 con predicciones out-of-fold sobre fit",
        "brier_test": float(brier_score_loss(y_test, probabilities)),
        "log_loss_test": float(log_loss(y_test, probabilities)),
        "probabilidad_media_test": float(probabilities.mean()),
        "tasa_abandono_test": float(y_test.mean()),
        "limitacion_test": "Reevaluación: test ya fue observado en versiones anteriores; se necesita una cohorte independiente para validación confirmatoria.",
        "test": int(len(test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "matriz_confusion_tn_fp_fn_tp": matrix.tolist(),
        "advertencia": (
            "Evaluación sobre datos sintéticos; no acredita desempeño comercial "
            "ni validez en clientes reales."
        ),
    }
    ADVANCED_TEST_METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    scored = pd.DataFrame(
        {
            "id_cliente": test["id_cliente"].to_numpy(),
            "region": test["region"].to_numpy(),
            "probabilidad_abandono": probabilities,
            "abandono_observado": y_test.to_numpy(),
            "prediccion": predictions.astype(int),
        }
    )
    scored["nivel_riesgo"] = pd.cut(
        probabilities,
        [-0.001, 0.2, 0.5, 1.001],
        labels=["Bajo", "Medio", "Alto"],
    )
    scored.to_csv(FINAL_MODEL_SCORED_TEST_PATH, index=False)
    joblib.dump(
        {
            "pipeline": pipeline,
            "features": FEATURES,
            "threshold": selection["umbral"],
            "algorithm": selection["algoritmo"],
        },
        ADVANCED_MODEL_PATH,
    )
    # Salidas canónicas y alias compatibles con el dashboard anterior.
    from paths import (SCORED_TEST_PATH, REGIONAL_AUDIT_PATH, TEST_METRICS_PATH,
        MODEL_PATH, MODEL_COEFFICIENTS_PATH, TRAIN_PROCESSED_PATH, TEST_PROCESSED_PATH)
    scored.to_csv(SCORED_TEST_PATH, index=False)
    TEST_METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump({"pipeline": pipeline, "features": FEATURES, "threshold": selection["umbral"],
        "algorithm": selection["algoritmo"], "calibration": "sigmoid_cv3"}, MODEL_PATH)
    regional = []
    for region, group in scored.groupby("region"):
        truth, prediction = group.abandono_observado, group.prediccion
        regional.append({"region": region, "clientes_test": len(group),
            "tasa_abandono": float(truth.mean()), "probabilidad_promedio": float(group.probabilidad_abandono.mean()),
            "recall": recall_score(truth, prediction, zero_division=0),
            "precision": precision_score(truth, prediction, zero_division=0),
            "roc_auc": roc_auc_score(truth, group.probabilidad_abandono),
            "brier": brier_score_loss(truth, group.probabilidad_abandono)})
    pd.DataFrame(regional).to_csv(REGIONAL_AUDIT_PATH, index=False)
    pre = base_pipeline.named_steps["preprocess"]
    names = pre.get_feature_names_out()
    for frame, target_path in [(train, TRAIN_PROCESSED_PATH), (test, TEST_PROCESSED_PATH)]:
        values = pre.transform(frame[FEATURES])
        if hasattr(values, "toarray"): values = values.toarray()
        processed = pd.DataFrame(values, columns=names)
        processed.insert(0, "id_cliente", frame.id_cliente.to_numpy())
        processed[TARGET] = frame[TARGET].to_numpy()
        temp_matrix = target_path.with_suffix(".tmp")
        processed.to_csv(temp_matrix, index=False)
        temp_matrix.replace(target_path)
    estimator = base_pipeline.named_steps["model"]
    if hasattr(estimator, "coef_"):
        pd.DataFrame({"variable_transformada": names, "coeficiente": estimator.coef_[0]}).to_csv(MODEL_COEFFICIENTS_PATH, index=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, values in [("Sin calibrar", vb), ("Calibrado", vp)]:
        observed, predicted = calibration_curve(vy, values, n_bins=10, strategy="quantile")
        ax.plot(predicted, observed, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="Probabilidad media predicha", ylabel="Frecuencia observada", title="Calibración en validación sintética")
    ax.legend(); _save_figure(fig, "calibracion_validacion.png")
    _save_test_plots(y_test, probabilities, matrix, base_pipeline, X_test)
    # La figura SHAP explica el score del estimador base, no la probabilidad calibrada.
    import shutil
    shutil.copyfile(GRAPHS_DIR / "matriz_confusion_modelo_test.png", GRAPHS_DIR / "matriz_confusion.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    regional_frame = pd.DataFrame(regional)
    regional_frame.set_index("region")[["tasa_abandono", "probabilidad_promedio"]].mul(100).plot.bar(ax=ax)
    ax.set(ylabel="Porcentaje", title="Modelo final calibrado por región")
    ax.tick_params(axis="x", rotation=10); _save_figure(fig, "resumen_regional.png")
    # Escritura final atómica y alias de bytes idénticos.
    temporary = FINAL_MODEL_SCORED_TEST_PATH.with_suffix(".tmp")
    scored.to_csv(temporary, index=False)
    temporary.replace(FINAL_MODEL_SCORED_TEST_PATH)
    shutil.copyfile(FINAL_MODEL_SCORED_TEST_PATH, SCORED_TEST_PATH)
    return metrics, scored


def _save_test_plots(y_test, probabilities, matrix, pipeline, X_test):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    false_positive, true_positive, _ = roc_curve(y_test, probabilities)
    axes[0].plot(false_positive, true_positive, color="#0072B2", label="Modelo")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Azar")
    axes[0].set(
        title="Curva ROC — test reservado",
        xlabel="Tasa de falsos positivos",
        ylabel="Tasa de verdaderos positivos",
    )
    axes[0].legend(frameon=False)
    pr_precision, pr_recall, _ = precision_recall_curve(y_test, probabilities)
    axes[1].plot(pr_recall, pr_precision, color="#D55E00", label="Modelo")
    axes[1].axhline(
        float(np.mean(y_test)), linestyle="--", color="#777777",
        label="Prevalencia de abandono",
    )
    axes[1].set(
        title="Precision–Recall — test reservado",
        xlabel="Recall",
        ylabel="Precision",
    )
    axes[1].legend(frameon=False)
    figure.tight_layout()
    _save_figure(figure, "evaluacion_curvas_test.png")

    figure, axis = plt.subplots(figsize=(6.8, 5.2))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, label="Clientes")
    row_percentages = matrix / matrix.sum(axis=1, keepdims=True) * 100
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}\n{row_percentages[row, column]:.1f}%",
                ha="center",
                va="center",
            )
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Sin abandono", "Abandono"],
        yticklabels=["Sin abandono", "Abandono"],
        xlabel="Predicción",
        ylabel="Resultado real",
        title="Matriz de confusión — test reservado",
    )
    _save_figure(figure, "matriz_confusion_modelo_test.png")
    _save_gain_lift_plot(y_test, probabilities)
    _save_shap_plot(pipeline, X_test)


def _save_gain_lift_plot(y_test, probabilities):
    ranked = pd.DataFrame({"y": np.asarray(y_test), "score": probabilities})
    ranked = ranked.sort_values("score", ascending=False).reset_index(drop=True)
    ranked["decil"] = np.minimum(
        np.ceil((np.arange(len(ranked)) + 1) / len(ranked) * 10).astype(int), 10
    )
    gains = ranked.groupby("decil", as_index=False).agg(
        clientes=("y", "size"), abandonos=("y", "sum")
    )
    gains["ganancia_acumulada"] = gains["abandonos"].cumsum() / max(
        int(ranked["y"].sum()), 1
    )
    gains["lift_acumulado"] = gains["ganancia_acumulada"] / (
        gains["clientes"].cumsum() / len(ranked)
    )
    gains.to_csv(GAIN_LIFT_TEST_PATH, index=False)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].plot(
        gains["decil"], gains["ganancia_acumulada"] * 100,
        marker="o", color="#0072B2", label="Modelo",
    )
    axes[0].plot(
        gains["decil"], gains["decil"] * 10,
        linestyle="--", color="#777777", label="Selección aleatoria",
    )
    axes[0].set(
        title="Ganancia acumulada — test",
        xlabel="Decil acumulado de mayor riesgo",
        ylabel="Abandonos capturados (%)",
    )
    axes[1].plot(
        gains["decil"], gains["lift_acumulado"],
        marker="o", color="#009E73", label="Lift acumulado",
    )
    axes[1].axhline(1, linestyle="--", color="#777777", label="Referencia aleatoria")
    axes[1].set(
        title="Lift acumulado — test",
        xlabel="Decil acumulado de mayor riesgo",
        ylabel="Lift frente a selección aleatoria",
    )
    for axis in axes:
        axis.legend(frameon=False)
        axis.set_xticks(range(1, 11))
    figure.tight_layout()
    _save_figure(figure, "ganancia_lift_test.png")


def _save_shap_plot(pipeline, X_test, sample_size=200):
    preprocessor = pipeline.named_steps["preprocess"]
    estimator = pipeline.named_steps["model"]
    sample = X_test.sample(
        n=min(sample_size, len(X_test)), random_state=RANDOM_STATE
    )
    transformed = preprocessor.transform(sample)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    background_size = min(100, len(transformed))
    background = transformed[:background_size]
    explanation_data = transformed[background_size:] if len(transformed) > background_size else transformed
    if len(explanation_data) == 0:
        explanation_data = transformed
    explainer = shap.Explainer(estimator, background)
    explanation = explainer(explanation_data)
    values = explanation.values
    if values.ndim == 3:
        values = values[:, :, 1]
    importance = pd.DataFrame({"variable_transformada": preprocessor.get_feature_names_out(),
        "importancia_shap": np.abs(values).mean(axis=0)})
    def original_feature(name):
        suffix = name.split("__", 1)[-1]
        for feature in sorted(FEATURES, key=len, reverse=True):
            if suffix == feature or suffix.startswith(feature + "_") or suffix == "missingindicator_" + feature:
                return feature
        return suffix
    importance["variable"] = importance.variable_transformada.map(original_feature)
    importance = importance.groupby("variable", as_index=False).importancia_shap.sum().sort_values("importancia_shap", ascending=False)
    importance["interpretacion"] = "Contribución absoluta media al score base; no a probabilidad calibrada ni causalidad"
    importance.to_csv(PROCESSED_TEST_DIR / "importancia_shap_final.csv", index=False)
    importance.head(5).to_csv(PROCESSED_TEST_DIR / "top5_shap_final.csv", index=False)
    figure = plt.figure(figsize=(9, 6))
    shap.summary_plot(
        values,
        explanation_data,
        feature_names=preprocessor.get_feature_names_out(),
        show=False,
        plot_size=None,
    )
    plt.title("SHAP del score base antes de calibración — muestra sintética")
    _save_figure(figure, "explicabilidad_shap_test.png")


def _save_figure(figure, filename):
    figure.savefig(GRAPHS_DIR / filename, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
