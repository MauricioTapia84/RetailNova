"""Generate a reproducible, synthetic 36-month RetailNova customer history."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from paths import (
    DATASET_PATH,
    HISTORICAL_MONTHLY_PATH,
    METADATA_PATH,
    RAW_DIR,
    RAW_TEST_DIR,
    RAW_TRAIN_DIR,
)

SEED = 42
CUTOFF = pd.Timestamp("2026-04-01")
FOLLOWUP_END = pd.Timestamp("2026-10-01")
HISTORY_MONTHS = 36
FEATURE_MONTHS = 12


def _month_indices(dates: pd.DatetimeIndex, history_start: pd.Timestamp) -> np.ndarray:
    date_months = dates.to_numpy(dtype="datetime64[M]").astype(np.int64)
    start_month = np.datetime64(history_start, "M").astype(np.int64)
    return date_months - start_month


def generate(n: int = 185_000, output_dir: Path = RAW_DIR) -> pd.DataFrame:
    if n < 1:
        raise ValueError("n debe ser mayor que cero.")

    rng = np.random.default_rng(SEED)
    history_start = CUTOFF - pd.DateOffset(months=HISTORY_MONTHS)
    month_starts = pd.date_range(history_start, periods=HISTORY_MONTHS, freq="MS")
    registration_months = rng.integers(12, 73, size=n)
    registration_dates = CUTOFF - pd.to_timedelta(
        registration_months * (365.25 / 12), unit="D"
    )

    region = rng.choice(
        ["Metropolitana", "Valparaíso", "Biobío"], n, p=[0.65, 0.20, 0.15]
    )
    communes = {
        "Metropolitana": ["Santiago", "Puente Alto", "Maipú"],
        "Valparaíso": ["Valparaíso", "Viña del Mar", "Quilpué"],
        "Biobío": ["Concepción", "Talcahuano", "Los Ángeles"],
    }
    comuna = np.array([rng.choice(communes[value]) for value in region])
    age = rng.integers(18, 81, n).astype(float)
    channel = rng.choice(
        ["Tienda física", "Comercio electrónico", "Mixto"],
        n,
        p=[0.45, 0.30, 0.25],
    )
    category = rng.choice(
        ["Hogar", "Tecnología", "Electrodomésticos"],
        n,
        p=[0.36, 0.34, 0.30],
    )
    average_ticket = rng.lognormal(np.log(65_000), 0.6, n)
    promotion_propensity = rng.beta(2, 3, n)

    monthly_purchases = np.zeros((n, HISTORY_MONTHS), dtype=np.int16)
    for month_index, month_start in enumerate(month_starts[: HISTORY_MONTHS - FEATURE_MONTHS]):
        month_end = month_start + pd.DateOffset(months=1)
        active_start = np.maximum(
            registration_dates.values.astype("datetime64[ns]"),
            month_start.to_datetime64(),
        )
        active_days = np.maximum(
            0,
            (month_end.to_datetime64() - active_start)
            .astype("timedelta64[D]")
            .astype(int),
        )
        days_in_month = (month_end - month_start).days
        expected_count = 0.68 * active_days / days_in_month
        monthly_purchases[:, month_index] = rng.poisson(expected_count)

    purchases_12m = 2 + rng.poisson(6, n)
    recency_days = np.minimum(rng.gamma(2, 38, n).astype(int) + 1, 365)
    last_purchase_dates = CUTOFF - pd.to_timedelta(recency_days, unit="D")

    repeat_customer_ids = np.repeat(np.arange(n), purchases_12m - 1)
    earlier_day_offsets = recency_days[repeat_customer_ids] + np.floor(
        rng.random(len(repeat_customer_ids))
        * (366 - recency_days[repeat_customer_ids])
    ).astype(int)
    recent_customer_ids = np.concatenate((np.arange(n), repeat_customer_ids))
    recent_day_offsets = np.concatenate((recency_days, earlier_day_offsets))
    recent_purchase_dates = pd.DatetimeIndex(
        CUTOFF - pd.to_timedelta(recent_day_offsets, unit="D")
    )
    recent_month_indices = _month_indices(recent_purchase_dates, history_start)
    recent_flat_indices = recent_customer_ids * HISTORY_MONTHS + recent_month_indices
    monthly_purchases[:, HISTORY_MONTHS - FEATURE_MONTHS :] = np.bincount(
        recent_flat_indices,
        minlength=n * HISTORY_MONTHS,
    ).reshape(n, HISTORY_MONTHS)[:, HISTORY_MONTHS - FEATURE_MONTHS :]

    monthly_returns = rng.binomial(monthly_purchases, 0.05).astype(np.int16)
    monthly_promotions = rng.binomial(
        monthly_purchases, promotion_propensity[:, None]
    ).astype(np.int16)
    monthly_complaints = rng.poisson(0.25 / 12, (n, HISTORY_MONTHS)).astype(
        np.int16
    )
    monthly_interactions = rng.poisson(3 / 12, (n, HISTORY_MONTHS)).astype(
        np.int16
    )
    monthly_ticket_variation = rng.lognormal(
        0, 0.12, (n, HISTORY_MONTHS)
    ).astype(np.float32)
    monthly_spend = (
        monthly_purchases.astype(np.float32)
        * average_ticket[:, None].astype(np.float32)
        * monthly_ticket_variation
    )

    purchases_12m = monthly_purchases[:, -FEATURE_MONTHS:].sum(axis=1)
    spend_12m = monthly_spend[:, -FEATURE_MONTHS:].sum(axis=1)
    returns_12m = monthly_returns[:, -FEATURE_MONTHS:].sum(axis=1)
    promotions_12m = monthly_promotions[:, -FEATURE_MONTHS:].sum(axis=1)
    complaints_12m = monthly_complaints[:, -FEATURE_MONTHS:].sum(axis=1)
    interactions_12m = monthly_interactions[:, -FEATURE_MONTHS:].sum(axis=1)
    promotion_share_12m = np.divide(
        promotions_12m,
        purchases_12m,
        out=np.zeros(n, dtype=float),
        where=purchases_12m > 0,
    )

    score = (
        0.009 * (recency_days - 75)
        - 0.10 * (purchases_12m - 8)
        + 0.5 * complaints_12m
        + 0.15 * returns_12m
        - 0.12 * interactions_12m
        + 0.4 * promotion_share_12m
    )
    lower, upper = -15.0, 15.0
    for _ in range(80):
        offset = (lower + upper) / 2
        if np.mean(1 / (1 + np.exp(-(score + offset)))) > 0.24:
            upper = offset
        else:
            lower = offset
    churn_probability = 1 / (1 + np.exp(-(score + (lower + upper) / 2)))
    churn = rng.binomial(1, churn_probability)
    future_purchases = np.where(churn == 1, 0, 1 + rng.poisson(3, n))
    future_recency = rng.integers(1, (FOLLOWUP_END - CUTOFF).days + 1, n)
    last_purchase_at_followup = FOLLOWUP_END - pd.to_timedelta(
        future_recency, unit="D"
    )
    last_purchase_at_followup = pd.Series(last_purchase_at_followup)
    last_purchase_at_followup.loc[churn == 1] = pd.Series(
        last_purchase_dates
    ).loc[churn == 1]

    missing_demographics = rng.choice(n, round(0.08 * n), replace=False)
    customer_df = pd.DataFrame(
        {
            "id_cliente": np.arange(1, n + 1),
            "edad": age,
            "comuna": comuna,
            "region": region,
            "fecha_registro": registration_dates,
            "canal_compra": channel,
            "antiguedad_meses": registration_months,
            "cantidad_compras_12m": purchases_12m,
            "frecuencia_compra_mensual": purchases_12m / FEATURE_MONTHS,
            "monto_promedio_compra": np.divide(
                spend_12m,
                purchases_12m,
                out=np.zeros(n, dtype=float),
                where=purchases_12m > 0,
            ),
            "monto_total_12m": spend_12m,
            "categoria_principal": category,
            "proporcion_compras_promocion": promotion_share_12m,
            "devoluciones_12m": returns_12m,
            "reclamos_12m": complaints_12m,
            "interacciones_campanas_12m": interactions_12m,
            "fecha_ultima_compra_pre_corte": last_purchase_dates,
            "recencia_dias": recency_days,
            "cantidad_compras_36m": monthly_purchases.sum(axis=1),
            "monto_total_36m": monthly_spend.sum(axis=1),
            "compras_periodo_objetivo": future_purchases,
            "fecha_ultima_compra_al_cierre": last_purchase_at_followup,
            "churn_6m": (future_purchases == 0).astype(int),
        }
    )
    customer_df.loc[missing_demographics, "edad"] = np.nan
    customer_df.loc[missing_demographics[::2], "comuna"] = None

    customer_ids, month_indices = np.nonzero(
        (monthly_purchases > 0)
        | (monthly_complaints > 0)
        | (monthly_interactions > 0)
    )
    historical_df = pd.DataFrame(
        {
            "id_cliente": customer_ids + 1,
            "mes": month_starts.to_numpy()[month_indices],
            "region": region[customer_ids],
            "canal_principal": channel[customer_ids],
            "categoria_principal": category[customer_ids],
            "compras": monthly_purchases[customer_ids, month_indices],
            "monto_total": monthly_spend[customer_ids, month_indices],
            "compras_promocion": monthly_promotions[customer_ids, month_indices],
            "devoluciones": monthly_returns[customer_ids, month_indices],
            "reclamos": monthly_complaints[customer_ids, month_indices],
            "interacciones_campana": monthly_interactions[
                customer_ids, month_indices
            ],
        }
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir == RAW_DIR:
        RAW_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
        RAW_TEST_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / DATASET_PATH.name
    historical_path = output_dir / HISTORICAL_MONTHLY_PATH.name
    metadata_path = output_dir / METADATA_PATH.name
    customer_df.to_csv(dataset_path, index=False)
    historical_df.to_csv(historical_path, index=False, compression="gzip")

    metadata = {
        "tipo": "datos sintéticos; no son evidencia comercial real",
        "seed": SEED,
        "clientes": n,
        "cohorte": (
            "clientes recurrentes simulados con al menos dos compras "
            "en los 12 meses previos al corte"
        ),
        "historial_mensual": {
            "archivo": historical_path.name,
            "meses_historicos": HISTORY_MONTHS,
            "inicio_inclusivo": str(history_start.date()),
            "fin_exclusivo": str(CUTOFF.date()),
            "meses_disponibles": int(historical_df["mes"].nunique()),
            "filas_con_actividad": len(historical_df),
            "granularidad": (
                "resumen por cliente y mes; los meses sin fila representan "
                "ausencia de compras, reclamos e interacciones"
            ),
        },
        "corte_variables": str(CUTOFF.date()),
        "fin_objetivo_exclusivo": str(FOLLOWUP_END.date()),
        "ventana_features_meses": FEATURE_MONTHS,
        "churn_definicion": (
            "cero compras entre 2026-04-01 inclusive y 2026-10-01 exclusive"
        ),
        "churn_observado": float(customer_df["churn_6m"].mean()),
        "demografia_incompleta": float(
            customer_df[["edad", "comuna", "region"]]
            .isna()
            .any(axis=1)
            .mean()
        ),
        "supuestos": (
            "Se simulan 185000 clientes recurrentes para demostrar el flujo; "
            "el caso informa clientes registrados, no que todos sean recurrentes. "
            "El modelo usa atributos de los 12 meses anteriores al corte, "
            "derivados de un historial agregado de 36 meses. La etiqueta se "
            "genera para una ventana futura de seis meses."
        ),
        "limitaciones": (
            "Los datos y asociaciones son sintéticos. El historial mensual "
            "agregado no representa transacciones reales. Las métricas no "
            "demuestran causalidad, eficacia de campañas ni desempeño en "
            "clientes reales."
        ),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return customer_df


if __name__ == "__main__":
    generate()
