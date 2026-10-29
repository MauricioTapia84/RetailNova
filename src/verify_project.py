"""Controles de consistencia temporal, datos y artefactos del proyecto."""
import json
import numpy as np
import pandas as pd
from analyze import FEATURES
from paths import (
    ADVANCED_MODEL_PATH,
    ADVANCED_TEST_METRICS_PATH,
    DATASET_PATH,
    DATA_DIR,
    DASHBOARD_TRAIN_PATH,
    FINAL_MODEL_SCORED_TEST_PATH,
    GRAPHS_DIR,
    GAIN_LIFT_TEST_PATH,
    HISTORICAL_MONTHLY_PATH,
    METADATA_PATH,
    MODEL_DIR,
    MODEL_PATH,
    NOTEBOOKS_DIR,
    OUTPUTS_DIR,
    PARTITIONS_PATH,
    PROCESSED_DIR,
    PROCESSED_TEST_DIR,
    PROCESSED_TRAIN_DIR,
    RAW_DIR,
    RAW_TEST_DIR,
    RAW_TRAIN_DIR,
    CLASS_BALANCE_RESULTS_PATH,
    CLUSTERING_RESULTS_PATH,
    CLUSTER_MODEL_PATH,
    OPTUNA_RESULTS_PATH,
    SUPERVISED_COMPARISON_PATH,
    TEST_PROCESSED_PATH,
    TEST_RAW_PATH,
    TRAIN_CLEAN_PATH,
    TRAIN_PROCESSED_PATH,
    TRAIN_RAW_PATH,
    TRAIN_SOURCE_PATH,
)

df=pd.read_csv(DATASET_PATH)
assert len(df)==185000 and df.id_cliente.is_unique
assert set(df.region)=={'Metropolitana','Valparaíso','Biobío'}
assert abs(df.churn_6m.mean()-.24)<.01
assert df[['edad','comuna','region']].isna().any(axis=1).sum()==14800
assert (df.cantidad_compras_12m>=2).all()
assert (pd.to_datetime(df.fecha_ultima_compra_pre_corte)<pd.Timestamp('2026-04-01')).all()
assert (pd.to_datetime(df.fecha_registro)<pd.to_datetime(df.fecha_ultima_compra_pre_corte)).all()
assert ((df.compras_periodo_objetivo==0).astype(int)==df.churn_6m).all()
assert ((pd.to_datetime(df.fecha_ultima_compra_al_cierre)<pd.Timestamp('2026-04-01')).astype(int)==df.churn_6m).all()
assert not set(FEATURES)&{'id_cliente','churn_6m','compras_periodo_objetivo','fecha_ultima_compra_al_cierre','prioridad_atencion'}
metadata=json.loads(METADATA_PATH.read_text(encoding='utf-8'))
history_metadata=metadata['historial_mensual']
assert history_metadata['meses_historicos']==36
assert history_metadata['inicio_inclusivo']=='2023-04-01'
assert history_metadata['fin_exclusivo']=='2026-04-01'
assert history_metadata['meses_disponibles']==36
historical=pd.read_csv(
    HISTORICAL_MONTHLY_PATH,
    usecols=['id_cliente','mes','compras','monto_total','compras_promocion','devoluciones','reclamos','interacciones_campana'],
    parse_dates=['mes'],
)
assert len(historical)==history_metadata['filas_con_actividad']
assert historical['mes'].dt.to_period('M').nunique()==36
assert str(historical['mes'].min().date())=='2023-04-01'
assert str(historical['mes'].max().date())=='2026-03-01'
assert not historical.duplicated(['id_cliente','mes']).any()
recent_history=historical.loc[historical['mes']>='2025-04-01']
history_columns=[
    'compras','monto_total','compras_promocion','devoluciones',
    'reclamos','interacciones_campana',
]
recent_summary=recent_history.groupby('id_cliente')[history_columns].sum().reindex(df.id_cliente,fill_value=0)
history_summary=historical.groupby('id_cliente')[history_columns].sum().reindex(df.id_cliente,fill_value=0)
assert (recent_summary['compras'].to_numpy()==df.cantidad_compras_12m.to_numpy()).all()
assert (history_summary['compras'].to_numpy()==df.cantidad_compras_36m.to_numpy()).all()
assert np.allclose(recent_summary['monto_total'],df.monto_total_12m)
assert np.allclose(history_summary['monto_total'],df.monto_total_36m)
assert (recent_summary['devoluciones'].to_numpy()==df.devoluciones_12m.to_numpy()).all()
assert (recent_summary['reclamos'].to_numpy()==df.reclamos_12m.to_numpy()).all()
assert (recent_summary['interacciones_campana'].to_numpy()==df.interacciones_campanas_12m.to_numpy()).all()
assert np.allclose(
    recent_summary['compras_promocion']/recent_summary['compras'],
    df.proporcion_compras_promocion,
)
split=pd.read_csv(PARTITIONS_PATH);assert split.id_cliente.is_unique and len(split)==len(df)
assert split.particion.value_counts().to_dict()=={'train':118400,'validacion':29600,'test':37000}
raw_train=pd.read_csv(TRAIN_RAW_PATH)
raw_test=pd.read_csv(TEST_RAW_PATH)
train_clean=pd.read_csv(TRAIN_CLEAN_PATH)
train_source=pd.read_csv(TRAIN_SOURCE_PATH)
processed_train=pd.read_csv(TRAIN_PROCESSED_PATH)
processed_test=pd.read_csv(TEST_PROCESSED_PATH)
dashboard=pd.read_csv(DASHBOARD_TRAIN_PATH)
assert len(raw_train)==148000 and len(raw_test)==37000
assert RAW_TRAIN_DIR.is_dir() and RAW_TEST_DIR.is_dir()
assert PROCESSED_TRAIN_DIR.is_dir() and PROCESSED_TEST_DIR.is_dir()
assert not (RAW_DIR/'train.csv').exists() and not (RAW_DIR/'test.csv').exists()
assert not (PROCESSED_DIR/'train.csv').exists() and not (PROCESSED_DIR/'test.csv').exists()
assert {path.name for path in RAW_TRAIN_DIR.glob('*.csv')}=={'train.csv'}
assert {path.name for path in RAW_TEST_DIR.glob('*.csv')}=={'test.csv'}
assert {path.name for path in PROCESSED_DIR.glob('*.csv')}=={'particiones.csv'}
assert not (RAW_DIR/'historial_mensual_12m.csv.gz').exists()
assert len(train_clean)==148000 and len(train_source)==148000 and len(dashboard)==148000
assert len(processed_train)==148000 and len(processed_test)==37000
assert train_clean.id_cliente.is_unique and dashboard.customer_id.is_unique
assert train_source.id_cliente.is_unique and not train_clean.isna().any().any()
assert abs(train_source.edad.isna().mean()-.08)<.002
assert raw_train.id_cliente.is_unique and raw_test.id_cliente.is_unique
assert not set(raw_train.id_cliente)&set(raw_test.id_cliente)
assert set(raw_train.id_cliente)==set(train_clean.id_cliente)==set(train_source.id_cliente)
assert set(processed_train.id_cliente)==set(raw_train.id_cliente)
assert set(processed_test.id_cliente)==set(raw_test.id_cliente)
assert {'tasa_devoluciones','churn_6m'}<=set(train_clean.columns)
assert 'observed_churn_6m' in dashboard.columns
assert {path.name for path in OUTPUTS_DIR.iterdir()}=={'graphs','model'}
assert not any(path.is_file() and path.suffix.lower()=='.csv' for path in OUTPUTS_DIR.rglob('*'))
assert all(DATA_DIR in path.parents for path in DATA_DIR.parent.rglob('*.csv'))
assert (GRAPHS_DIR/'distribucion_objetivo.png').is_file()
assert (GRAPHS_DIR/'matriz_confusion.png').is_file()
assert MODEL_PATH.is_file() and MODEL_DIR.is_dir() and RAW_DIR.is_dir()
assert ADVANCED_MODEL_PATH.is_file() and CLUSTER_MODEL_PATH.is_file()
assert ADVANCED_TEST_METRICS_PATH.is_file()
assert FINAL_MODEL_SCORED_TEST_PATH.is_file()
assert GAIN_LIFT_TEST_PATH.is_file()
assert CLASS_BALANCE_RESULTS_PATH.is_file()
assert CLUSTERING_RESULTS_PATH.is_file() and OPTUNA_RESULTS_PATH.is_file()
assert SUPERVISED_COMPARISON_PATH.is_file()
supervised=pd.read_csv(SUPERVISED_COMPARISON_PATH)
assert set(supervised.modelo)=={
    'LogisticRegression','RandomForest','ExtraTrees','BalancedRandomForest'
}
assert len(pd.read_csv(FINAL_MODEL_SCORED_TEST_PATH))==37000
assert (GRAPHS_DIR/'clustering_codo_pca.png').is_file()
assert (GRAPHS_DIR/'optuna_comparacion.png').is_file()
assert (GRAPHS_DIR/'evaluacion_curvas_test.png').is_file()
assert (GRAPHS_DIR/'ganancia_lift_test.png').is_file()
assert (GRAPHS_DIR/'explicabilidad_shap_test.png').is_file()
for path in sorted(NOTEBOOKS_DIR.glob('*.ipynb')):
    nb=json.loads(path.read_text(encoding='utf-8'))
    for cell in nb['cells']:
        if cell['cell_type']=='code':
            assert cell['execution_count'] is not None
            assert all(o['output_type']!='error' for o in cell['outputs'])
print('PASS: cohorte, 36 meses históricos derivados, ventanas temporales, calidad, splits, modelo y notebooks.')
