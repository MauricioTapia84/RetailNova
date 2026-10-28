"""Controles de consistencia temporal, datos y artefactos del proyecto."""
from pathlib import Path
import json
import pandas as pd
from analyze import FEATURES
ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/'data/generated/retailnova_case1_sintetico.csv')
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
split=pd.read_csv(ROOT/'outputs/particiones.csv');assert split.id_cliente.is_unique and len(split)==len(df)
assert split.particion.value_counts().to_dict()=={'train':111000,'validacion':37000,'test':37000}
for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
    nb=json.loads(path.read_text())
    for cell in nb['cells']:
        if cell['cell_type']=='code':
            assert cell['execution_count'] is not None
            assert all(o['output_type']!='error' for o in cell['outputs'])
print('PASS: cohorte, calidad, temporalidad, exclusión de fugas conocidas, particiones y notebooks ejecutados.')
