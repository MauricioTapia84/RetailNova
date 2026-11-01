"""Puntuación de nuevos clientes con el modelo final calibrado.\nUso: python src/predict.py --input ruta.csv --output ruta_resultado.csv\nNo requiere ni utiliza etiquetas futuras; conservar el mismo corte de variables.\n"""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from paths import ADVANCED_MODEL_PATH

def score(input_path, output_path):
    # Cargar únicamente artefactos propios de origen conocido.
    artifact=joblib.load(ADVANCED_MODEL_PATH)
    frame=pd.read_csv(input_path)
    if 'id_cliente' not in frame or not frame.id_cliente.is_unique:
        raise ValueError('Se requiere id_cliente único.')
    if 'tasa_devoluciones' not in frame and {'devoluciones_12m','cantidad_compras_12m'}<=set(frame):
        frame['tasa_devoluciones']=frame.devoluciones_12m/frame.cantidad_compras_12m.clip(lower=1)
    missing=set(artifact['features'])-set(frame)
    if missing: raise ValueError(f'Faltan variables anteriores al corte: {sorted(missing)}')
    probabilities=artifact['pipeline'].predict_proba(frame[artifact['features']])[:,1]
    result=frame[['id_cliente']].copy()
    if 'region' in frame:result['region']=frame.region
    result['probabilidad_abandono']=probabilities
    result['prediccion']=(probabilities>=artifact['threshold']).astype(int)
    result['nivel_riesgo']=pd.cut(probabilities,[-.001,.2,.5,1.001],labels=['Bajo','Medio','Alto'])
    target=Path(output_path);target.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(target,index=False)
    print(f'Clientes puntuados: {len(result)}. Archivo: {target}')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();score(args.input,args.output)
