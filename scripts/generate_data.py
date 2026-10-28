"""Simulación académica: no contiene clientes ni resultados reales."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
SEED = 42
CUTOFF = pd.Timestamp("2026-04-01")
END = pd.Timestamp("2026-10-01")
def generate(n=185000):
    rng = np.random.default_rng(SEED)
    region = rng.choice(["Metropolitana", "Valparaíso", "Biobío"], n, p=[.65,.20,.15])
    comuna = np.array([rng.choice({"Metropolitana":["Santiago","Puente Alto","Maipú"],"Valparaíso":["Valparaíso","Viña del Mar","Quilpué"],"Biobío":["Concepción","Talcahuano","Los Ángeles"]}[r]) for r in region])
    age = rng.integers(18,81,n).astype(float)
    tenure = rng.integers(13,31,n)
    purchases = 2 + rng.poisson(6,n)
    recency = np.minimum(rng.gamma(2,38,n).astype(int)+1,365)
    ticket = np.round(rng.lognormal(np.log(65000),.6,n))
    promotions = rng.beta(2,3,n)
    complaints = rng.poisson(.25,n)
    returns = np.minimum(rng.poisson(.4,n),purchases)
    interactions = rng.poisson(3,n)
    channel = rng.choice(["Tienda física","Comercio electrónico","Mixto"],n,p=[.45,.3,.25])
    score = .009*(recency-75) -.10*(purchases-8) + .5*complaints + .15*returns -.12*interactions + .4*promotions
    lo,hi=-15.,15.
    for _ in range(80):
        mid=(lo+hi)/2
        if np.mean(1/(1+np.exp(-(score+mid))))>.24: hi=mid
        else: lo=mid
    probability=1/(1+np.exp(-(score+(lo+hi)/2)))
    churn=rng.binomial(1,probability)
    future_count=np.where(churn==1,0,1+rng.poisson(3,n))
    future_recency=rng.integers(1,(END-CUTOFF).days+1,n)
    last_pre=CUTOFF-pd.to_timedelta(recency,unit="D")
    last_end=pd.Series(END-pd.to_timedelta(future_recency,unit="D"))
    last_end.loc[churn==1]=pd.Series(last_pre).loc[churn==1]
    df=pd.DataFrame({"id_cliente":np.arange(1,n+1),"edad":age,"comuna":comuna,"region":region,
        "fecha_registro":CUTOFF-pd.to_timedelta(tenure*30,unit="D"),"canal_compra":channel,
        "antiguedad_meses":tenure,"cantidad_compras_12m":purchases,"frecuencia_compra_mensual":purchases/12,
        "monto_promedio_compra":ticket,"monto_total_12m":ticket*purchases,
        "categoria_principal":rng.choice(["Hogar","Tecnología","Electrodomésticos"],n),
        "proporcion_compras_promocion":promotions.round(4),"devoluciones_12m":returns,
        "reclamos_12m":complaints,"interacciones_campanas_12m":interactions,
        "fecha_ultima_compra_pre_corte":last_pre,"recencia_dias":recency,
        "compras_periodo_objetivo":future_count,"fecha_ultima_compra_al_cierre":last_end,
        "churn_6m":(future_count==0).astype(int)})
    missing=rng.choice(n,round(.08*n),replace=False)
    df.loc[missing,"edad"]=np.nan
    # Región se conserva para auditar representación; 8% tiene demografía incompleta.
    df.loc[missing[::2],"comuna"]=None
    out=ROOT/"data/generated";out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/"retailnova_case1_sintetico.csv",index=False)
    meta={"tipo":"datos sintéticos; no son evidencia comercial real","seed":SEED,"clientes":n,
        "cohorte":"clientes con al menos dos compras en los 12 meses previos al corte",
        "corte_variables":str(CUTOFF.date()),"fin_objetivo_exclusivo":str(END.date()),
        "historia_disponible_caso_meses":36,"ventana_features_meses":12,
        "churn_definicion":"cero compras entre 2026-04-01 inclusive y 2026-10-01 exclusive",
        "churn_observado":float(df.churn_6m.mean()),"demografia_incompleta":float(df[["edad","comuna","region"]].isna().any(axis=1).mean()),
        "supuestos":"185000 clientes recurrentes simulados para demostración; el caso no afirma que los 185000 registrados sean todos recurrentes. Probabilidad calibrada a 24% esperado, no etiqueta forzada. No se simulan 36 meses de transacciones individuales.",
        "limitaciones":"Las asociaciones fueron diseñadas; las métricas solo evalúan la simulación. No demuestran causalidad ni eficacia de campañas."}
    (out/"retailnova_case1_sintetico_metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    print(json.dumps(meta,ensure_ascii=False,indent=2))
    return df
if __name__=="__main__": generate()
