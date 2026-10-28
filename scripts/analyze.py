"""Entrenamiento y evaluación con lista explícita de variables previas al corte."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix
import joblib
ROOT=Path(__file__).resolve().parents[1]
FEATURES=["edad","canal_compra","antiguedad_meses","cantidad_compras_12m","monto_promedio_compra","categoria_principal","proporcion_compras_promocion","devoluciones_12m","reclamos_12m","interacciones_campanas_12m","recencia_dias"]
def run():
    df=pd.read_csv(ROOT/"data/generated/retailnova_case1_sintetico.csv")
    out=ROOT/"outputs";out.mkdir(exist_ok=True)
    # Región/comuna se reservan para auditoría; ID, fechas posteriores y etiqueta nunca son predictores.
    train,rest=train_test_split(df.index,test_size=.4,stratify=df.churn_6m,random_state=42)
    valid,test=train_test_split(rest,test_size=.5,stratify=df.loc[rest,"churn_6m"],random_state=42)
    X=df[FEATURES];y=df.churn_6m
    nums=X.select_dtypes(include="number").columns.tolist(); cats=[c for c in FEATURES if c not in nums]
    def preprocessing():
        return ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median",add_indicator=True)),("scale",StandardScaler())]),nums),("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("encode",OneHotEncoder(handle_unknown="ignore"))]),cats)])
    models={"Baseline":DummyClassifier(strategy="prior"),"Logistica":LogisticRegression(max_iter=500,random_state=42),"RandomForest":RandomForestClassifier(n_estimators=60,max_depth=10,min_samples_leaf=40,n_jobs=2,random_state=42)}
    fitted={}; rows=[]
    for name,model in models.items():
        pipe=Pipeline([("preprocess",preprocessing()),("model",model)])
        pipe.fit(X.loc[train],y.loc[train]); probs=pipe.predict_proba(X.loc[valid])[:,1]
        rows.append({"modelo":name,"validacion_roc_auc":roc_auc_score(y.loc[valid],probs),"validacion_average_precision":average_precision_score(y.loc[valid],probs)})
        fitted[name]=pipe
    table=pd.DataFrame(rows).sort_values("validacion_average_precision",ascending=False)
    best=table.iloc[0].modelo; pipe=fitted[best]
    vp=pipe.predict_proba(X.loc[valid])[:,1]
    thresholds=np.arange(.1,.81,.01)
    threshold=float(max(thresholds,key=lambda t:f1_score(y.loc[valid],vp>=t)))
    probs=pipe.predict_proba(X.loc[test])[:,1];pred=probs>=threshold; truth=y.loc[test]
    metrics={"modelo_seleccionado":best,"umbral_elegido_en_validacion":threshold,"train":len(train),"validacion":len(valid),"test":len(test),"accuracy":accuracy_score(truth,pred),"precision":precision_score(truth,pred,zero_division=0),"recall":recall_score(truth,pred),"f1":f1_score(truth,pred),"roc_auc":roc_auc_score(truth,probs),"average_precision":average_precision_score(truth,probs),"matriz_confusion_tn_fp_fn_tp":confusion_matrix(truth,pred).tolist(),"advertencia":"Evaluación sobre datos sintéticos, sin validación en clientes reales."}
    table.to_csv(out/"comparacion_validacion.csv",index=False)
    (out/"metricas_test.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2))
    pd.DataFrame({"id_cliente":df.id_cliente,"particion":np.select([df.index.isin(train),df.index.isin(valid)],["train","validacion"],default="test")}).to_csv(out/"particiones.csv",index=False)
    scored=pd.DataFrame({"id_cliente":df.loc[test,"id_cliente"],"region":df.loc[test,"region"],"probabilidad_abandono":probs,"abandono_observado":truth,"prediccion":pred.astype(int)})
    scored["nivel_riesgo"]=pd.cut(probs,[-.001,.2,.5,1.001],labels=["Bajo","Medio","Alto"])
    scored.to_csv(out/"clientes_riesgo_test.csv",index=False)
    audit=[]
    for region,g in scored.groupby("region"):
        audit.append({"region":region,"clientes_test":len(g),"tasa_abandono":g.abandono_observado.mean(),"recall":recall_score(g.abandono_observado,g.prediccion),"precision":precision_score(g.abandono_observado,g.prediccion,zero_division=0),"roc_auc":roc_auc_score(g.abandono_observado,g.probabilidad_abandono)})
    pd.DataFrame(audit).to_csv(out/"auditoria_regional.csv",index=False)
    joblib.dump({"pipeline":pipe,"features":FEATURES,"threshold":threshold,"nivel_riesgo_cortes":[.2,.5]},out/"modelo_retailnova.joblib")
    fig,ax=plt.subplots(1,3,figsize=(15,4))
    df.churn_6m.value_counts().sort_index().plot.bar(ax=ax[0],color=["#247c83","#de7945"]);ax[0].set_xticklabels(["Sin abandono","Abandono"],rotation=0);ax[0].set_title("Cohorte sintética")
    df.region.value_counts().plot.bar(ax=ax[1],color="#247c83");ax[1].tick_params(axis="x",rotation=15);ax[1].set_title("Representación regional")
    df.groupby("region").churn_6m.mean().mul(100).plot.bar(ax=ax[2],color="#de7945");ax[2].tick_params(axis="x",rotation=15);ax[2].set_title("Abandono simulado (%)")
    fig.tight_layout();fig.savefig(out/"resumen_datos.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,4));cm=confusion_matrix(truth,pred);ax.imshow(cm,cmap="Blues")
    for i in range(2):
        for j in range(2):ax.text(j,i,str(cm[i,j]),ha="center",va="center")
    ax.set(xticks=[0,1],yticks=[0,1],xlabel="Predicción",ylabel="Real",title="Test sintético: matriz de confusión");fig.tight_layout();fig.savefig(out/"matriz_confusion.png",dpi=150);plt.close(fig)
    # Coeficientes de asociación, no explicaciones causales.
    if best=="Logistica":
        names=pipe.named_steps["preprocess"].get_feature_names_out();coef=pipe.named_steps["model"].coef_[0]
        pd.DataFrame({"variable_transformada":names,"coeficiente":coef}).sort_values("coeficiente").to_csv(out/"coeficientes_modelo.csv",index=False)
    (out/"resumen_calidad.json").write_text(json.dumps({"filas":len(df),"columnas":len(df.columns),"ids_unicos":bool(df.id_cliente.is_unique),"duplicados":int(df.duplicated().sum()),"nulos_por_columna":df.isna().sum().to_dict()},indent=2))
    print(table.to_string(index=False));print(json.dumps(metrics,ensure_ascii=False,indent=2));return metrics
if __name__=="__main__":run()
