"""Motor ÚNICO para web e informe. Datos mensuales positivos, índices MS."""
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing

MODELOS=['PMS3','PMS6','PMS12','PMP3','SES','Holt','HW_aditivo','HW_multiplicativo']
EXP=MODELOS[4:]

def preparar(serie):
    s=serie.copy().astype(float).sort_index()
    s.index=pd.to_datetime(s.index)
    if s.index.duplicated().any():raise ValueError('Fechas duplicadas')
    s=s.asfreq('MS')
    if s.isna().any() or (s<=0).any() or not np.isfinite(s).all():raise ValueError('Precios ausentes o no positivos')
    return s

def estimar(s,modelo):
    s=preparar(s)
    if modelo.startswith('PMS'):
        k=int(modelo[3:]);return s.rolling(k).mean().shift(1),None
    if modelo=='PMP3':
        pesos=np.arange(1,4,dtype=float)
        return s.rolling(3).apply(lambda x:np.dot(x,pesos)/pesos.sum(),raw=True).shift(1),None
    if modelo=='SES':obj=SimpleExpSmoothing(s,initialization_method='estimated')
    elif modelo=='Holt':obj=Holt(s,initialization_method='estimated')
    elif modelo in ('HW_aditivo','HW_multiplicativo'):
        obj=ExponentialSmoothing(s,trend='add',seasonal='add' if modelo=='HW_aditivo' else 'mul',seasonal_periods=12,initialization_method='estimated')
    else:raise ValueError(modelo)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit=obj.fit(optimized=True)
    return pd.Series(np.asarray(fit.fittedvalues),index=s.index),fit

def metricas(real,estimado):
    a=np.asarray(real,dtype=float);b=np.asarray(estimado,dtype=float)
    if len(a)!=len(b) or not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Observaciones no válidas')
    e=a-b
    return {'MAE':float(np.mean(np.abs(e))),'RMSE':float(np.sqrt(np.mean(e**2))),'MAPE (%)':float(np.mean(np.abs(e/a))*100)}

def entrenamiento(s,comienzo=24):
    s=preparar(s);filas=[];ajustes={};fallos={}
    for nombre in MODELOS:
        try:
            ajustado,_=estimar(s,nombre)
            filas.append({'Modelo':nombre,**metricas(s.iloc[comienzo:],ajustado.iloc[comienzo:]),'Meses evaluados':len(s)-comienzo})
            ajustes[nombre]=ajustado
        except Exception as e:fallos[nombre]=str(e)
    return pd.DataFrame(filas).sort_values('RMSE').reset_index(drop=True),ajustes,fallos

def validacion(s,h=12):
    s=preparar(s);train=s.iloc[:-h];test=s.iloc[-h:];filas=[];predicciones={};fallos={}
    for nombre in MODELOS:
        try:
            if nombre.startswith('PMS') or nombre=='PMP3':
                # Pronóstico fijo desde origen: sin observar meses de prueba.
                if nombre.startswith('PMS'):k=int(nombre[3:]);valor=float(train.iloc[-k:].mean())
                else:valor=float(np.dot(train.iloc[-3:],np.arange(1,4))/6)
                futuro=np.repeat(valor,h)
            else:
                _,fit=estimar(train,nombre);futuro=np.asarray(fit.forecast(h),dtype=float)
            predicciones[nombre]=pd.Series(futuro,index=test.index)
            filas.append({'Modelo':nombre,**metricas(test,futuro),'Meses evaluados':h})
        except Exception as e:fallos[nombre]=str(e)
    return pd.DataFrame(filas).sort_values('RMSE').reset_index(drop=True),predicciones,fallos

def escenarios(s,modelo,h=6,n_sim=1000,seed=123):
    """Central y bandas EXPLORATORIAS mediante remuestreo de residuos de ajuste.
    No son intervalos de predicción calibrados; no se interpretan como cobertura 90%.
    """
    s=preparar(s)
    if modelo not in EXP:raise ValueError('Escenarios disponibles para suavizamiento exponencial')
    ajustado,fit=estimar(s,modelo)
    central=np.asarray(fit.forecast(h),dtype=float)
    resid=(s-ajustado).iloc[24:].to_numpy()
    resid=resid[np.isfinite(resid)]
    rng=np.random.default_rng(seed)
    # Escala sqrt(h) es un supuesto didáctico; usar backtesting para calibrar bandas.
    shocks=rng.choice(resid,size=(n_sim,h),replace=True)*np.sqrt(np.arange(1,h+1))[None,:]
    simulados=central[None,:]+shocks
    idx=pd.date_range(s.index[-1]+pd.offsets.MonthBegin(1),periods=h,freq='MS')
    return pd.DataFrame({'Pesimista (exploratorio)':np.maximum(0,np.quantile(simulados,.05,axis=0)),'Central':central,'Favorable (exploratorio)':np.quantile(simulados,.95,axis=0)},index=idx)
