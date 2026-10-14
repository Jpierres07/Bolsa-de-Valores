from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

st.set_page_config(page_title='Observatorio de acciones', page_icon='📈', layout='wide')
st.markdown("""<style>
.stApp {background:#0b1925;color:#e5f2f4;}
[data-testid="stSidebar"] {background:#102c3a;}
h1,h2,h3 {color:#d9f4f2;}
[data-testid="stMetric"] {background:#123342;border:1px solid #24505d;border-radius:12px;padding:16px;}
[data-testid="stMetricValue"] {color:#63ded0;}
[data-testid="stMetricLabel"] {color:#bfd7df;}
</style>""", unsafe_allow_html=True)
px.defaults.template = 'plotly_dark'
BASE = Path(__file__).resolve().parent
EMPRESAS = {'BVN': 'Buenaventura', 'SCCO': 'Southern Copper', 'BAP': 'Credicorp'}
METODOS = ['PMS', 'Ponderado', 'SES', 'Holt', 'Holt-Winters aditivo', 'Holt-Winters multiplicativo']
COLORES = dict(zip(METODOS, ['#f0ba68','#b4a1ec','#64cbe3','#63ded0','#ec8b8b','#8eaee3']))

@st.cache_data
def cargar(ruta):
    d = pd.read_csv(ruta, sep=';', encoding='utf-8-sig')
    d['Fecha'] = pd.to_datetime(d['Fecha'], errors='raise')
    if d['Fecha'].duplicated().any():
        raise ValueError(f'Fechas duplicadas en {Path(ruta).name}')
    d['Adj Close'] = pd.to_numeric(d['Adj Close'], errors='raise')
    if d['Adj Close'].isna().any() or not np.isfinite(d['Adj Close']).all() or (d['Adj Close'] <= 0).any():
        raise ValueError(f'Precios ausentes o no válidos en {Path(ruta).name}')
    return d.sort_values('Fecha').set_index('Fecha')


def ajustar(s, metodo, ventana):
    if metodo == 'PMS':
        return s.rolling(ventana).mean().shift(1), None
    if metodo == 'Ponderado':
        pesos = np.arange(1, ventana+1, dtype=float)
        return s.rolling(ventana).apply(lambda x: np.dot(x, pesos)/pesos.sum(), raw=True).shift(1), None
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter('always')
        if metodo == 'SES':
            modelo = SimpleExpSmoothing(s, initialization_method='estimated')
        elif metodo == 'Holt':
            modelo = Holt(s, initialization_method='estimated')
        else:
            modelo = ExponentialSmoothing(s, trend='add', seasonal='add' if metodo.endswith('aditivo') else 'mul', seasonal_periods=12, initialization_method='estimated')
        fit = modelo.fit(optimized=True)
    if any('converge' in str(a.message).lower() for a in avisos):
        raise ValueError('La optimización no convergió')
    if not np.isfinite(fit.fittedvalues).all():
        raise ValueError('Ajuste no finito')
    return pd.Series(np.asarray(fit.fittedvalues), index=s.index), fit


def futuro(s, metodo, h, ventana, fit=None):
    if metodo in ['PMS', 'Ponderado']:
        valores = list(s.values)
        pred = []
        pesos = np.arange(1, ventana+1)
        for _ in range(h):
            x = np.asarray(valores[-ventana:])
            v = float(x.mean() if metodo == 'PMS' else np.dot(x, pesos)/pesos.sum())
            valores.append(v)
            pred.append(v)
        pred = np.asarray(pred)
    else:
        if fit is None:
            _, fit = ajustar(s, metodo, ventana)
        pred = np.asarray(fit.forecast(h))
    if not np.isfinite(pred).all() or (pred <= 0).any():
        raise ValueError('El modelo produce precios futuros no válidos')
    return pred


def errores(real, pred):
    e = np.asarray(real)-np.asarray(pred)
    return {'MAE': float(np.mean(abs(e))), 'RMSE': float(np.sqrt(np.mean(e**2))), 'MAPE (%)': float(np.mean(abs(e/np.asarray(real)))*100)}

@st.cache_data(show_spinner=False)
def evaluar(s, ventana, h=12):
    if len(s) < max(48, ventana+24):
        raise ValueError('Selecciona al menos 48 meses para comparar los modelos.')
    train, test = s.iloc[:-h], s.iloc[-h:]
    tablas, pruebas, ajustes, fallos = [], {}, {}, {}
    # Mismos meses de entrenamiento para todos; excluye inicialización estacional.
    comienzo = max(24, ventana)
    for metodo in METODOS:
        try:
            fitted, fit = ajustar(train, metodo, ventana)
            pred = futuro(train, metodo, h, ventana, fit)
            tablas.append({'Modelo': metodo, 'Fase': 'Entrenamiento', **errores(train.iloc[comienzo:], fitted.iloc[comienzo:])})
            tablas.append({'Modelo': metodo, 'Fase': 'Validación (12 meses)', **errores(test, pred)})
            pruebas[metodo] = pd.Series(pred, index=test.index)
            ajustes[metodo] = fitted
        except Exception as exc:
            fallos[metodo] = str(exc)
    return pd.DataFrame(tablas), pruebas, ajustes, fallos


def linea(series, titulo, eje='US$ por acción (cierre ajustado)'):
    f = go.Figure()
    for nombre, s in series.items():
        f.add_trace(go.Scatter(x=s.index, y=s.values, name=nombre, mode='lines'))
    f.update_layout(title=titulo, xaxis_title='Fecha', yaxis_title=eje, height=440, hovermode='x unified', template='plotly_dark', paper_bgcolor='#102633', plot_bgcolor='#102633')
    return f

try:
    mensuales = {t: cargar(BASE/'datos'/f'{t}_mensuales.csv')['Adj Close'] for t in EMPRESAS}
    diarios = {t: cargar(BASE/'datos'/f'{t}_diarios.csv')['Adj Close'] for t in EMPRESAS}
    datos = pd.concat(mensuales, axis=1).sort_index()
    inicio = max(s.index.min() for s in mensuales.values())
    fin = min(s.index.max() for s in mensuales.values())
    datos = datos.loc[inicio:fin].asfreq('MS')
    if datos.isna().any().any():
        raise ValueError('Hay meses ausentes en el periodo común; revisa los CSV.')
except Exception as exc:
    st.error(f'No se pudieron cargar los datos: {exc}')
    st.stop()

st.sidebar.title('📈 Observatorio de acciones')
pagina = st.sidebar.radio('¿Qué quieres conocer?', ['🏠 Resumen', '📈 Historia de cada empresa', '🔮 Pronósticos', '⚖️ Riesgos y simulador', '🎯 Comparación para decidir', '🎓 Análisis académico'])
ticker = st.sidebar.selectbox('Empresa', list(EMPRESAS), format_func=lambda t: f'{t} · {EMPRESAS[t]}')
ventana = st.sidebar.slider('Ventana de promedios móviles (meses)', 3, 24, 12)
st.sidebar.caption(f'Periodo mensual común: {inicio:%m/%Y}–{fin:%m/%Y} · {len(datos)} meses')
st.sidebar.caption('Fuente: Yahoo Finance. Archivos descargados con yfinance; datos históricos.')
s = datos[ticker].copy()

if pagina == '🏠 Resumen':
    st.title('📈 Acciones: Buenaventura, Southern Copper y Credicorp')
    st.write('Compara cómo evolucionaron los precios y sus fluctuaciones.')
    for col, t in zip(st.columns(3), EMPRESAS):
        x = datos[t]
        col.metric(f'{t} · {EMPRESAS[t]}', f'US$ {x.iloc[-1]:,.2f}', f'{(x.iloc[-1]/x.iloc[-2]-1)*100:+.2f}% último mes')
    n = st.radio('Periodo de comparación', [12, 60, len(datos)], format_func=lambda n: f'{n} meses', horizontal=True)
    corte = datos.tail(n)
    st.plotly_chart(linea({t: corte[t]/corte[t].iloc[0]*100 for t in EMPRESAS}, 'Evolución comparada: todos empiezan en 100', 'Índice base 100'), width='stretch')
    st.caption('La base 100 compara cambios relativos; los precios nominales de las empresas son distintos.')

elif pagina == '📈 Historia de cada empresa':
    st.title(f'📈 {EMPRESAS[ticker]} ({ticker})')
    frecuencia = st.radio('Frecuencia', ['Mensual', 'Diaria'], horizontal=True)
    x = s if frecuencia == 'Mensual' else diarios[ticker]
    rango = st.date_input('Periodo del gráfico', value=(x.index.min().date(), x.index.max().date()), min_value=x.index.min().date(), max_value=x.index.max().date())
    if len(rango) == 2:
        x = x.loc[str(rango[0]):str(rango[1])]
    if x.empty:
        st.info('No hay registros en el periodo elegido.')
        st.stop()
    st.plotly_chart(linea({ticker: x}, 'Historia del cierre ajustado'), width='stretch')
    st.subheader('Cambios por año: últimos cinco años completos y año actual')
    anual = s.groupby(s.index.year).last()
    cambio = anual.pct_change(fill_method=None).mul(100).tail(6)
    labels = [f'{a} (hasta {s.index[-1]:%m})' if a == s.index[-1].year and s.index[-1].month < 12 else str(a) for a in cambio.index]
    st.plotly_chart(px.bar(x=labels, y=cambio.values, labels={'x':'Año', 'y':'Cambio (%)'}), width='stretch')
    st.caption('Cada año se compara con diciembre del año anterior. El último año puede ser incompleto.')
    st.download_button('Descargar serie mostrada', x.rename('Adj Close').to_csv().encode('utf-8-sig'), f'{ticker}_{frecuencia}.csv')

elif pagina in ['🔮 Pronósticos', '🎓 Análisis académico']:
    st.title(f'{pagina} · {ticker}')
    with st.spinner('Comparando métodos...'):
        tabla, pruebas, ajustes, fallos = evaluar(s, ventana)
    if tabla.empty:
        st.error('Ningún modelo pudo estimarse.')
        st.stop()
    validacion = tabla[tabla['Fase'] == 'Validación (12 meses)'].sort_values('RMSE')
    mejor = validacion.iloc[0]['Modelo']
    if pagina == '🔮 Pronósticos':
        st.info(f'Menor RMSE en los últimos 12 meses reservados para validación: {mejor}.')
        metodo = st.selectbox('Método', list(validacion['Modelo']))
        h = st.slider('Meses a pronosticar', 1, 24, 12)
        try:
            pred = futuro(s, metodo, h, ventana)
        except Exception as exc:
            st.error(str(exc)); st.stop()
        fechas = pd.date_range(s.index[-1]+pd.offsets.MonthBegin(), periods=h, freq='MS')
        proy = pd.Series(pred, index=fechas)
        st.plotly_chart(linea({'Histórico': s.tail(60), 'Pronóstico': pd.concat([s.tail(1), proy])}, 'Histórico y escenario futuro'), width='stretch')
        st.metric(f'Precio ajustado estimado en {fechas[-1]:%m/%Y}', f'US$ {pred[-1]:,.2f}', f'{(pred[-1]/s.iloc[-1]-1)*100:+.2f}%')
        st.dataframe(validacion, hide_index=True)
        st.caption('Validación: se reservan 12 meses y se pronostican desde un único origen, sin usar sus precios para ajustar. Después se reajusta el modelo con todo el historial. No se muestran intervalos de incertidumbre.')
        st.download_button('Descargar pronóstico', proy.rename('Pronostico').to_csv().encode('utf-8-sig'), f'{ticker}_pronostico.csv')
    else:
        tipo = st.radio('Descomposición clásica', ['Aditiva', 'Multiplicativa'], horizontal=True)
        des = seasonal_decompose(s, model=tipo.lower().replace('aditiva','additive').replace('multiplicativa','multiplicative'), period=12)
        comp = pd.DataFrame({'Observado': s, 'Tendencia': des.trend, 'Estacionalidad': des.seasonal, 'Residuo': des.resid})
        c = st.selectbox('Componente', list(comp))
        st.plotly_chart(linea({c: comp[c]}, f'{tipo}: {c}', 'Factor sin unidad' if tipo=='Multiplicativa' and c in ['Estacionalidad','Residuo'] else 'US$ por acción'), width='stretch')
        st.latex(r'Y_t=T_t+E_t+R_t' if tipo=='Aditiva' else r'Y_t=T_t\times E_t\times R_t')
        st.caption('Periodo estacional de 12 meses. Los extremos de la tendencia quedan vacíos al usar medias centradas. La reconstrucción con el residuo no sirve para medir capacidad predictiva.')
        st.download_button('Descargar componentes', comp.to_csv().encode('utf-8-sig'), f'{ticker}_componentes.csv')
        st.subheader('Errores de entrenamiento y validación')
        st.dataframe(tabla, hide_index=True)
        st.caption('Entrenamiento: mismos meses para todos los métodos, excluyendo los primeros 24 o la ventana si es mayor. MAE/RMSE en dólares por acción; MAPE en porcentaje. La selección usa RMSE de validación.')
        st.plotly_chart(linea({'Real': s.tail(12), **pruebas}, 'Validación: precios observados y pronósticos'), width='stretch')
        st.plotly_chart(linea({'Real': s.iloc[:-12].tail(60), **{k:v.tail(60) for k,v in ajustes.items()}}, 'Ajustes en entrenamiento'), width='stretch')
        st.caption('PMS y ponderado usan únicamente meses anteriores. Los pesos ponderados crecen de 1 a la ventana. Holt-Winters usa tendencia aditiva y estacionalidad aditiva o multiplicativa.')
        st.download_button('Descargar errores', tabla.to_csv(index=False).encode('utf-8-sig'), f'{ticker}_errores.csv')
    if fallos:
        with st.expander('Métodos que no pudieron estimarse'):
            st.json(fallos)

elif pagina == '⚖️ Riesgos y simulador':
    st.title('⚖️ Fluctuaciones y escenarios de inversión')
    vol = datos.tail(61).pct_change(fill_method=None).std()*100
    st.plotly_chart(px.bar(x=vol.index, y=vol.values, labels={'x':'Empresa','y':'Desviación de cambios mensuales (%)'}, title='Fluctuaciones durante los últimos cinco años'), width='stretch')
    capital = st.number_input('Capital hipotético (US$)', min_value=1.0, value=1000.0)
    tasa = st.slider('Cambio hipotético (%)', -80, 100, 5)
    st.metric('Valor bajo el escenario', f'US$ {capital*(1+tasa/100):,.2f}', f'US$ {capital*tasa/100:+,.2f}')
    st.caption('Simulación proporcional del capital, sin comisiones ni impuestos. El cierre ajustado es una serie histórica corregida; no representa un precio ejecutable de compra.')
    cambios = s.pct_change(fill_method=None).mul(100).dropna()
    meses = cambios.groupby(cambios.index.month).agg(['mean','median','count'])
    meses.index = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
    st.subheader('Cambios históricos por mes del año')
    st.plotly_chart(px.bar(meses, x=meses.index, y='mean', labels={'mean':'Cambio medio (%)','x':'Mes'}), width='stretch')
    st.caption('Promedios históricos del periodo común. Un patrón mensual no determina una fecha segura de compra.')

else:
    st.title('🎯 Comparar perspectivas de las tres empresas')
    h = st.select_slider('Horizonte (meses)', [3,6,12], value=6)
    filas = []
    with st.spinner('Evaluando las tres empresas...'):
        for t in EMPRESAS:
            x = datos[t]
            tabla, _, _, _ = evaluar(x, ventana)
            val = tabla[tabla['Fase']=='Validación (12 meses)'].sort_values('RMSE') if not tabla.empty else pd.DataFrame()
            if val.empty: continue
            m = val.iloc[0]['Modelo']
            try:
                pred = futuro(x, m, h, ventana)
                filas.append({'Empresa': t, 'Modelo': m, 'Cambio proyectado (%)': (pred[-1]/x.iloc[-1]-1)*100, 'MAPE validación (%)': val.iloc[0]['MAPE (%)'], 'Fluctuación mensual (%)': x.tail(61).pct_change(fill_method=None).std()*100})
            except Exception as exc:
                st.info(f'{t}: {exc}')
    resumen = pd.DataFrame(filas)
    if not resumen.empty:
        st.plotly_chart(px.bar(resumen, x='Empresa', y='Cambio proyectado (%)', title=f'Escenario a {h} meses'), width='stretch')
        st.dataframe(resumen, hide_index=True)
        st.download_button('Descargar comparación', resumen.to_csv(index=False).encode('utf-8-sig'), 'comparacion_acciones.csv')
    st.caption('La empresa con mayor cambio proyectado merece revisión; esa posición no equivale a una recomendación automática de compra. El error mostrado evalúa un horizonte de 12 meses, que puede diferir del horizonte seleccionado.')

st.divider()
st.caption('Las series mensuales se descargaron directamente de Yahoo Finance. La fecha etiqueta el mes, no necesariamente el día del cierre. No se interpolan valores ni se eliminan movimientos extremos automáticamente.')
