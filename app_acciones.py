from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

st.set_page_config(page_title='Análisis bursátil | Entregable 1', layout='wide')
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


def errores(real, pred):
    e = np.asarray(real)-np.asarray(pred)
    return {'MAE': float(np.mean(abs(e))), 'RMSE': float(np.sqrt(np.mean(e**2))), 'MAPE (%)': float(np.mean(abs(e/np.asarray(real)))*100)}


def linea(series, titulo, eje='US$ por acción (cierre ajustado)'):
    f = go.Figure()
    for nombre, s in series.items():
        f.add_trace(go.Scatter(x=s.index, y=s.values, name=nombre, mode='lines'))
    f.update_layout(title=titulo, xaxis_title='Fecha', yaxis_title=eje, height=500, hovermode='x unified', dragmode='zoom', xaxis=dict(rangeslider=dict(visible=True), rangeselector=dict(buttons=[dict(count=1,label='1 año',step='year',stepmode='backward'),dict(count=5,label='5 años',step='year',stepmode='backward'),dict(label='Todo',step='all')])), template='plotly_dark', paper_bgcolor='#102633', plot_bgcolor='#102633')
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

st.sidebar.title('Análisis bursátil')
pagina = st.sidebar.radio('Contenido del entregable', ['Presentación del estudio', 'Registro de los datos', 'Exploración de precios', 'Descomposición temporal', 'Estimación y errores'])
ticker = st.sidebar.selectbox('Acción', list(EMPRESAS), format_func=lambda t: f'{t} | {EMPRESAS[t]}')
s = datos[ticker].copy()
st.sidebar.caption(f'{inicio:%m/%Y} a {fin:%m/%Y} | {len(datos)} registros mensuales por empresa')

if pagina == 'Presentación del estudio':
    st.title('Análisis de precios históricos de BVN, SCCO y BAP mediante series temporales')
    st.subheader('Planteamiento del problema')
    st.write('Los precios de las acciones cambian a lo largo del tiempo. Estas variaciones dificultan reconocer su evolución y distinguir movimientos de largo plazo de fluctuaciones transitorias. Se estudiarán las acciones de Buenaventura, Southern Copper y Credicorp para describir sus patrones y comparar el ajuste de métodos de suavizamiento.')
    st.write('¿Cómo evolucionaron los precios de cierre ajustados de las tres empresas y qué método presenta menor error de ajuste en el periodo analizado?')
    st.subheader('Justificación')
    st.write('El análisis permite aplicar herramientas de series temporales a información bursátil: identificar tendencias, examinar posibles patrones estacionales y comparar métodos con criterios comunes. En términos prácticos, organiza la información histórica para comprender los movimientos de precios de empresas de los sectores minero y financiero.')
    st.subheader('Objetivo del primer entregable')
    st.write('Verificar la consistencia de los datos, explorar las series y aplicar descomposición clásica, promedios móviles y suavizamiento exponencial. La comparación se limita a los errores de entrenamiento.')

elif pagina == 'Registro de los datos':
    st.title('Características y revisión de la base')
    ficha = pd.DataFrame([
        ('Fuente', 'Yahoo Finance; extracción mediante yfinance'),
        ('Empresas', 'BVN: Buenaventura; SCCO: Southern Copper; BAP: Credicorp'),
        ('Variable', 'Cierre ajustado (Adj Close)'),
        ('Unidad', 'Dólares estadounidenses por acción'),
        ('Frecuencia disponible', 'Diaria y mensual, descargadas por separado'),
        ('Periodo mensual común', f'{inicio:%m/%Y} a {fin:%m/%Y}'),
        ('Observaciones mensuales', f'{len(datos)} por empresa'),
        ('Uso de las frecuencias', 'Diaria y mensual: exploración. Mensual: descomposición y ajuste'),
        ('Faltantes', 'Sin faltantes en el periodo mensual común verificado; no se interpolan'),
        ('Duplicados', 'Se comprueba que no existan fechas duplicadas al cargar cada archivo'),
        ('Valores atípicos', 'Se examinan gráficamente y no se eliminan automáticamente')
    ],columns=['Característica','Descripción'])
    st.dataframe(ficha, hide_index=True, width='stretch')
    st.caption('Los CSV mensuales de BVN y BAP comienzan en julio de 1996; SCCO incluye junio. Se utiliza julio como inicio común. Las fechas mensuales identifican el mes, no el día efectivo de cierre.')
    filas=[]
    for t in EMPRESAS:
        for f,x in [('Diaria',diarios[t]),('Mensual común',datos[t])]:
            filas.append({'Empresa':t,'Frecuencia':f,'Inicio':x.index.min().date(),'Fin':x.index.max().date(),'Registros':len(x),'Faltantes':int(x.isna().sum()),'Duplicados':int(x.index.duplicated().sum())})
    st.dataframe(pd.DataFrame(filas),hide_index=True,width='stretch')
    st.caption('El cierre ajustado incorpora correcciones por dividendos y divisiones de acciones para facilitar comparaciones históricas.')

elif pagina == 'Exploración de precios':
    st.title('Comportamiento de las cotizaciones')
    frecuencia=st.radio('Frecuencia del gráfico',['Mensual','Diaria'],horizontal=True)
    base=datos if frecuencia=='Mensual' else pd.concat(diarios,axis=1)
    empresas=st.multiselect('Series visibles',list(EMPRESAS),default=[ticker])
    escala=st.radio('Forma de comparación',['Precio ajustado','Índice base 100'],horizontal=True)
    rango=st.date_input('Intervalo a explorar',value=(base.index.min().date(),base.index.max().date()),min_value=base.index.min().date(),max_value=base.index.max().date())
    x=base.loc[str(rango[0]):str(rango[1])] if len(rango)==2 else base
    if not empresas or x.empty:
        st.info('Selecciona al menos una empresa y un intervalo con datos.');st.stop()
    x=x[empresas]
    if escala=='Índice base 100':x=x.div(x.iloc[0]).mul(100)
    fig=linea({t:x[t] for t in empresas},'Trayectoria de las series','Índice base 100' if escala=='Índice base 100' else 'US$ por acción')
    if st.checkbox('Escala logarítmica',value=False):fig.update_yaxes(type='log')
    st.plotly_chart(fig,width='stretch',config={'displaylogo':False,'scrollZoom':True})
    st.caption('Arrastra para ampliar un tramo, usa el control inferior para desplazarte y pulsa una leyenda para ocultar una serie. Haz doble clic para recuperar la vista.')
    st.caption('En base 100, un valor de 120 representa un aumento del 20% frente al primer registro del intervalo. Las caídas o saltos aislados requieren revisión; el gráfico no identifica sus causas.')
    st.subheader('Distribución y valores extremos')
    vista=st.radio('Gráfico complementario',['Distribución de precios','Cambios mensuales'],horizontal=True)
    if vista=='Distribución de precios':
        st.plotly_chart(px.box(datos,points='outliers',labels={'variable':'Empresa','value':'Cierre ajustado (US$)'}),width='stretch')
        st.caption('Los puntos fuera de los límites del diagrama son candidatos a revisión. En una serie con tendencia, no implican necesariamente errores de registro.')
    else:
        cambios=datos.pct_change(fill_method=None).mul(100).dropna()
        st.plotly_chart(linea({t:cambios[t] for t in empresas},'Variaciones entre meses consecutivos','Cambio (%)'),width='stretch')
    st.dataframe(datos[empresas].describe().T, width='stretch')

elif pagina == 'Descomposición temporal':
    st.title(f'Componentes de la serie de {ticker}')
    tipo=st.radio('Estructura',['Aditiva','Multiplicativa'],horizontal=True)
    des=seasonal_decompose(s,model='additive' if tipo=='Aditiva' else 'multiplicative',period=12)
    componentes=pd.DataFrame({'Observado':s,'Tendencia':des.trend,'Estacionalidad':des.seasonal,'Residuo':des.resid})
    from plotly.subplots import make_subplots
    fig=make_subplots(rows=4,cols=1,shared_xaxes=True,subplot_titles=list(componentes),vertical_spacing=.07)
    for i,c in enumerate(componentes,1):
        fig.add_trace(go.Scatter(x=s.index,y=componentes[c],name=c,mode='lines'),row=i,col=1)
    fig.update_layout(height=850,template='plotly_dark',showlegend=False,hovermode='x unified')
    st.plotly_chart(fig,width='stretch',config={'displaylogo':False,'scrollZoom':True})
    st.latex(r'Y_t=T_t+E_t+R_t' if tipo=='Aditiva' else r'Y_t=T_t\times E_t\times R_t')
    st.caption('Se usa un ciclo de 12 meses. La tendencia resume cambios prolongados; la estacionalidad estima un patrón anual; el residuo recoge lo restante. Los extremos de la tendencia quedan sin estimar por el promedio centrado.')
    st.info('En la estructura aditiva, los componentes se expresan en dólares por acción. En la multiplicativa, estacionalidad y residuo son factores: 1 indica ausencia de efecto. La descomposición no demuestra por sí sola una estacionalidad estable.')
    st.download_button('Exportar componentes',componentes.to_csv().encode('utf-8-sig'),f'{ticker}_descomposicion.csv')

else:
    st.title(f'Ajuste de métodos para {ticker}')
    ventana=st.slider('Ventana de promedios móviles (meses)',3,24,12)
    st.caption('Se utilizan todos los meses como entrenamiento. Los errores se comparan en los mismos registros, excluyendo los primeros 24 meses para reducir el efecto de inicialización.')
    ajustes={};filas=[];fallos={};comienzo=max(24,ventana)
    with st.spinner('Estimando métodos...'):
        for m in METODOS:
            try:
                f,fit=ajustar(s,m,ventana)
                ajustes[m]=f
                filas.append({'Método':m,**errores(s.iloc[comienzo:],f.iloc[comienzo:]),'Meses evaluados':len(s)-comienzo})
            except Exception as exc:fallos[m]=str(exc)
    if not filas:st.error('No se pudieron calcular los métodos.');st.stop()
    tabla=pd.DataFrame(filas).sort_values('RMSE')
    visibles=st.multiselect('Métodos visibles',list(ajustes),default=[tabla.iloc[0]['Método']])
    st.plotly_chart(linea({'Observado':s,**{m:ajustes[m] for m in visibles}},'Valores observados y ajustes'),width='stretch',config={'displaylogo':False,'scrollZoom':True})
    st.dataframe(tabla,hide_index=True,width='stretch')
    ganador=tabla.iloc[0]
    st.info(f'{ganador["Método"]} presenta el menor RMSE de entrenamiento: US$ {ganador["RMSE"]:,.2f} por acción. Este resultado describe el ajuste al historial; no evalúa precisión futura.')
    st.caption('MAE: error absoluto medio. RMSE: penaliza más los errores grandes. Ambos están en dólares por acción. MAPE: error absoluto porcentual medio. Menores valores indican mejor ajuste en los registros evaluados.')
    with st.expander('Cómo se calculan los métodos'):
        st.write('El promedio móvil simple asigna el mismo peso a los meses anteriores. El ponderado asigna pesos de 1 a la ventana, dando más importancia al mes más reciente. Ambos se desplazan un mes para evitar usar el precio que se intenta estimar. SES estima nivel; Holt incorpora tendencia; Holt-Winters incorpora tendencia y estacionalidad de 12 meses, aditiva o multiplicativa. Los parámetros exponenciales se optimizan por suma de errores cuadrados.')
    if fallos:st.warning('Métodos sin estimación: '+str(fallos))
    st.download_button('Exportar errores de entrenamiento',tabla.to_csv(index=False).encode('utf-8-sig'),f'{ticker}_errores_entrenamiento.csv')

st.divider()
st.caption('Primer entregable | Fuente: Yahoo Finance | Variable: cierre ajustado | Datos históricos hasta septiembre de 2026')
