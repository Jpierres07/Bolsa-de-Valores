from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from motor_modelos import MODELOS, EXP, preparar, entrenamiento, validacion, escenarios

st.set_page_config(page_title='Bolsa de Valores | SCCO y BVN',layout='wide')
st.markdown('''<style>.stApp{background:#0b1925;color:#e5f2f4}[data-testid="stSidebar"]{background:#102c3a}h1,h2,h3{color:#d9f4f2}[data-testid="stMetric"]{background:#123342;border:1px solid #24505d;border-radius:12px;padding:14px}</style>''',unsafe_allow_html=True)
BASE=Path(__file__).resolve().parent
EMPRESAS={'SCCO':'Southern Copper','BVN':'Buenaventura'}
@st.cache_data
def cargar():
    series={};diarios={}
    for t in EMPRESAS:
        for freq,dic in [('mensuales',series),('diarios',diarios)]:
            df=pd.read_csv(BASE/'datos'/f'{t}_{freq}.csv',sep=';',encoding='utf-8-sig',parse_dates=['Fecha']).set_index('Fecha').sort_index()
            dic[t]=preparar(df['Adj Close']) if freq=='mensuales' else df['Adj Close']
    return series,diarios
series,diarios=cargar();s=series['SCCO']
st.sidebar.title('📈 Bolsa de Valores')
st.sidebar.caption('Guía')
pagina=st.sidebar.radio('Secciones',[
    '1. Presentación de la investigación',
    '2. Exploración histórica',
    '3. Descomposición clásica',
    '4. Comparación de ocho modelos',
    '5. Validación y pronósticos',
    '6. Inversión y conclusiones',
])
ticker=st.sidebar.selectbox('Acción',list(EMPRESAS),format_func=lambda x:f'{x} · {EMPRESAS[x]}')
s=series[ticker]
st.sidebar.caption(f'Datos mensuales: {s.index[0]:%m/%Y} – {s.index[-1]:%m/%Y}')
def grafico(datos,titulo,ylabel='US$ por acción'):
    fig=go.Figure()
    for nom,ser in datos.items():fig.add_trace(go.Scatter(x=ser.index,y=ser.values,name=nom,mode='lines'))
    fig.update_layout(title=titulo,height=460,template='plotly_dark',paper_bgcolor='#102633',plot_bgcolor='#102633',yaxis_title=ylabel,hovermode='x unified')
    return fig
@st.cache_data(show_spinner=False)
def calculos(t):return entrenamiento(series[t]),validacion(series[t])
if pagina=='1. Presentación de la investigación':
    st.title('Bolsa de Valores y Oportunidades de Inversión')
    st.subheader('Análisis de series temporales y pronóstico de las acciones de Southern Copper y Buenaventura')
    st.caption('Estudio académico · Datos mensuales de julio de 1996 a septiembre de 2026 · Pronósticos desde octubre de 2026')
    a,b,c=st.columns(3)
    a.metric('Empresas analizadas','2','SCCO y BVN',delta_color='off')
    b.metric('Observaciones mensuales',f'{len(s)}','Por empresa',delta_color='off')
    c.metric('Horizonte principal','6 meses','Oct. 2026 – Mar. 2027',delta_color='off')
    st.subheader('Planteamiento del problema')
    st.write('Los precios de las acciones cambian con el tiempo y pueden generar ganancias o pérdidas. Para quienes desean invertir en empresas mineras como Southern Copper y Buenaventura, resulta difícil identificar tendencias, comparar alternativas y estimar el riesgo utilizando únicamente los precios recientes. Por ello, se analizan sus cierres ajustados históricos mediante modelos de series temporales y se comparan sus errores para elaborar pronósticos y escenarios de inversión.')
    st.markdown('**Pregunta de investigación**')
    st.info('¿Qué modelos de series temporales presentan el mejor desempeño en el ajuste y pronóstico de los precios de cierre ajustado mensuales de Southern Copper (SCCO) y Buenaventura (BVN), y qué oportunidades y riesgos de inversión se identifican bajo tres escenarios —favorable, más probable y pesimista— para los próximos seis meses?')
    st.subheader('Objetivo general')
    st.write('Analizar y comparar modelos de series temporales para pronosticar los precios mensuales de SCCO y BVN, y evaluar oportunidades y riesgos de inversión en un horizonte de seis meses.')
    st.subheader('Objetivos específicos')
    st.markdown('''1. Describir la evolución histórica de los precios de cierre ajustado y sus variaciones.
2. Examinar la tendencia, estacionalidad y residuos mediante descomposición aditiva y multiplicativa.
3. Aplicar PMS3, PMS6, PMS12, PMP3, SES, Holt, Holt Winters aditivo y Holt Winters multiplicativo.
4. Comparar los modelos mediante MAE, RMSE y MAPE en entrenamiento y en una prueba temporal.
5. Proyectar precios y analizar escenarios pesimista, central y favorable, considerando el riesgo de cada acción.''')
    st.subheader('Justificación')
    st.write('El estudio aplica métodos del curso a datos bursátiles reales y facilita explicar cómo se comportan los precios, qué tan precisos han sido los modelos y por qué una proyección no garantiza una ganancia.')
    st.divider()
    st.title('Ficha técnica de los datos')
    st.write('La investigación usa **precios de cierre ajustado mensuales** como variable principal. Los datos diarios se muestran solo como complemento exploratorio.')
    fichas=[]
    for t,nom in EMPRESAS.items():
        m=series[t];d=diarios[t]
        fichas.append({'Empresa':nom,'Símbolo':t,'Inicio mensual':m.index.min().strftime('%m/%Y'),'Fin mensual':m.index.max().strftime('%m/%Y'),'Registros mensuales':len(m),'Registros diarios':len(d),'Faltantes mensuales':int(m.isna().sum()),'Precio mínimo (USD)':round(float(m.min()),2),'Precio máximo (USD)':round(float(m.max()),2)})
    st.dataframe(pd.DataFrame(fichas),hide_index=True,use_container_width=True)
    st.markdown('''| Elemento | Descripción |
|---|---|
| Tema | Precios bursátiles de Southern Copper y Buenaventura |
| Fuente | Yahoo Finance; archivos históricos recopilados por el grupo |
| Unidad de análisis | Una acción por empresa en cada periodo |
| Variable principal | `Adj Close` (precio de cierre ajustado) |
| Unidad de medida | Dólares estadounidenses por acción (US$) |
| Frecuencia principal | Mensual; una observación por mes |
| Frecuencia complementaria | Diaria; días con negociación bursátil |
| Periodo de estudio | Julio de 1996 a septiembre de 2026 |
| Métodos | Descomposición clásica, PMS, PMP, SES, Holt y Holt-Winters |
| Métricas | MAE, RMSE y MAPE |
| Validación | Últimos 12 meses separados para prueba temporal |
| Horizonte de pronóstico | 3, 6 y 12 meses; análisis principal de 6 meses |''')
    st.subheader('Preparación y control de calidad')
    st.markdown('''- Se ordenan las fechas y se comprueba que no existan meses faltantes.
- Se mantiene el **cierre ajustado**, que contempla ajustes históricos del proveedor, en lugar de mezclarlo con el precio de cierre sin ajustar.
- Los valores extremos se revisan visualmente; no se eliminan automáticamente, porque podrían corresponder a movimientos reales del mercado.
- La comparación de ajuste utiliza los mismos meses evaluados para los ocho modelos (desde la observación 25).
- La validación separa los últimos 12 meses sin utilizarlos para estimar el modelo.''')
    st.download_button('Descargar ficha técnica (CSV)',pd.DataFrame(fichas).to_csv(index=False).encode('utf-8-sig'),'ficha_tecnica_SCCO_BVN.csv')
elif pagina=='2. Exploración histórica':
    st.title('Exploración histórica de las acciones')
    st.write('**¿Qué veremos?** Cómo han cambiado los precios y qué tan grandes fueron sus variaciones. Utiliza la vista mensual para explicar tendencias y la diaria solo para movimientos recientes.')
    freq=st.radio('Frecuencia',['Mensual','Diaria'],horizontal=True)
    x=pd.concat(series if freq=='Mensual' else diarios,axis=1).dropna()
    seleccion=st.multiselect('Empresas',list(EMPRESAS),default=list(EMPRESAS))
    if not seleccion:st.stop()
    forma=st.radio('Comparación',['Precio ajustado','Índice base 100'],horizontal=True)
    x=x[seleccion]
    if forma=='Índice base 100':x=x.div(x.iloc[0]).mul(100)
    st.plotly_chart(grafico({c:x[c] for c in seleccion},'Evolución histórica',forma),use_container_width=True)
    st.subheader('Rendimientos mensuales')
    r=pd.concat(series,axis=1).pct_change(fill_method=None).mul(100).dropna()
    st.plotly_chart(grafico({c:r[c] for c in seleccion},'Cambios mensuales (%)','%'),use_container_width=True)
    st.subheader('Acontecimientos históricos relevantes')
    st.caption('Contexto económico orientativo; la coincidencia temporal no demuestra causalidad.')
    st.dataframe(pd.DataFrame([
        {'Periodo':'2003–2007','Acontecimiento':'Expansión de la demanda china de metales','Relación':'Contexto favorable para el cobre y SCCO'},
        {'Periodo':'2008–2009','Acontecimiento':'Crisis financiera mundial','Relación':'Caídas bursátiles y posterior recuperación'},
        {'Periodo':'2010–2011','Acontecimiento':'Auge del oro y la plata','Relación':'Contexto favorable para BVN'},
        {'Periodo':'2013–2015','Acontecimiento':'Debilidad de los precios de los metales','Relación':'Presión sobre cotizaciones mineras'},
        {'Periodo':'2020–2021','Acontecimiento':'Pandemia y recuperación económica','Relación':'Alta volatilidad y recuperación del cobre'},
        {'Periodo':'2025–2026','Acontecimiento':'Precios altos de metales y resultados empresariales','Relación':'Coincide con repuntes recientes de SCCO y BVN'},
    ]),hide_index=True,use_container_width=True)
elif pagina=='3. Descomposición clásica':
    st.title(f'Descomposición clásica · {ticker}')
    st.write('**¿Qué veremos?** Separamos la serie en tendencia, estacionalidad y variación no explicada. No toda fluctuación implica una estacionalidad repetible.')
    tipo=st.radio('Modelo',['Aditivo','Multiplicativo'],horizontal=True)
    dec=seasonal_decompose(s,model='additive' if tipo=='Aditivo' else 'multiplicative',period=12)
    from plotly.subplots import make_subplots
    fig=make_subplots(rows=4,cols=1,shared_xaxes=True,subplot_titles=['Observado','Tendencia','Estacionalidad','Residuo'],vertical_spacing=.07)
    for i,v in enumerate([dec.observed,dec.trend,dec.seasonal,dec.resid],1):fig.add_trace(go.Scatter(x=v.index,y=v.values,mode='lines',showlegend=False),row=i,col=1)
    fig.update_layout(height=820,template='plotly_dark')
    st.plotly_chart(fig,use_container_width=True)
    st.caption('Periodicidad 12 meses. Una descomposición no demuestra por sí sola estacionalidad estable.')
elif pagina=='4. Comparación de ocho modelos':
    st.title(f'Ocho modelos · ajuste en entrenamiento · {ticker}')
    st.write('**¿Qué veremos?** Qué método reproduce mejor los precios históricos. Un RMSE menor indica menor error de ajuste, pero no asegura mejores pronósticos.')
    (tabla,ajustes,fallos),_=calculos(ticker)
    st.dataframe(tabla,hide_index=True,use_container_width=True)
    elegidos=st.multiselect('Mostrar ajustes',MODELOS,default=[tabla.iloc[0]['Modelo']])
    st.plotly_chart(grafico({'Observado':s,**{m:ajustes[m] for m in elegidos if m in ajustes}},'Observado vs. ajustado'),use_container_width=True)
    st.info(f'Menor RMSE de entrenamiento: {tabla.iloc[0]["Modelo"]} = {tabla.iloc[0]["RMSE"]:.4f} USD/acción. Se comparan los últimos {len(s)-24} meses (mismo intervalo para los ocho métodos).')
    st.caption('PMP3 utiliza pesos 1, 2 y 3, con mayor peso para el mes más reciente. Los promedios móviles se desplazan un mes. Los modelos exponenciales se optimizan automáticamente.')
    st.download_button('Descargar tabla de entrenamiento',tabla.to_csv(index=False).encode('utf-8-sig'),f'{ticker}_entrenamiento.csv')
    if fallos:st.warning(str(fallos))
elif pagina=='5. Validación y pronósticos':
    st.title(f'Validación temporal · {ticker}')
    st.write('**¿Qué veremos?** Comparamos pronósticos contra 12 meses que no se usaron al estimar los modelos. Esta es la evaluación más útil para elegir un método predictivo.')
    _,(tabla,preds,fallos)=calculos(ticker)
    st.write('Entrenamiento hasta septiembre de 2025; prueba de 12 meses: octubre de 2025 a septiembre de 2026. Pronóstico de 12 pasos desde un único origen, sin usar los precios de prueba para ajustar.')
    st.dataframe(tabla,hide_index=True,use_container_width=True)
    elegido=st.selectbox('Pronóstico de prueba',list(preds),index=list(preds).index(tabla.iloc[0]['Modelo']))
    st.plotly_chart(grafico({'Observado':s.iloc[-24:],'Pronóstico':preds[elegido]},'Evaluación fuera de muestra'),use_container_width=True)
    st.download_button('Descargar errores de prueba',tabla.to_csv(index=False).encode('utf-8-sig'),f'{ticker}_prueba.csv')
    if fallos:st.warning(str(fallos))
    st.divider()
    st.title(f'Pronósticos y escenarios · {ticker}')
    st.write('**¿Qué veremos?** Una trayectoria central y dos escenarios exploratorios. Los resultados son estimaciones, no precios garantizados.')
    _,(tabla,_,_)=calculos(ticker)
    orden=[m for m in tabla['Modelo'] if m in EXP]
    modelo=st.selectbox('Modelo de suavizamiento',orden)
    h=st.select_slider('Horizonte de inversión (meses)',options=[3,6,12],value=6)
    e=escenarios(s,modelo,h=h)
    st.plotly_chart(grafico({'Últimos 36 meses':s.iloc[-36:],**{c:e[c] for c in e}},'Tres escenarios de precio'),use_container_width=True)
    ultimo=float(s.iloc[-1]);resultado=e.copy();resultado['Rentabilidad central (%)']=(resultado['Central']/ultimo-1)*100
    st.dataframe(resultado.round(3),use_container_width=True)
    st.metric('Último cierre ajustado',f'US$ {ultimo:,.2f}')
    st.metric('Rentabilidad central al horizonte',f'{(e["Central"].iloc[-1]/ultimo-1)*100:+.2f}%')
    st.warning('Las bandas favorable y pesimista son simulaciones exploratorias con remuestreo de residuos y escala √h. NO son intervalos de predicción calibrados ni probabilidades garantizadas. No incluyen comisiones, impuestos, dividendos ni tipo de cambio.')
    st.download_button('Descargar escenarios',resultado.to_csv().encode('utf-8-sig'),f'{ticker}_escenarios.csv')
elif pagina=='6. Inversión y conclusiones':
    st.title('Comparación de rentabilidad, riesgo y escenarios')
    st.write('**¿Qué veremos?** La rentabilidad proyectada y la volatilidad histórica de ambas acciones para debatir una posible decisión de inversión.')
    filas=[]
    h=st.select_slider('Horizonte (meses)',options=[3,6,12],value=6)
    for t,s1 in series.items():
        ret=s1.pct_change(fill_method=None).dropna()
        _,(test,_,_)=calculos(t)
        candidato=next((m for m in test['Modelo'] if m in EXP),None)
        if candidato is None:continue
        e=escenarios(s1,candidato,h=h);ultimo=s1.iloc[-1]
        filas.append({'Acción':t,'Modelo seleccionado':candidato,'Precio base USD':ultimo,'Volatilidad anualizada %':ret.std()*np.sqrt(12)*100,'Rent. pesimista %':(e.iloc[-1,0]/ultimo-1)*100,'Rent. central %':(e.iloc[-1,1]/ultimo-1)*100,'Rent. favorable %':(e.iloc[-1,2]/ultimo-1)*100})
    tabla=pd.DataFrame(filas);st.dataframe(tabla.round(2),hide_index=True,use_container_width=True)
    st.plotly_chart(px.bar(tabla,x='Acción',y=['Rent. pesimista %','Rent. central %','Rent. favorable %'],barmode='group',title='Escenarios de rentabilidad potencial (%)'),use_container_width=True)
    st.warning('No se recomienda comprar únicamente por el rendimiento central proyectado. Comparar riesgo, incertidumbre, costos, liquidez y horizonte.')
    st.download_button('Descargar comparación',tabla.to_csv(index=False).encode('utf-8-sig'),'comparacion_inversion.csv')
    st.subheader('Conclusiones de la investigación')
    st.markdown('''**Lectura de los resultados:** la mejor alternativa de ajuste histórico no necesariamente es la de mejor validación temporal. Las decisiones deben contrastar pronósticos, escenarios y volatilidad.

**Límites:** los escenarios son exploratorios, no garantizan rendimientos y no incluyen costos, impuestos ni tipo de cambio.''')
    st.info('Mensaje final: una inversión responsable compara rentabilidad potencial y riesgo; ninguna proyección asegura ganancias.')
st.divider();st.caption('Fuente: Yahoo Finance | Datos hasta septiembre de 2026')
