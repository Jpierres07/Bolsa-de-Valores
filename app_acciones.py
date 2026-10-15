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
st.sidebar.title('Bolsa de Valores e Inversión')
pagina=st.sidebar.radio('Módulo',['Exploración histórica','Descomposición','Modelos y entrenamiento','Validación temporal','Pronósticos y escenarios','Comparación de inversión','Datos de metales'])
ticker=st.sidebar.selectbox('Acción',list(EMPRESAS),format_func=lambda x:f'{x} · {EMPRESAS[x]}')
s=series[ticker]
st.sidebar.caption(f'{len(s)} meses | {s.index[0]:%m/%Y} – {s.index[-1]:%m/%Y}')
def grafico(datos,titulo,ylabel='US$ por acción'):
    fig=go.Figure()
    for nom,ser in datos.items():fig.add_trace(go.Scatter(x=ser.index,y=ser.values,name=nom,mode='lines'))
    fig.update_layout(title=titulo,height=460,template='plotly_dark',paper_bgcolor='#102633',plot_bgcolor='#102633',yaxis_title=ylabel,hovermode='x unified')
    return fig
@st.cache_data(show_spinner=False)
def calculos(t):return entrenamiento(series[t]),validacion(series[t])
if pagina=='Exploración histórica':
    st.title('Exploración histórica de las acciones')
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
    st.plotly_chart(px.box(r[seleccion],points='outliers',title='Posibles valores extremos de rendimientos'),use_container_width=True)
    st.caption('Los movimientos extremos reales no deben eliminarse automáticamente.')
elif pagina=='Descomposición':
    st.title(f'Descomposición clásica · {ticker}')
    tipo=st.radio('Modelo',['Aditivo','Multiplicativo'],horizontal=True)
    dec=seasonal_decompose(s,model='additive' if tipo=='Aditivo' else 'multiplicative',period=12)
    from plotly.subplots import make_subplots
    fig=make_subplots(rows=4,cols=1,shared_xaxes=True,subplot_titles=['Observado','Tendencia','Estacionalidad','Residuo'],vertical_spacing=.07)
    for i,v in enumerate([dec.observed,dec.trend,dec.seasonal,dec.resid],1):fig.add_trace(go.Scatter(x=v.index,y=v.values,mode='lines',showlegend=False),row=i,col=1)
    fig.update_layout(height=820,template='plotly_dark')
    st.plotly_chart(fig,use_container_width=True)
    st.caption('Periodicidad 12 meses. Una descomposición no demuestra por sí sola estacionalidad estable.')
elif pagina=='Modelos y entrenamiento':
    st.title(f'Ocho modelos · ajuste en entrenamiento · {ticker}')
    (tabla,ajustes,fallos),_=calculos(ticker)
    st.dataframe(tabla,hide_index=True,use_container_width=True)
    elegidos=st.multiselect('Mostrar ajustes',MODELOS,default=[tabla.iloc[0]['Modelo']])
    st.plotly_chart(grafico({'Observado':s,**{m:ajustes[m] for m in elegidos if m in ajustes}},'Observado vs. ajustado'),use_container_width=True)
    st.info(f'Menor RMSE de entrenamiento: {tabla.iloc[0]["Modelo"]} = {tabla.iloc[0]["RMSE"]:.4f} USD/acción. Se comparan los últimos {len(s)-24} meses (mismo intervalo para los ocho métodos).')
    st.caption('PMP3 utiliza pesos 1, 2 y 3, con mayor peso para el mes más reciente. Los promedios móviles se desplazan un mes. Los modelos exponenciales se optimizan automáticamente.')
    st.download_button('Descargar tabla de entrenamiento',tabla.to_csv(index=False).encode('utf-8-sig'),f'{ticker}_entrenamiento.csv')
    if fallos:st.warning(str(fallos))
elif pagina=='Validación temporal':
    st.title(f'Validación temporal · {ticker}')
    _,(tabla,preds,fallos)=calculos(ticker)
    st.write('Entrenamiento hasta septiembre de 2025; prueba de 12 meses: octubre de 2025 a septiembre de 2026. Pronóstico de 12 pasos desde un único origen, sin usar los precios de prueba para ajustar.')
    st.dataframe(tabla,hide_index=True,use_container_width=True)
    elegido=st.selectbox('Pronóstico de prueba',list(preds),index=list(preds).index(tabla.iloc[0]['Modelo']))
    st.plotly_chart(grafico({'Observado':s.iloc[-24:],'Pronóstico':preds[elegido]},'Evaluación fuera de muestra'),use_container_width=True)
    st.download_button('Descargar errores de prueba',tabla.to_csv(index=False).encode('utf-8-sig'),f'{ticker}_prueba.csv')
    if fallos:st.warning(str(fallos))
elif pagina=='Pronósticos y escenarios':
    st.title(f'Pronósticos y escenarios · {ticker}')
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
elif pagina=='Comparación de inversión':
    st.title('Comparación de rentabilidad, riesgo y escenarios')
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
else:
    st.title('Metales internacionales y acciones mineras')
    d=pd.read_csv(BASE/'datos'/'metales_y_acciones.csv',parse_dates=['Fecha']).set_index('Fecha')
    st.caption('Banco Mundial (Pink Sheet): cobre, oro y plata. Variables complementarias de contexto, no regresores de SES/Holt/Holt-Winters.')
    metal=st.selectbox('Metal',['Copper','Gold','Silver'])
    st.plotly_chart(grafico({metal:d[metal]},'Precio internacional de '+metal,'Precio del metal (unidad de origen)'),use_container_width=True)
    st.subheader('Correlación descriptiva de variaciones mensuales')
    retornos=d[[f'{t}_AdjClose' for t in EMPRESAS]+['Copper','Gold','Silver']].pct_change(fill_method=None).dropna()
    st.dataframe(retornos.corr().round(3),use_container_width=True)
    st.caption('Correlación contemporánea no implica causalidad ni capacidad predictiva.')
st.divider();st.caption('Fuente bursátil: Yahoo Finance, archivos del grupo | Metales: Banco Mundial Pink Sheet | Datos hasta septiembre de 2026')
