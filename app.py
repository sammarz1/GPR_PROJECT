import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILO
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Analítica de Rendimiento Universitario",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Personalizado
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Estilo para las tarjetas de métricas (KPIs) */
    div[data-testid="stMetric"] {
        background-color: #f0f2f6;
        border: 1px solid #e6e9ef;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }
    
    div[data-testid="stMetric"] label {
        color: #31333F !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. CARGA Y PROCESAMIENTO DE DATOS
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_excel('DATA.xlsx')
    except FileNotFoundError:
        st.error("No se encontró el archivo 'DATA.xlsx - aggregated_COMPLETOS.csv'.")
        return None

    # Limpiar nombres de columnas
    df.columns = [col.replace('combined_', '').replace('_cleaned', '') for col in df.columns]
    
    # Definir columnas numéricas
    numerical_cols = ['Abandono', 'Eficiencia', 'Gestion', 'Docencia', 
                      'Rendimiento', 'Desempleo', 'Formacion']
    
    # CRÍTICO: Reemplazar 0 con NaN
    df_clean = df.copy()
    for col in numerical_cols:
        df_clean[col] = df_clean[col].replace(0, np.nan)
        
    return df_clean, numerical_cols

# Diccionario de definiciones de variables (Tooltip)
variable_info = {
    "Abandono": "Tasa de abandono (Escala 0-100%)",
    "Eficiencia": "Tasa de eficiencia (Escala 0-100%)",
    "Gestion": "Satisfacción con Gestión del Título (Escala 0-10)",
    "Docencia": "Satisfacción con la Docencia (Escala 0-10)",
    "Rendimiento": "Tasa de rendimiento (Escala 0-100%)",
    "Desempleo": "Tasa de empleo (Escala 0-100%)", 
    "Formacion": "Satisfacción con la Formación (Escala 0-10)"
}

# Cargar datos
data_load_state = st.text('Cargando datos...')
df, numerical_cols = load_data()
data_load_state.text('')

if df is None:
    st.stop()

# -----------------------------------------------------------------------------
# 3. FILTROS Y BARRA LATERAL
# -----------------------------------------------------------------------------
st.sidebar.title("Filtros")

# Sección de Definiciones
with st.sidebar.expander("¿Qué es cada variable?", expanded=True):
    st.markdown("""
    * **Abandono:** Tasa de abandono (0-100)
    * **Eficiencia:** Eficiencia (0-100)
    * **Gestión:** Gestión del Título (0-10)
    * **Docencia:** Satisfacción con docencia (0-10)
    * **Rendimiento:** Rendimiento (0-100)
    * **Empleo:** Tasa de empleo (0-100)
    * **Formación:** Formación (0-10)
    """)

st.sidebar.markdown("---")
st.sidebar.subheader("Opciones de Filtrado")

# Filtros
with st.sidebar.expander("Curso Académico", expanded=False):
    all_years = sorted(df['CURSO'].unique())
    selected_years = st.multiselect("Selecciona Cursos", all_years, default=[], help="Vacío = Todos")

with st.sidebar.expander("Tipo de Titulación", expanded=False):
    all_types = sorted(df['TIPO'].unique())
    selected_types = st.multiselect("Selecciona Tipos", all_types, default=[], help="Vacío = Todos")

with st.sidebar.expander("Escuela/Facultad", expanded=False):
    all_centers = sorted(df['CENTRO'].unique())
    selected_centers = st.multiselect("Selecciona Centros", all_centers, default=[], help="Vacío = Todos")

# Aplicar Filtros Globales
df_filtered = df.copy()
if selected_years:
    df_filtered = df_filtered[df_filtered['CURSO'].isin(selected_years)]
if selected_types:
    df_filtered = df_filtered[df_filtered['TIPO'].isin(selected_types)]
if selected_centers:
    df_filtered = df_filtered[df_filtered['CENTRO'].isin(selected_centers)]

st.sidebar.info(f"Mostrando **{len(df_filtered)}** registros.")
st.sidebar.warning("Nota: Valores de '0' excluidos.")

# -----------------------------------------------------------------------------
# 4. PESTAÑAS PRINCIPALES
# -----------------------------------------------------------------------------
st.title("Dashboard Métricas Carreras UPV")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Visión General", 
    "Análisis por Titulación", 
    "Distribuciones", 
    "Correlaciones", 
    "Ranking",
    "Datos"
])

# --- TAB 1: VISIÓN GENERAL ---
with tab1:
    st.header("Indicadores Clave (Promedio)")
    cols = st.columns(4)
    kpis = ['Rendimiento', 'Eficiencia', 'Abandono', 'Desempleo']
    labels = ['Rendimiento', 'Eficiencia', 'Abandono', 'Tasa Empleo'] 
    
    for i, col_name in enumerate(kpis):
        avg_val = df_filtered[col_name].mean()
        with cols[i]:
            label = labels[i]
            if pd.notna(avg_val):
                st.metric(label=label, value=f"{avg_val:.2f}%", help=variable_info[col_name])
            else:
                st.metric(label=label, value="N/A")

    st.subheader("Composición")
    c1, c2 = st.columns(2)
    with c1:
        # Pie chart con mejor estilo
        fig_pie = px.pie(
            df_filtered, 
            names='TIPO', 
            title='Distribución por Tipo', 
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(template="plotly_white")
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with c2:
        # Bar chart mejorado
        counts = df_filtered['CURSO'].value_counts().reset_index()
        counts.columns = ['CURSO', 'Conteo']
        fig_bar = px.bar(
            counts, 
            x='CURSO', 
            y='Conteo', 
            title='Registros por Curso',
            text='Conteo',
            color='Conteo', 
            color_continuous_scale='Viridis'
        )
        fig_bar.update_layout(template="plotly_white", xaxis_title="Curso Académico", yaxis_title="Número de Registros")
        st.plotly_chart(fig_bar, use_container_width=True)

# --- TAB 2: ANÁLISIS POR TITULACIÓN ---
with tab2:
    st.header("Detalle por Titulación")
    st.markdown("Selecciona una titulación específica para ver su ficha detallada y evolución.")
    
    available_degrees = sorted(df_filtered['TITULACION'].unique())
    selected_degree = st.selectbox("Busca una Titulación:", available_degrees)
    
    if selected_degree:
        subset_degree = df[df['TITULACION'] == selected_degree].sort_values('CURSO')
        
        if not subset_degree.empty:
            last_record = subset_degree.iloc[-1]
            st.subheader(f"{selected_degree}")
            st.caption(f"Centro: {last_record['CENTRO']} | Tipo: {last_record['TIPO']}")
            
            st.markdown("##### Últimos Datos Disponibles")
            km1, km2, km3, km4 = st.columns(4)
            km1.metric("Rendimiento", f"{last_record['Rendimiento']:.1f}%")
            km2.metric("Eficiencia", f"{last_record['Eficiencia']:.1f}%")
            km3.metric("Abandono", f"{last_record['Abandono']:.1f}%")
            km4.metric("Tasa Empleo", f"{last_record['Desempleo']:.1f}%") 
            
            ks1, ks2, ks3 = st.columns(3)
            ks1.metric("Satisf. Docencia (0-10)", f"{last_record['Docencia']:.2f}")
            ks2.metric("Satisf. Gestión (0-10)", f"{last_record['Gestion']:.2f}")
            ks3.metric("Satisf. Formación (0-10)", f"{last_record['Formacion']:.2f}")
            
            st.divider()
            
            st.subheader("Evolución Histórica")
            metrics_100 = ['Rendimiento', 'Eficiencia', 'Abandono', 'Desempleo']
            metrics_10 = ['Docencia', 'Gestion', 'Formacion']
            
            c_chart1, c_chart2 = st.columns(2)
            
            with c_chart1:
                st.markdown("**Indicadores de Porcentaje (0-100)**")
                df_long_100 = subset_degree.melt(id_vars=['CURSO'], value_vars=metrics_100, var_name='Indicador', value_name='Valor')
                df_long_100['Indicador'] = df_long_100['Indicador'].replace('Desempleo', 'Tasa Empleo')
                
                fig_evo1 = px.line(df_long_100, x='CURSO', y='Valor', color='Indicador', markers=True)
                fig_evo1.update_yaxes(range=[0, 105])
                fig_evo1.update_layout(template="plotly_white", legend_title_text='')
                st.plotly_chart(fig_evo1, use_container_width=True)
                
            with c_chart2:
                st.markdown("**Indicadores de Satisfacción (0-10)**")
                df_long_10 = subset_degree.melt(id_vars=['CURSO'], value_vars=metrics_10, var_name='Indicador', value_name='Valor')
                fig_evo2 = px.line(df_long_10, x='CURSO', y='Valor', color='Indicador', markers=True)
                fig_evo2.update_yaxes(range=[0, 10.5])
                fig_evo2.update_layout(template="plotly_white", legend_title_text='')
                st.plotly_chart(fig_evo2, use_container_width=True)

        else:
            st.warning("No hay datos disponibles para esta titulación con los filtros actuales.")

# --- TAB 3: DISTRIBUCIONES ---
with tab3:
    st.header("Distribuciones de Variables")
    col_sel, _ = st.columns([1, 2])
    target_col = col_sel.selectbox("Variable:", numerical_cols, index=4, format_func=lambda x: f"{x} - {variable_info[x]}")

    c1, c2 = st.columns(2)
    with c1:
        fig_hist = px.histogram(
            df_filtered, 
            x=target_col, 
            nbins=30, 
            title=f"Distribución: {target_col}", 
            color_discrete_sequence=['#2E86C1'], 
            marginal="box"
        )
        fig_hist.update_layout(template="plotly_white", bargap=0.1)
        st.plotly_chart(fig_hist, use_container_width=True)
    with c2:
        fig_box = px.box(
            df_filtered, 
            x='TIPO', 
            y=target_col, 
            color='TIPO', 
            title=f"{target_col} por Tipo",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_box.update_layout(template="plotly_white")
        st.plotly_chart(fig_box, use_container_width=True)

# --- TAB 4: CORRELACIONES ---
with tab4:
    st.header("Relaciones y Tendencias")
    
    st.subheader("Matriz de Correlación")
    corr_matrix = df_filtered[numerical_cols].corr()
    
    # Mejora de la Matriz de Correlación
    fig_corr = px.imshow(
        corr_matrix, 
        text_auto=".2f", 
        color_continuous_scale='RdBu_r', 
        zmin=-1, zmax=1,
        origin='lower',
        title="Correlación entre métricas (Rojo=Inv, Azul=Dir)",
        height=600  # Altura fija para evitar que se aplaste
    )
    # Ajustes para que encaje mejor
    fig_corr.update_layout(
        template="plotly_white",
        margin=dict(l=50, r=50, t=50, b=50)
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.subheader("Dispersión (Scatter)")
    cx1, cx2, cx3 = st.columns(3)
    x_ax = cx1.selectbox("Eje X", numerical_cols, index=1)
    y_ax = cx2.selectbox("Eje Y", numerical_cols, index=4)
    c_by = cx3.selectbox("Color", ['TIPO', 'CENTRO', 'CURSO'])
    
    fig_scat = px.scatter(
        df_filtered, 
        x=x_ax, 
        y=y_ax, 
        color=c_by, 
        hover_data=['TITULACION'], 
        title=f"{y_ax} vs {x_ax}", 
        opacity=0.7,
        size_max=15
    )
    fig_scat.update_layout(template="plotly_white")
    st.plotly_chart(fig_scat, use_container_width=True)

# --- TAB 5: RANKING ---
with tab5:
    st.header("Ranking de Titulaciones")
    rc1, rc2, rc3 = st.columns(3)
    r_metric = rc1.selectbox("Métrica:", numerical_cols, index=4)
    r_n = rc2.slider("Cantidad:", 5, 20, 10)
    r_ord = rc3.radio("Mostrar:", ["Top (Mejores)", "Bottom (Peores)"])
    
    rank_df = df_filtered.dropna(subset=[r_metric])
    if r_ord == "Top (Mejores)":
        sorted_df = rank_df.nlargest(r_n, r_metric)
        color_s = 'Teal'
    else:
        sorted_df = rank_df.nsmallest(r_n, r_metric)
        color_s = 'Reds'
        
    fig_rank = px.bar(
        sorted_df, 
        y='TITULACION', 
        x=r_metric, 
        color=r_metric, 
        orientation='h', 
        color_continuous_scale=color_s, 
        hover_data=['CENTRO'], 
        title=f"{r_ord} {r_n} por {r_metric}",
        text=r_metric
    )
    fig_rank.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig_rank.update_layout(yaxis=dict(autorange="reversed"), template="plotly_white")
    st.plotly_chart(fig_rank, use_container_width=True)

# --- TAB 6: DATOS ---
with tab6:
    st.header("Datos en Bruto")
    st.dataframe(df_filtered, use_container_width=True)
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar CSV", csv, "datos.csv", "text/csv")
