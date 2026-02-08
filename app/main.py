import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
import re
import numpy as np
from rapidfuzz import process, fuzz
from forecast_models import (
        to_monthly_series,
        evaluate_models,
        fit_best_and_forecast,
        residual_quantile_bands,
    )

# Configuración general
st.set_page_config(
    page_title="Proyecto de Tesis",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data():

    df = pd.read_csv("app/data/data_interservice.csv")

    # Columnas que deberían ser numéricas
    columnas_numericas = [
        "AÑO DOCUMENTO","MES","subtipo","SUB_PED","PEDIDO",
        "VALOR NETO","por_iva","IVA","VALOR CON IVA",
        "% ICA","RETENCION  ICA","%RETENCION","RETEFUENTE",
        "VAL TOTAL"
    ]

    for col in columnas_numericas:
        if col in df.columns:

            # Convertir todo a string primero
            df[col] = df[col].astype(str)

            # Limpiar formato latino
            df[col] = (
                df[col]
                .str.replace(".", "", regex=False)   # miles
                .str.replace(",", ".", regex=False)  # decimales
                .str.replace(" ", "", regex=False)   # espacios
                .str.replace("$", "", regex=False)   # moneda
            )

            # Convertir a numérico
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

@st.cache_data
def load_etl_final():
    df = pd.read_csv("app/data/data_interservice_etl_final.csv")
    # Columnas que deberían ser numéricas
    columnas_numericas = [
         "diff_factura_servicio"
    ]

    for col in columnas_numericas:
        if col in df.columns:

            # Convertir todo a string primero
            df[col] = df[col].astype(str)

            # Limpiar formato latino
            df[col] = (
                df[col]
                .str.replace(".", "", regex=False)   # miles
                .str.replace(",", ".", regex=False)  # decimales
                .str.replace(" ", "", regex=False)   # espacios
                .str.replace("$", "", regex=False)   # moneda
            )

            # Convertir a numérico
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# ===============================
# LOGO PRINCIPAL
# ===============================

logo = Image.open("app/assets/logo.png")

st.sidebar.image(logo, width=180)

st.sidebar.markdown(
    "<h2 style='text-align:center;'>📈 Sistema de alertas con aprendizaje automático para la detección de anomalías en procesos de facturación para una empresa de logística en Colombia </h2>",
    unsafe_allow_html=True
)

st.sidebar.divider()

# ===============================
# MENÚ
# ===============================

menu = st.sidebar.radio(
    "📂 Módulos",
    [
        "🏁 Introducción",
        "🧹 ETL (Preparación de Datos)",
        "📊 EDA",
        "📈 Pronósticos + 🚨 Anomalías",
        "🤖 MCP + Bot",
    ]
)

# ===============================
# CONTENIDO
# ===============================

if menu == "🏁 Introducción":
    st.image("app/assets/logo.png", width=120)
    st.markdown(
    """
    ## Introducción al problema

    En mensajería y logística, la **prestación del servicio** y su **facturación** no siempre ocurren en el mismo mes.  
    Esto genera dos series de tiempo distintas, pero estrechamente relacionadas:

    ### 1) Serie de Producción del Servicio (Ventas generadas)
    - Representa el valor total de los servicios **prestados** en cada mes.  
    - Corresponde al momento en que el servicio se ejecuta para el cliente, **sin importar** cuándo se facture.

    ### 2) Serie de Facturación (Ventas facturadas)
    - Representa el valor total **facturado** al cliente en cada mes.  
    - Depende de las condiciones comerciales pactadas (**30, 60 o 90 días**).

    > **Nota:** Este desfase es normal en la operación, pero introduce complejidad en el análisis financiero, el control de ingresos y la detección de anomalías.

    ---

    ## Ejemplo base

    **Producción del servicio**
    - **Octubre:** 10 M  
    - **Noviembre:** 11 M  
    - **Diciembre:** 22 M  

    **Facturación**
    - **Noviembre:** 10 M (corresponde a octubre)  
    - **Diciembre:** 8 M (facturación parcial de noviembre)  
    - **Enero:** 24 M (saldo restante de noviembre + diciembre)  

    En este escenario, la facturación **se desplaza en el tiempo** y puede acumular valores de meses anteriores.

    ---

    ## Escenarios típicos de desfase

    - **Facturación a 30 días**  
    Se liquida mensualmente. Lo facturado en el mes *t* corresponde a lo prestado en *t-1*.

    - **Facturación a 60 días**  
    Lo prestado en el mes *t* se factura en *t+2*.

    - **Facturación a 90 días**  
    La facturación ocurre hasta tres meses después, generando acumulaciones y picos.

    ---

    ## Importancia para el análisis

    - Comparar producción y facturación **sin considerar el desfase** puede llevar a conclusiones incorrectas.  
    - Picos/caídas en facturación no siempre reflejan cambios reales en la operación.  
    - La detección de anomalías debe incorporar explícitamente el **desfase temporal**.

    Por esta razón, este proyecto analiza ambas series de manera conjunta, evaluando sus desfases como parte central del modelo analítico.
    """
    )
    
    

    # =========================
    # Datos simulados (6 meses)
    # =========================
    meses = ["Oct", "Nov", "Dic", "Ene", "Feb", "Mar"]
    produccion = [10, 11, 22, 15, 18, 20]  # millones

    df_demo = pd.DataFrame({
        "Mes": meses,
        "Produccion": produccion
    })

    # ==========================================
    # Símbolos por cohorte (mes de PRODUCCIÓN)
    # ==========================================
    symbol_map = {
        "Oct": "diamond",
        "Nov": "square",
        "Dic": "triangle-up",
        "Ene": "circle",
        "Feb": "x",
        "Mar": "star"
    }

    # ==========================================
    # Función: construir facturación desplazada
    # ==========================================
    def construir_facturacion(df: pd.DataFrame, shift: int):
        """
        shift:
        1 = 30 días (t -> t+1)
        2 = 60 días (t -> t+2)
        3 = 90 días (t -> t+3)

        Retorna:
        - df_fact con columnas: Mes, Facturacion, Symbol
            (solo incluye los meses donde hay facturación)
        """
        rows = []
        for i, row in df.iterrows():
            j = i + shift
            if j < len(df):
                mes_fact = df.loc[j, "Mes"]
                rows.append({
                    "Mes": mes_fact,
                    "Facturacion": row["Produccion"],
                    "Symbol": symbol_map[row["Mes"]]  # símbolo del mes de origen
                })

        return pd.DataFrame(rows)

    # ==========================================
    # UI: selector 30/60/90 días
    # ==========================================
    st.subheader("Desfase entre Producción y Facturación")

    opcion = st.radio(
        "Selecciona condición de facturación",
        options=["30 días", "60 días", "90 días"],
        horizontal=True
    )

    shift_map = {"30 días": 1, "60 días": 2, "90 días": 3}
    shift = shift_map[opcion]

    # ==========================================
    # Construir facturación para escenario elegido
    # ==========================================
    df_fact = construir_facturacion(df_demo, shift=shift)

    # ==========================================
    # Figura Plotly (una sola gráfica)
    # ==========================================
    fig = go.Figure()

    # --- Producción (verde) con línea ---
    fig.add_trace(
        go.Scatter(
            x=df_demo["Mes"],
            y=df_demo["Produccion"],
            mode="lines+markers",
            name="Producción del servicio",
            line=dict(color="green", width=3),
            marker=dict(
                size=12,
                symbol=[symbol_map[m] for m in df_demo["Mes"]],
                color="green"
            ),
            hovertemplate="<b>Producción</b><br>Mes=%{x}<br>Valor=%{y}M<extra></extra>"
        )
    )

    # --- Facturación (naranja) con línea (solo donde existe) ---
    fig.add_trace(
        go.Scatter(
            x=df_fact["Mes"],
            y=df_fact["Facturacion"],
            mode="lines+markers",
            name=f"Facturación ({opcion})",
            line=dict(color="orange", width=3, dash="dash"),
            marker=dict(
                size=12,
                symbol=df_fact["Symbol"].tolist(),
                color="orange"
            ),
            hovertemplate="<b>Facturación</b><br>Mes=%{x}<br>Valor=%{y}M<extra></extra>"
        )
    )

    # Layout
    fig.update_layout(
        title=(
            f"Producción vs Facturación — {opcion}<br>"
            "<sup>El símbolo identifica el mes de origen del servicio (cohorte)</sup>"
        ),
        xaxis_title="Mes",
        yaxis_title="Millones de pesos",
        legend_title="Serie",
        template="plotly_white",
        margin=dict(l=40, r=40, t=80, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Lectura: La línea verde muestra la producción real. "
        "La línea naranja muestra cuándo se factura esa producción según el desfase. "
        "El símbolo se repite para rastrear el mes de origen."
    )

    st.markdown("# Metodología:")

    # =========================
    # 1. Definición del problema
    # =========================
    st.markdown("## 1. Definición del problema")
    st.markdown(
        "Sea $y_t$ la facturación mensual de un cliente en el periodo $t$, con $t=1,\\dots,T$. "
        "El objetivo es detectar anomalías en el valor de facturación del periodo $T+1$, combinando:"
    )
    st.markdown(
        "- un modelo de pronóstico univariado,\n"
        "- intervalos de predicción basados en residuales (cuantiles),\n"
        "- y un esquema de simulación Monte Carlo para generar escenarios plausibles del valor observado."
    )
    st.markdown(
        "Una observación se considera anómala si no es consistente con el rango de comportamiento esperado definido por el modelo."
    )
    st.markdown("---")

    # =========================
    # 2. Serie univariada de análisis
    # =========================
    st.markdown("## 2. Serie univariada de análisis")
    st.markdown("La serie mensual se construye como:")
    st.latex(r"y_t = \sum_{i \in \mathcal{I}_t} v_i")
    st.markdown(
        "donde $\\mathcal{I}_t$ es el conjunto de operaciones asociadas al mes $t$ y $v_i$ corresponde al valor monetario de la operación. "
        "La serie es mensual y no negativa: $y_t \\ge 0$."
    )
    st.markdown("---")

    # =========================
    # 2.1 Segmentación del cliente (frecuente vs intermitente)
    # =========================
    st.markdown("## 2.1 Segmentación del cliente (frecuente vs intermitente)")
    st.markdown(
        "En presencia de intermitencia (meses con $y_t=0$), se define una segmentación para estabilizar el entrenamiento del modelo."
    )
    st.markdown("Se calcula la proporción de meses en cero:")
    st.latex(r"\rho_0 = \frac{1}{T}\sum_{t=1}^{T} \mathbb{1}[y_t = 0]")
    st.markdown(
        "Si $\\rho_0$ excede un umbral (por ejemplo, $0.30$), el cliente se trata como **intermitente** y "
        "se entrena el modelo sobre el **último bloque activo reciente** (subserie) para evitar mezclar regímenes apagados con actividad actual."
    )
    st.markdown("---")

    # =========================
    # 3. Modelo de pronóstico y residuales
    # =========================
    st.markdown("## 3. Modelo de pronóstico y residuales")
    st.markdown("Se define un modelo de pronóstico univariado que produce una estimación:")
    st.latex(r"\hat{y}_t = f(y_{1}, \dots, y_{t-1})")
    st.markdown("El residual asociado se define como:")
    st.latex(r"e_t = y_t - \hat{y}_t")
    st.markdown(
        "Para obtener una estimación robusta del error, el modelo se evalúa con validación temporal (rolling origin / walk-forward), "
        "generando residuales en un conjunto de entrenamiento:"
    )
    st.latex(r"\mathcal{E} = \{ e_t \}_{t=t_0}^{T}")
    st.markdown("---")

    # =========================
    # 3.1 Residuales en meses activos (robustez ante ceros)
    # =========================
    st.markdown("## 3.1 Residuales en meses activos (robustez ante ceros)")
    st.markdown(
        "Dado que la serie puede contener meses en cero, los cuantiles de error se estiman preferiblemente sobre meses **activos**:"
    )
    st.latex(r"\mathcal{E}^{+} = \{ e_t \in \mathcal{E} \;:\; y_t > 0 \}")
    st.markdown(
        "Si $|\\mathcal{E}^{+}|$ es suficientemente grande (por ejemplo, al menos 8 observaciones), "
        "se usan los cuantiles de $\\mathcal{E}^{+}$. En caso contrario, se utiliza el conjunto completo $\\mathcal{E}$ (fallback)."
    )
    st.markdown("---")

    # =========================
    # 4. Intervalos de predicción basados en cuantiles + no negatividad
    # =========================
    st.markdown("## 4. Intervalos de predicción basados en cuantiles")
    st.markdown("A partir de la distribución empírica de residuales, se definen intervalos no paramétricos:")
    st.latex(r"LL_t = \hat{y}_t + Q_{\alpha}(\mathcal{E}^\star), \qquad UL_t = \hat{y}_t + Q_{1-\alpha}(\mathcal{E}^\star)")
    st.markdown(
        "donde $\\mathcal{E}^\\star$ denota $\\mathcal{E}^{+}$ si hay suficiente información, o $\\mathcal{E}$ en caso contrario."
    )
    st.markdown(
        "Adicionalmente, se impone la restricción de no negatividad en las bandas:"
    )
    st.latex(r"LL_t \leftarrow \max(0, LL_t), \qquad UL_t \leftarrow \max(0, UL_t)")
    st.markdown("Para el periodo $T+1$ se obtiene el intervalo $[LL_{T+1}, UL_{T+1}]$.")
    st.markdown("---")

    # =========================
    # 5. Simulación Monte Carlo del valor en T+1 (a nivel operación)
    # =========================
    st.markdown("## 5. Simulación Monte Carlo del valor en $T+1$")

    # 5.1 Bootstrap por operación
    st.markdown("### 5.1 Bootstrap por operación (tamaños de transacción)")
    st.markdown(
        "Se construye un conjunto de valores por operación a partir de una ventana histórica de los últimos $K$ meses del cliente, "
        "por ejemplo a nivel de documento/operación. Sea $V = \\{v_1,\\dots,v_M\\}$ el pool de valores observados."
    )
    st.markdown(
        "Para simular el valor del mes $T+1$, primero se simula el número de operaciones $N_{T+1}$ mediante remuestreo (bootstrap) "
        "de los conteos históricos por mes:"
    )
    st.latex(r"N_{T+1} \sim \mathrm{Sample}(\{N_{T-K+1},\dots,N_T\})")

    # 5.2 Bernoulli factura/no factura
    st.markdown("### 5.2 Componente estocástico de facturación (Bernoulli)")
    st.markdown(
        "Dado un parámetro $p \\in [0,1]$ (probabilidad de que una operación sea facturada), se simula por operación:"
    )
    st.latex(r"X_j \sim \mathrm{Bernoulli}(p), \quad j=1,\dots,N_{T+1}")
    st.markdown(
        "Se remuestrea un valor por operación $v_j \\sim \\mathrm{Sample}(V)$ y se construye la facturación simulada del mes $T+1$:"
    )
    st.latex(r"y^{sim}_{T+1} = \sum_{j=1}^{N_{T+1}} X_j\, v_j")
    st.markdown(
        "Este procedimiento define una distribución empírica para $y^{sim}_{T+1}$ bajo el escenario de facturación parcial controlado por $p$."
    )
    st.markdown("---")

    # =========================
    # 6. Detección de anomalías
    # =========================
    st.markdown("## 6. Detección de anomalías")
    st.markdown("Sea $y^{sim}_{T+1}$ un valor generado por simulación. La regla de detección se define como:")
    st.latex(r"""
    A_{T+1}(y^{sim}) =
    \begin{cases}
    1, & y^{sim}_{T+1} < LL_{T+1} \ \text{o}\ y^{sim}_{T+1} > UL_{T+1} \\
    0, & \text{en otro caso}
    \end{cases}
    """)
    st.markdown("---")

    # =========================
    # 7. Estimación de riesgo de anomalía
    # =========================
    st.markdown("## 7. Estimación de riesgo de anomalía")
    st.markdown(
        "Al ejecutar $S$ simulaciones Monte Carlo, se estima la probabilidad de anomalía (baja/alta) como:"
    )
    st.latex(r"\mathbb{P}(\text{anomalía}) \approx \frac{1}{S}\sum_{s=1}^{S} A_{T+1}(y^{sim,(s)})")
    st.markdown(
        "De forma análoga se puede estimar $\\mathbb{P}(y^{sim}_{T+1} < LL_{T+1})$ y $\\mathbb{P}(y^{sim}_{T+1} > UL_{T+1})$."
    )
    st.markdown("---")

    # =========================
    # 8. Resultados del sistema
    # =========================
    st.markdown("## 8. Resultados del sistema")
    st.markdown(
        "- Pronóstico puntual $\\hat{y}_{T+1}$\n"
        "- Intervalo de predicción $[LL_{T+1}, UL_{T+1}]$ con no negatividad\n"
        "- Distribución simulada de $y^{sim}_{T+1}$ (Monte Carlo)\n"
        "- Indicador binario de anomalía\n"
        "- Probabilidades estimadas: $\\mathbb{P}(y^{sim}_{T+1} < LL_{T+1})$, $\\mathbb{P}(y^{sim}_{T+1} > UL_{T+1})$"
    )



elif menu == "🧹 ETL (Preparación de Datos)":

    st.markdown("""
        ## Preparación de los datos (ETL)

        En esta etapa se realiza un **ETL mínimo** para estructurar la fecha de prestación del servicio,
        la cual se encuentra embebida en texto libre dentro de la columna **OBSERVACION**.

        El proceso incluye dos filtros:
        1. **ETL por OBSERVACION**: se excluyen registros donde no se puede inferir el mes del servicio.
        2. **Consistencia temporal**: se excluyen registros con `diff_factura_servicio < 0` (facturación antes del servicio).

        El objetivo final es dejar un dataset consistente para construir y comparar series de tiempo.
    """)

    st.title("🧹 ETL – Preparación mínima de los datos")

    # ============================
    # 0) Cargar datos
    # ============================
    df = load_data()

    required_cols = ["AÑO DOCUMENTO", "MES", "OBSERVACION"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Faltan columnas requeridas para el ETL: {missing}")
        st.stop()

    st.subheader("Vista inicial")
    st.dataframe(df.head(5), use_container_width=True)

    # ============================
    # 1) Diccionarios de meses
    # ============================
    meses_full = {
        "ENERO": 1, "FEBRERO": 2, "MARZO": 3, "ABRIL": 4, "MAYO": 5, "JUNIO": 6,
        "JULIO": 7, "AGOSTO": 8, "SEPTIEMBRE": 9, "OCTUBRE": 10,
        "NOVIEMBRE": 11, "DICIEMBRE": 12,
    }
    meses_abrev = {
        "ENE": 1, "FEB": 2, "MAR": 3, "ABR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AGO": 8, "SEP": 9, "SET": 9,
        "OCT": 10, "NOV": 11, "DIC": 12,
    }
    MES_KEYS = list(meses_full.keys()) + list(meses_abrev.keys())

    patron_anio = r"\b(20\d{2})\b"

    patron_rango_full = (
        r"\b(ENERO|FEBRERO|MARZO|ABRIL|MAYO|JUNIO|JULIO|AGOSTO|SEPTIEMBRE|OCTUBRE|NOVIEMBRE|DICIEMBRE)"
        r"\s*[-/]\s*"
        r"(ENERO|FEBRERO|MARZO|ABRIL|MAYO|JUNIO|JULIO|AGOSTO|SEPTIEMBRE|OCTUBRE|NOVIEMBRE|DICIEMBRE)"
        r"\s*(20\d{2})\b"
    )

    patron_rango_abrev = (
        r"\b(ENE|FEB|MAR|ABR|MAY|JUN|JUL|AGO|SEP|SET|OCT|NOV|DIC)"
        r"\s*[-/]\s*"
        r"(ENE|FEB|MAR|ABR|MAY|JUN|JUL|AGO|SEP|SET|OCT|NOV|DIC)"
        r"\s*(20\d{2})\b"
    )

    patron_mes_full = r"\b(" + "|".join(meses_full.keys()) + r")\b"
    patron_mes_abrev = r"\b(" + "|".join(meses_abrev.keys()) + r")\b"

    def extraer_mes_anio(obs, threshold_fuzzy=85):
        if pd.isna(obs):
            return pd.Series([np.nan, np.nan])

        texto = str(obs).upper()

        # Guard: solo si hay contexto real de servicio/mes
        if not any(k in texto for k in [" MES ", "DEL MES", "DURANTE", "PRESTADO", "PRESTADOS", "SERVICIO", "PERIODO", "CORRESPONDIENTE"]):
            return pd.Series([np.nan, np.nan])

        # Rango FULL: DICIEMBRE-ENERO 2024 -> toma mes final + año
        m = re.search(patron_rango_full, texto)
        if m:
            _, mes_fin, anio = m.groups()
            return pd.Series([meses_full[mes_fin], int(anio)])

        # Rango ABREV: DIC-ENE 2024
        m = re.search(patron_rango_abrev, texto)
        if m:
            _, mes_fin, anio = m.groups()
            return pd.Series([meses_abrev[mes_fin], int(anio)])

        # Mes FULL exacto
        m = re.search(patron_mes_full, texto)
        if m:
            mes = meses_full[m.group(1)]
            anios = re.findall(patron_anio, texto)
            anio = int(anios[-1]) if anios else np.nan  # último año para evitar contrato 2020
            return pd.Series([mes, anio])

        # Mes ABREV exacto
        m = re.search(patron_mes_abrev, texto)
        if m:
            mes = meses_abrev[m.group(1)]
            anios = re.findall(patron_anio, texto)
            anio = int(anios[-1]) if anios else np.nan
            return pd.Series([mes, anio])

        # Fuzzy (último recurso, tolera typos)
        tokens = re.findall(r"[A-ZÁÉÍÓÚÑ]{4,}", texto)  # evita tokens cortos tipo "EM"
        mejor_score, mes_num = 0, np.nan

        for tok in tokens:
            match = process.extractOne(tok, MES_KEYS, scorer=fuzz.partial_ratio)
            if match:
                key, score, _ = match
                if score >= threshold_fuzzy and score > mejor_score:
                    mejor_score = score
                    mes_num = meses_full.get(key, meses_abrev.get(key))

        anios = re.findall(patron_anio, texto)
        anio = int(anios[-1]) if anios else np.nan
        return pd.Series([mes_num, anio])

    def inferir_anio_servicio(row):
        if not pd.isna(row["ANIO_SERVICIO"]):
            return int(row["ANIO_SERVICIO"])

        # Enero suele facturar servicios del año anterior
        if int(row["MES"]) == 1:
            return int(row["AÑO DOCUMENTO"]) - 1

        return int(row["AÑO DOCUMENTO"])

    # ============================
    # 2) Paso 1: ETL por OBSERVACION
    # ============================
    total_inicial = len(df)

    df[["MES_SERVICIO", "ANIO_SERVICIO"]] = df["OBSERVACION"].apply(extraer_mes_anio)

    mask_etl_ok = df["MES_SERVICIO"].notna()
    df.loc[mask_etl_ok, "ANIO_SERVICIO"] = df.loc[mask_etl_ok].apply(inferir_anio_servicio, axis=1)

    df_etl = df[mask_etl_ok].copy()

    # Construir fechas para diff
    df_etl["fecha_servicio"] = pd.to_datetime(
        dict(year=df_etl["ANIO_SERVICIO"], month=df_etl["MES_SERVICIO"], day=1),
        errors="coerce"
    ).dt.to_period("M").astype(str)

    df_etl["fecha_facturacion"] = pd.to_datetime(
        dict(year=df_etl["AÑO DOCUMENTO"], month=df_etl["MES"], day=1),
        errors="coerce"
    ).dt.to_period("M").astype(str)

    df_etl["diff_factura_servicio"] = (
        pd.to_datetime(df_etl["fecha_facturacion"]).dt.to_period("M").astype(int)
        - pd.to_datetime(df_etl["fecha_servicio"]).dt.to_period("M").astype(int)
    )

    # ============================
    # 3) Paso 2: Filtrar diffs inconsistentes
    #    Se conservan solo desfases entre 0 y 12 meses
    # ============================
    mask_diff_ok = (
        (df_etl["diff_factura_servicio"] >= 0) &
        (df_etl["diff_factura_servicio"] < 12)
    )

    df_final = df_etl[mask_diff_ok].copy()


    # ============================
    # 4) Métricas y porcentajes
    # ============================
    excluidos_etl = total_inicial - len(df_etl)
    excluidos_diff = len(df_etl) - len(df_final)
    excluidos_total = total_inicial - len(df_final)

    st.subheader("Resumen del filtrado (2 pasos)")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total inicial", f"{total_inicial:,}")
    c2.metric("Tras ETL OBSERVACION", f"{len(df_etl):,}")
    c3.metric("Dataset final (diff >= 0)", f"{len(df_final):,}")

    st.markdown("### Detalle de exclusiones")
    colA, colB, colC = st.columns(3)

    colA.metric(
        "Excluidos por ETL (sin mes)",
        f"{excluidos_etl:,}",
        f"{(excluidos_etl / total_inicial):.1%}"
    )
    colB.metric(
        "Excluidos por diff inválido (<0 o >12)",
        f"{excluidos_diff:,}",
        f"{(excluidos_diff / total_inicial):.1%} del total inicial"
    )
    colC.metric(
        "Exclusión total (ambos pasos)",
        f"{excluidos_total:,}",
        f"{(excluidos_total / total_inicial):.1%}"
    )

    st.success(
        f"Dataset listo para análisis: {len(df_final):,} registros "
        f"({(len(df_final) / total_inicial):.1%} del total inicial)."
    )

    st.dataframe(df_final, use_container_width=True)

    

    # ============================
    # Guardar CSV final 
    # ============================
    output_path = "app/data/data_interservice_etl_final.csv"

    try:
        df_final.to_csv(output_path, index=False, encoding="utf-8-sig")
        st.success(f"✅ ETL final guardado en: {output_path}")
    except Exception as e:
        st.error(f"❌ No fue posible guardar el CSV del ETL: {e}")
        st.stop()

    st.markdown("""
                ### Conclusiones

                - **Paso 1 (ETL OBSERVACION):** se excluyeron registros donde no fue posible inferir
                de forma confiable el mes de prestación del servicio.
                - **Paso 2 (Consistencia temporal):** se excluyeron registros con desfases
                negativos (facturación antes del servicio) o excesivos (>12 meses),
                al no corresponder a condiciones comerciales normales.
                - El dataset final es consistente para análisis de series de tiempo,
                detección de anomalías y modelado de desfases de facturación.
                """)





elif menu == "📊 EDA":

    st.title("📊 Análisis Exploratorio de Datos (EDA)")
    st.markdown("""
    Este análisis exploratorio se enfoca en **caracterizar el comportamiento temporal**
    entre la **prestación del servicio** y su **facturación**, identificando patrones normales
    de desfase y posibles anomalías operativas o financieras.
    """)

    # ============================
    # Dataset
    # ============================

    try:
        df = load_etl_final()
        st.success("Dataset post-ETL cargado correctamente ✅")
    except FileNotFoundError:
        st.error("⚠️ No existe el archivo post-ETL. Ejecuta primero el módulo 🧹 ETL.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error cargando el dataset post-ETL: {e}")
        st.stop()


    # ============================
    # Vista previa
    # ============================
    st.subheader("Vista previa de los datos analizados")
    st.dataframe( df.head(20) )   
    total_valor_neto = df["VALOR NETO"].sum() / 1_000_000

    # ============================
    # KPIs Financieros
    # ============================

    # Asegurar numéricos (hazlo una vez, antes de KPIs y gráficas)
    df["VALOR NETO"] = pd.to_numeric(df["VALOR NETO"], errors="coerce")
    df["diff_factura_servicio"] = pd.to_numeric(df["diff_factura_servicio"], errors="coerce")

    # Total (sin filtros)
    total_valor_neto = df["VALOR NETO"].sum()
    total_valor_neto_mm = total_valor_neto / 1_000_000

    # A partir del mes 3 (>= 3)
    df_mes3 = df[df["diff_factura_servicio"] >= 3]
    valor_neto_mes3 = df_mes3["VALOR NETO"].sum()
    valor_neto_mes3_mm = valor_neto_mes3 / 1_000_000

    # Porcentaje del total (evitar división por cero)
    pct_mes3 = (valor_neto_mes3 / total_valor_neto * 100) if total_valor_neto else 0

    st.subheader("📊 KPIs Financieros")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            label="💰 Valor Neto Total (MM)",
            value=f"${total_valor_neto_mm:,.0f}"
        )

    with c2:
        st.metric(
            label="⏱️ Valor Neto desde mes 3 (MM)",
            value=f"${valor_neto_mes3_mm:,.0f}"
        )

    with c3:
        st.metric(
            label="% del total (desde mes 3)",
            value=f"{pct_mes3:.1f}%"
        )


    # ============================
    # Distribución del desfase + valor facturado
    # ============================
    st.subheader("Distribución del desfase producción → facturación")

    # Asegurar tipos
    df["diff_factura_servicio"] = pd.to_numeric(df["diff_factura_servicio"], errors="coerce")
    df["VAL TOTAL"] = pd.to_numeric(df["VAL TOTAL"], errors="coerce")

    df_plot = (
        df
        .dropna(subset=["diff_factura_servicio", "VAL TOTAL"])
        .groupby("diff_factura_servicio", as_index=False)
        .agg(
            cantidad=("diff_factura_servicio", "count"),
            valor_total=("VAL TOTAL", "sum"),
        )
        .sort_values("diff_factura_servicio")
    )

    # Escalar a miles de millones para lectura
    df_plot["valor_total_mm"] = df_plot["valor_total"] / 1_000_000

    fig = go.Figure()

    # Barras: cantidad de registros
    fig.add_trace(
        go.Bar(
            x=df_plot["diff_factura_servicio"],
            y=df_plot["cantidad"],
            name="Cantidad de registros",
            marker_color="#1f77b4",
            yaxis="y",
        )
    )

    # Línea: valor total facturado
    fig.add_trace(
        go.Scatter(
            x=df_plot["diff_factura_servicio"],
            y=df_plot["valor_total_mm"],
            name="Valor total facturado (MM)",
            mode="lines+markers+text",
            text=df_plot["valor_total_mm"].round(2),
            textposition="top center",
            marker=dict(color="#ff7f0e", size=8),
            line=dict(width=2),
            yaxis="y2",
        )
    )

    fig.update_layout(
        title="Distribución del desfase y valor facturado",
        xaxis=dict(
            title="Meses de desfase",
            dtick=1
        ),
        yaxis=dict(
            title="Cantidad de registros",
            showgrid=False
        ),
        yaxis2=dict(
            title="Valor total facturado (miles de millones)",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        legend=dict(
            orientation="h",   # horizontal
            x=0.5,
            y=-0.25,
            xanchor="center"
        ),
        margin=dict(b=120),   # espacio para la leyenda abajo
        bargap=0.25,
    )

    st.plotly_chart(fig, use_container_width=True)
    # ============================
    # Desfase vs Valor
    # ============================
    st.subheader("Relación entre desfase y valor facturado")

    fig_scatter = px.scatter(
        df,
        x="diff_factura_servicio",
        y="VAL TOTAL",
        size="VAL TOTAL",
        color="diff_factura_servicio",
        title="Desfase vs Valor Total Facturado",
        labels={
            "diff_factura_servicio": "Meses de desfase",
            "VAL TOTAL": "Valor total",
        },
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ============================
    # Análisis por cliente (interactivo)
    # ============================
    st.subheader("Análisis por cliente")

    # Asegurar tipos (hazlo una sola vez en todo el EDA)
    df["VALOR NETO"] = pd.to_numeric(df["VALOR NETO"], errors="coerce")
    df["VAL TOTAL"] = pd.to_numeric(df["VAL TOTAL"], errors="coerce")
    df["diff_factura_servicio"] = pd.to_numeric(df["diff_factura_servicio"], errors="coerce")

    # Selector de cliente
    clientes = (
        df["ID_NOMBRE_CLIENTE"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    cliente_sel = st.selectbox("Selecciona un cliente", clientes)

    df_c = df[df["ID_NOMBRE_CLIENTE"].astype(str) == str(cliente_sel)].copy()

    # ============================
    # KPIs del cliente seleccionado
    # ============================
    st.markdown("### 📊 KPIs financieros del cliente")

    total_cliente = df_c["VALOR NETO"].sum()
    total_cliente_mm = total_cliente / 1_000_000

    valor_mes3_cliente = df_c.loc[df_c["diff_factura_servicio"] >= 3, "VALOR NETO"].sum()
    valor_mes3_cliente_mm = valor_mes3_cliente / 1_000_000

    pct_mes3_cliente = (valor_mes3_cliente / total_cliente * 100) if total_cliente else 0

    ops_cliente = df_c["VALOR NETO"].notna().sum()

    # ============================
    # Gap final (Producción acum - Facturación acum) desde df_c
    # ============================

    df_c["fecha_servicio"] = pd.to_datetime(df_c["fecha_servicio"], errors="coerce")
    df_c["fecha_facturacion"] = pd.to_datetime(df_c["fecha_facturacion"], errors="coerce")
    df_c["VAL TOTAL"] = pd.to_numeric(df_c["VAL TOTAL"], errors="coerce")

    prod_total = (
        df_c.dropna(subset=["fecha_servicio", "VAL TOTAL"])["VAL TOTAL"]
        .sum()
    )

    fact_total = (
        df_c.dropna(subset=["fecha_facturacion", "VAL TOTAL"])["VAL TOTAL"]
        .sum()
    )

    gap_final = prod_total - fact_total
    gap_final_mm = gap_final / 1_000_000


    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric("💰 Valor Neto Total (MM)", f"${total_cliente_mm:,.0f}")
    with c2:
        st.metric("⏱️ Valor Neto desde mes 3 (MM)", f"${valor_mes3_cliente_mm:,.0f}")
    with c3:
        st.metric("% del total (desde mes 3)", f"{pct_mes3_cliente:.1f}%")
    with c4:
        st.metric("📦 Operaciones", f"{ops_cliente:,.0f}")
    with c5:
        st.metric("📉 Gap final Prod − Fact (MM)", f"${gap_final_mm:,.0f}")



    # ============================
    # Serie de tiempo comparativa (por cliente)
    # ============================
    st.subheader("Serie de tiempo: Producción vs Facturación (cliente)")

    # Asegurar fechas
    df_c["fecha_servicio"] = pd.to_datetime(df_c["fecha_servicio"], errors="coerce")
    df_c["fecha_facturacion"] = pd.to_datetime(df_c["fecha_facturacion"], errors="coerce")

    prod = (
        df_c.dropna(subset=["fecha_servicio"])
        .groupby("fecha_servicio", as_index=False)["VAL TOTAL"]
        .sum()
        .rename(columns={"fecha_servicio": "fecha", "VAL TOTAL": "Produccion"})
    )

    fact = (
        df_c.dropna(subset=["fecha_facturacion"])
        .groupby("fecha_facturacion", as_index=False)["VAL TOTAL"]
        .sum()
        .rename(columns={"fecha_facturacion": "fecha", "VAL TOTAL": "Facturacion"})
    )

    df_ts = prod.merge(fact, on="fecha", how="outer").fillna(0).sort_values("fecha")

    fig_ts = px.line(
        df_ts,
        x="fecha",
        y=["Produccion", "Facturacion"],
        title=f"Producción vs Facturación — {cliente_sel}",
        markers=True,
    )

    st.plotly_chart(fig_ts, use_container_width=True)

    # ============================
    # Serie de tiempo acumulada (por cliente)
    # ============================
    st.subheader("Acumulado: Producción vs Facturación (cliente)")

    df_ts_acc = df_ts.copy()

    df_ts_acc["Produccion_acum"] = df_ts_acc["Produccion"].cumsum()
    df_ts_acc["Facturacion_acum"] = df_ts_acc["Facturacion"].cumsum()

    fig_acc = px.line(
        df_ts_acc,
        x="fecha",
        y=["Produccion_acum", "Facturacion_acum"],
        title=f"Acumulado Producción vs Facturación — {cliente_sel}",
        markers=True,
    )

    st.plotly_chart(fig_acc, use_container_width=True)  


elif menu ==  "📈 Pronósticos + 🚨 Anomalías":    

    st.title("📈 Pronósticos de Series de Tiempo (por cliente)")
    st.write("""
    Este módulo evalúa múltiples modelos clásicos de pronóstico y selecciona automáticamente
    el que minimiza el **Error Cuadrático Medio (MSE)** mediante **backtesting walk-forward**.
    """)

    # =========================
    # Cargar dataset
    # =========================
    try:
        df = load_etl_final()
    except FileNotFoundError:
        st.error("⚠️ No existe el archivo post-ETL. Ejecuta primero el módulo 🧹 ETL.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error cargando el dataset post-ETL: {e}")
        st.stop()

    # =========================
    # Constantes fijas (según tu requerimiento)
    # =========================
    date_col = "fecha_facturacion"
    metric_col = "VAL TOTAL"

    if "ID_NOMBRE_CLIENTE" not in df.columns:
        st.error("No existe la columna 'ID_NOMBRE_CLIENTE' en el dataset.")
        st.stop()

    clientes = (
        df["ID_NOMBRE_CLIENTE"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    if not clientes:
        st.warning("No hay clientes disponibles en la base.")
        st.stop()

    # =========================
    # UI: 2 columnas (cliente / horizonte)
    # =========================
    c1, c2 = st.columns(2)
    with c1:
        cliente_sel = st.selectbox("Selecciona un cliente", clientes)
    with c2:
        horizon = st.slider("Horizonte (meses)", 1, 6, 1, 1)

    st.caption(
        "📌 Los pronósticos se realizan **exclusivamente** sobre la facturación mensual "
        "(VAL TOTAL agregado por fecha_facturacion)."
    )
    st.divider()

    # =========================
    # Filtrar cliente y serie mensual
    # =========================
    df_c = df.loc[df["ID_NOMBRE_CLIENTE"].astype(str) == str(cliente_sel)].copy()

    y = to_monthly_series(
        df_c,
        date_col=date_col,
        value_col=metric_col,
        fill_missing="zero",
    )

    if y.empty:
        st.warning("Este cliente no tiene datos válidos para construir la serie.")
        st.stop()

    # Usar máximo disponible (sin min_months)
    # initial_train razonable: mínimo 12 o 70% de la serie (lo que sea menor, pero >=12 si alcanza)
    if len(y) < 8:
        st.warning(f"Serie demasiado corta ({len(y)} meses). Se recomienda mínimo 12 meses para selección robusta.")
        st.dataframe(y.rename("y").reset_index().rename(columns={"index": "mes"}), use_container_width=True)
        st.stop()

    initial_train = 12 if len(y) >= 13 else max(6, int(len(y) * 0.6))

    # =========================
    # Preparar DF histórico
    # =========================
    df_y = y.rename("y").reset_index().rename(columns={y.index.name or "index": "mes"})
    df_y["mes"] = pd.to_datetime(df_y["mes"])



    # =========================
    # Evaluación modelos + estacionalidad auto
    # =========================
    st.subheader("Selección automática de modelo (MSE)")

    with st.spinner("Evaluando modelos (walk-forward)..."):
        ranking, seasonality_info = evaluate_models(
            y,
            initial_train=initial_train,
            h=1,  # backtesting más estable 1-step (recomendado)
            ma_window=3,
            arima_order=(1, 1, 1),
            sarima_order=(1, 1, 1),
            seasonal_candidates=(12, 6),
            seasonal_threshold=0.30,
        )

    st.info(
        f"Estacionalidad detectada: **{seasonality_info.get('is_seasonal')}** | "
        f"Periodo: **{seasonality_info.get('seasonal_period')}** | "
        f"Fuerza (ACF): **{seasonality_info.get('strength'):.3f}**"
    )

    # Mejor modelo
    best_model = ranking.iloc[0]["model"]
    best_mse = ranking.iloc[0]["mse"]
    st.success(f"✅ Mejor modelo: **{best_model}** | MSE (1-step): **{best_mse:,.2f}**")
    # =========================
    # Ajuste final y forecast (h pasos)
    # =========================
    with st.spinner("Ajustando mejor modelo y generando pronóstico..."):
        best_res = fit_best_and_forecast(
            y,
            ranking,
            seasonality_info=seasonality_info,
            h=horizon,
            ma_window=3,
            arima_order=(1, 1, 1),
            sarima_order=(1, 1, 1),
        )

    # =========================
    # Bandas (LL/UL) por cuantiles de residuales del mejor modelo
    # =========================
    bands = residual_quantile_bands(
        y=y,
        model_name=best_model,
        seasonality_info=seasonality_info,
        alpha=0.05,
        ma_window=3,
        arima_order=(1, 1, 1),
        sarima_order=(1, 1, 1),
    )

    q_low, q_high = bands["q_low"], bands["q_high"]

    df_fore = best_res.y_hat.rename("forecast").reset_index().rename(columns={"index": "mes"})
    df_fore["mes"] = pd.to_datetime(df_fore["mes"])
    df_fore["LL"] = (df_fore["forecast"] + q_low).clip(lower=0)
    df_fore["UL"] = (df_fore["forecast"] + q_high).clip(lower=0)


    # =========================
    # Gráfica: histórico + forecast + bandas
    # =========================
    fig = go.Figure()

    # Histórico
    fig.add_trace(go.Scatter(
        x=df_y["mes"], y=df_y["y"],
        mode="lines+markers",
        name="Histórico"
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=df_fore["mes"], y=df_fore["forecast"],
        mode="lines+markers",
        name=f"Forecast ({best_model})"
    ))

    # Banda UL
    fig.add_trace(go.Scatter(
        x=df_fore["mes"], y=df_fore["UL"],
        mode="lines",
        name="UL (P95 residual)",
        line=dict(dash="dash")
    ))

    # Banda LL
    fig.add_trace(go.Scatter(
        x=df_fore["mes"], y=df_fore["LL"],
        mode="lines",
        name="LL (P05 residual)",
        line=dict(dash="dash")
    ))

    fig.update_layout(
        title=f"Pronóstico con bandas — {cliente_sel} ({metric_col}) | horizonte={horizon} mes(es)",
        xaxis_title="Mes",
        yaxis_title="Valor",
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
        margin=dict(b=120),
    )

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # Resultados
    # =========================
   

    st.subheader("Ranking de modelos (MSE)")
    st.dataframe(ranking, use_container_width=True)

    st.subheader("Pronóstico y límites")
    st.dataframe(df_fore, use_container_width=True)


    # =========================
    # Módulo de Anomalías (Monte Carlo) - Mes T+1
    # =========================
    st.subheader("🚨 Simulación Monte Carlo para anomalías (mes T+1)")

    cA, cB, cC = st.columns(3)
    with cA:
        p_factura = st.slider("Probabilidad de facturar (p)", 0.0, 1.0, 0.85, 0.01)
    with cB:
        n_months_mc = st.slider("Ventana histórica (meses)", 3, 24, 12, 1)
    with cC:
        n_sims = st.slider("Simulaciones (S)", 200, 5000, 1000, 100)

    # --- detectar columna de operación/documento para contar transacciones
    possible_doc_cols = ["Numero documento", "NUMERO DOCUMENTO", "Numero_documento", "NUMERO_DOCUMENTO", "PEDIDO"]
    doc_col = next((c for c in possible_doc_cols if c in df_c.columns), None)

    if doc_col is None:
        st.warning(
            "No encontré columna para contar transacciones (ej: 'Numero documento' o 'PEDIDO'). "
            "Se simulará usando 1 operación por mes (simplificado)."
        )

    # --- preparar df con fechas y valores a nivel transacción (lo que tengas)
    df_mc = df_c.copy()
    df_mc[date_col] = pd.to_datetime(df_mc[date_col], errors="coerce")
    df_mc[metric_col] = pd.to_numeric(df_mc[metric_col], errors="coerce")
    df_mc = df_mc.dropna(subset=[date_col, metric_col])

    # mes (primer día)
    df_mc["mes"] = df_mc[date_col].values.astype("datetime64[M]")
    df_mc["mes"] = pd.to_datetime(df_mc["mes"])

    # Ventana: últimos n meses del cliente (según data real)
    last_month = df_mc["mes"].max()
    if pd.isna(last_month):
        st.warning("No hay datos válidos para simular.")
    else:
        window_start = (last_month - pd.DateOffset(months=n_months_mc - 1)).to_period("M").to_timestamp()
        df_win = df_mc[df_mc["mes"] >= window_start].copy()

        # --- 1) distribución de conteo de operaciones por mes
        if doc_col:
            # contar operaciones (documentos) por mes
            ops_by_month = (
                df_win.groupby("mes")[doc_col]
                .nunique()
                .reindex(pd.date_range(window_start, last_month, freq="MS"), fill_value=0)
            )
        else:
            # fallback: 1 operación por mes
            ops_by_month = pd.Series(
                1,
                index=pd.date_range(window_start, last_month, freq="MS"),
                dtype=int
            )

        # --- 2) pool de valores por operación (bootstrap)
        # si hay columna doc, tomamos valor por documento (sum por doc) para simular "tamaño de operación"
        if doc_col:
            df_vals = (
                df_win.groupby(["mes", doc_col])[metric_col]
                .sum()
                .reset_index()
            )
            value_pool = df_vals[metric_col].values.astype(float)
        else:
            # fallback: usar valores fila a fila
            value_pool = df_win[metric_col].values.astype(float)

        if len(value_pool) < 10:
            st.warning("Muy pocos valores en la ventana para bootstrap. Aumenta la ventana histórica.")
        else:
            # LL/UL para el primer mes pronosticado
            ll_1 = float(df_fore.loc[df_fore.index[0], "LL"])
            ul_1 = float(df_fore.loc[df_fore.index[0], "UL"])
            f_1 = float(df_fore.loc[df_fore.index[0], "forecast"])

            # --- Monte Carlo
            rng = np.random.default_rng(42)

            ops_hist = ops_by_month.values.astype(int)
            # Para simular #ops del mes T+1: bootstrap de conteos históricos (no paramétrico)
            ops_sim = rng.choice(ops_hist, size=n_sims, replace=True)
            ops_sim = np.maximum(ops_sim, 0)

            y_sim = np.zeros(n_sims, dtype=float)

            for s in range(n_sims):
                m = int(ops_sim[s])
                if m == 0:
                    y_sim[s] = 0.0
                    continue

                # sample de tamaños de operación
                vals = rng.choice(value_pool, size=m, replace=True)

                # Bernoulli(p) por operación: factura o no
                x = rng.binomial(n=1, p=p_factura, size=m)

                y_sim[s] = float(np.sum(vals * x))

            # --- Probabilidades de anomalía
            p_baja = float(np.mean(y_sim < ll_1))
            p_alta = float(np.mean(y_sim > ul_1))
            p_in = float(np.mean((y_sim >= ll_1) & (y_sim <= ul_1)))

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Forecast T+1", f"${f_1:,.0f}")
            with k2:
                st.metric("P(anomalía baja) = P(y<LL)", f"{p_baja:.2%}")
            with k3:
                st.metric("P(anomalía alta) = P(y>UL)", f"{p_alta:.2%}")
            with k4:
                st.metric("P(dentro de bandas)", f"{p_in:.2%}")

            # --- Visual distribución vs bandas
            import plotly.graph_objects as go

            fig_mc = go.Figure()
            fig_mc.add_trace(go.Histogram(x=y_sim, nbinsx=40, name="y_sim (T+1)"))
            fig_mc.add_vline(x=ll_1, line_dash="dash", annotation_text="LL", annotation_position="top left")
            fig_mc.add_vline(x=ul_1, line_dash="dash", annotation_text="UL", annotation_position="top right")
            fig_mc.add_vline(x=f_1, line_dash="solid", annotation_text="Forecast", annotation_position="bottom right")
            fig_mc.update_layout(
                title="Distribución simulada de facturación (T+1) vs bandas",
                xaxis_title="Valor simulado",
                yaxis_title="Frecuencia",
                bargap=0.05,
            )
            st.plotly_chart(fig_mc, use_container_width=True)

            # Tabla resumen
            st.caption("Resumen de simulación (T+1)")
            st.write(pd.DataFrame({
                "LL": [ll_1],
                "UL": [ul_1],
                "forecast": [f_1],
                "p_factura": [p_factura],
                "ventana_meses": [n_months_mc],
                "simulaciones": [n_sims],
                "P_baja": [p_baja],
                "P_alta": [p_alta],
                "P_dentro": [p_in],
                "ops_prom_ventana": [float(np.mean(ops_hist))],
            }))

elif menu == "🤖 MCP + Bot":

    st.title("🤖 Chat de Prueba (Groq)")

    st.info("Bot para consulta del estado de Facturación.")

    # Inicializar LLM (SIN agentes, SIN pandas)
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=st.secrets["GROQ_API_KEY"],
        timeout=30
    )

    # Historial simple
    if "chat_test" not in st.session_state:
        st.session_state.chat_test = []

    for msg in st.session_state.chat_test:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Di algo (ej: Hola)")

    if user_input:
        st.session_state.chat_test.append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Respondiendo…"):
                try:
                    # 👇 LLAMADA DIRECTA AL MODELO
                    response = llm.invoke(user_input)
                    answer = response.content
                except Exception as e:
                    answer = f"⚠️ Error llamando al modelo: {e}"

                st.markdown(answer)

        st.session_state.chat_test.append(
            {"role": "assistant", "content": answer}
        )



    
    


# ===============================
# FOOTER
# ===============================

st.sidebar.divider()

st.sidebar.markdown(
    """
    <div style="text-align:center;font-size:12px;">
    © 2026 | Dani Ortiz <br>
    Maestría 
    </div>
    """,
    unsafe_allow_html=True
)
