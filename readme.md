# Interservice — Pronóstico y Detección de Anomalías en Facturación

Dashboard desarrollado en Streamlit para el análisis de facturación,
pronóstico de series de tiempo por cliente y detección de anomalías mediante
bandas basadas en residuales y simulación Monte Carlo.

---

## 1. Instalación y ejecución en local

### 1.1 Requisitos
- Python 3.10 o superior
- Git
- Sistema operativo: Windows, macOS o Linux

Nota:
El repositorio no versiona entornos virtuales ni datos sensibles.

---

### 1.2 Clonar el repositorio

git clone https://github.com/DannyCerort/interservice-forecasting-anomaly-detection.git

cd <TU_REPO>

---

### 1.3 Crear y activar entorno virtual (venv)

Windows (PowerShell):  
python -m venv .venv  
.\\.venv\\Scripts\\Activate.ps1  

macOS / Linux:  
python3 -m venv .venv  
source .venv/bin/activate  

---

### 1.4 Instalar dependencias

pip install -r requirements.txt

---

### 1.5 Datos

- Los datos deben ubicarse en la carpeta data/.
- Ejemplo: data_interservice_etl_final.csv.
- Los archivos de datos reales no se versionan en GitHub.

---

### 1.6 Ejecutar la aplicación

streamlit run app/main.py

---

## 2. Metodología del proyecto

### 2.1 Definición del problema

Sea y_t la facturación mensual de un cliente en el periodo t,
con t = 1, …, T.

El objetivo es detectar anomalías en el valor de facturación del periodo
T+1, combinando:

- Pronóstico univariado por cliente  
- Intervalos de predicción basados en residuales  
- Simulación Monte Carlo para cuantificar riesgo de anomalía  

Una observación se considera anómala si no es consistente con el rango
esperado [LL_{T+1}, UL_{T+1}].

---

### 2.2 Construcción de la serie univariada

La serie mensual se construye como:

y_t = Σ_{i ∈ I_t} v_i

donde:
- I_t es el conjunto de operaciones asociadas al mes t
- v_i es el valor monetario de cada operación

La serie es mensual y no negativa:

y_t ≥ 0

---

### 2.3 Intermitencia y segmentación del cliente

Se define la proporción de meses sin facturación:

ρ_0 = (1 / T) · Σ_{t=1}^{T} 𝟙[y_t = 0]

- Clientes con alta ρ_0 se consideran intermitentes
- Para estos casos, el modelo se ajusta sobre el último bloque activo
  para evitar mezclar regímenes inactivos con actividad reciente

---

### 2.4 Modelos de pronóstico evaluados

Los modelos se comparan mediante validación temporal walk-forward
utilizando el Error Cuadrático Medio (MSE).

Modelos incluidos:

1. Naive  
2. Seasonal Naive  
3. Moving Average  
4. Simple Exponential Smoothing (SES)  
5. Holt (tendencia)  
6. Holt-Winters (tendencia + estacionalidad)  
7. ARIMA / SARIMA  

El modelo con menor MSE es seleccionado automáticamente.

---

### 2.5 Detección automática de estacionalidad

La estacionalidad se detecta mediante una heurística basada en la
autocorrelación (ACF) para rezagos candidatos (por ejemplo 6 y 12 meses).

Si la fuerza estacional supera un umbral, se habilitan modelos estacionales.

---

### 2.6 Pronóstico y residuales

Un modelo de pronóstico produce:

ŷ_t = f(y_1, …, y_{t−1})

Los residuales se definen como:

e_t = y_t − ŷ_t

Estos se obtienen mediante validación temporal para construir una
distribución empírica robusta.

---

### 2.7 Bandas de predicción (LL / UL)

Las bandas se definen mediante cuantiles de los residuales:

LL_t = ŷ_t + Q_α(E*)  
UL_t = ŷ_t + Q_{1−α}(E*)

donde E* corresponde preferiblemente a residuales en meses activos
(y_t > 0), con fallback al conjunto completo si hay pocos datos.

Se impone la restricción de no negatividad:

LL_t = max(0, LL_t)  
UL_t = max(0, UL_t)

---

### 2.8 Simulación Monte Carlo (factura / no factura)

Dado que la base histórica contiene únicamente operaciones facturadas,
se introduce una probabilidad p ∈ [0,1] de que una operación sea facturada.

Para cada operación j:

X_j ~ Bernoulli(p)

La facturación simulada del periodo T+1 se calcula como:

y_sim_{T+1} = Σ_j X_j · v_j

Este proceso genera una distribución de escenarios plausibles de
facturación futura.

---

### 2.9 Detección de anomalías y riesgo

Una simulación se considera anómala si:

y_sim_{T+1} < LL_{T+1}  
o  
y_sim_{T+1} > UL_{T+1}

La probabilidad de anomalía se estima como:

P(anomalía) ≈ (1 / S) · Σ_{s=1}^{S} A_{T+1}(y_sim^{(s)})

---

## Resultados del sistema

- Pronóstico puntual por cliente  
- Bandas LL / UL robustas (cuantiles de residuales con recorte a cero)  
- Distribución Monte Carlo de facturación futura  
- Probabilidad de anomalía (baja / alta)  
- Visualizaciones interactivas en Streamlit  

---

## Licencia

MIT

