"""
forecast_models.py

Modelos "suaves" (clásicos) de pronóstico univariado mensual:
- Naive (último valor)
- Seasonal Naive (y_{t-s})
- Moving Average (media móvil)
- SES
- Holt
- Holt-Winters (tendencia + estacionalidad)
- ARIMA
- SARIMA (si estacionalidad)

Incluye:
- preparación de serie mensual
- detección automática de estacionalidad (ACF en lag estacional)
- evaluación walk-forward con MSE
- selección automática del mejor modelo por MSE
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


# =========================
# Tipos
# =========================

@dataclass
class ForecastResult:
    model_name: str
    y_hat: pd.Series
    fitted: Optional[pd.Series] = None
    details: Optional[dict] = None


# =========================
# Preparación de series
# =========================

def to_monthly_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    freq: str = "MS",
    fill_missing: str = "zero",  # "zero" | "nan"
) -> pd.Series:
    """
    Convierte un DataFrame a serie mensual (index datetime, freq MS por defecto).
    """
    tmp = df[[date_col, value_col]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col, value_col])

    if tmp.empty:
        return pd.Series(dtype=float)

    # Mes como primer día del mes (robusto en pandas)
    tmp["__month"] = tmp[date_col].values.astype("datetime64[M]")
    tmp["__month"] = pd.to_datetime(tmp["__month"])

    y = tmp.groupby("__month")[value_col].sum().sort_index()
    y.index = pd.DatetimeIndex(y.index)

    # Forzar frecuencia mensual
    y = y.asfreq(freq)

    if fill_missing == "zero":
        y = y.fillna(0.0)

    return y.astype(float)



def check_min_length(y: pd.Series, min_len: int) -> None:
    if len(y) < min_len:
        raise ValueError(f"Serie insuficiente: requiere >= {min_len} puntos, tiene {len(y)}.")


# =========================
# Detección automática de estacionalidad
# =========================

def detect_seasonality_acf(
    y: pd.Series,
    seasonal_period: int = 12,
    threshold: float = 0.30,
    min_cycles: int = 2,
) -> Dict[str, float | bool]:
    """
    Detecta estacionalidad usando autocorrelación en lag = seasonal_period.

    - Retorna dict con:
        * is_seasonal: bool
        * strength: autocorr(lag)
        * seasonal_period: int

    Criterio:
    - len(y) >= min_cycles * seasonal_period + 1
    - abs(autocorr(lag)) >= threshold
    """
    y = y.dropna().astype(float)

    enough = len(y) >= (min_cycles * seasonal_period + 1)
    if not enough:
        return {"is_seasonal": False, "strength": 0.0, "seasonal_period": seasonal_period}

    # autocorr en pandas (Pearson) en el lag
    strength = float(y.autocorr(lag=seasonal_period))
    is_seasonal = bool(np.isfinite(strength) and abs(strength) >= threshold)

    return {"is_seasonal": is_seasonal, "strength": strength, "seasonal_period": seasonal_period}


def detect_best_seasonality(
    y: pd.Series,
    candidate_periods: Tuple[int, ...] = (12, 6),
    threshold: float = 0.30,
    min_cycles: int = 2,
) -> Dict[str, float | bool | int]:
    """
    Prueba varios periodos estacionales (p.ej. 12 y 6) y elige el que tenga mayor |acf|.
    Retorna:
      - is_seasonal
      - seasonal_period (el mejor)
      - strength (acf del mejor)
    """
    best = {"is_seasonal": False, "seasonal_period": int(candidate_periods[0]), "strength": 0.0}

    for p in candidate_periods:
        r = detect_seasonality_acf(y, seasonal_period=p, threshold=threshold, min_cycles=min_cycles)
        if abs(float(r["strength"])) > abs(float(best["strength"])):
            best = {"is_seasonal": bool(r["is_seasonal"]), "seasonal_period": int(p), "strength": float(r["strength"])}

    # Recalcular is_seasonal con threshold para el "best"
    best["is_seasonal"] = bool(abs(best["strength"]) >= threshold and len(y.dropna()) >= (min_cycles * best["seasonal_period"] + 1))
    return best


# =========================
# Forecasts
# =========================

def forecast_naive(y_train: pd.Series, h: int = 1) -> ForecastResult:
    last = float(y_train.iloc[-1])
    idx = pd.date_range(y_train.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    return ForecastResult("naive", pd.Series([last] * h, index=idx, dtype=float))


def forecast_seasonal_naive(y_train: pd.Series, h: int = 1, season_length: int = 12) -> ForecastResult:
    check_min_length(y_train, season_length + 1)
    idx = pd.date_range(y_train.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    vals = [float(y_train.iloc[-season_length + (i % season_length)]) for i in range(h)]
    return ForecastResult("seasonal_naive", pd.Series(vals, index=idx, dtype=float), details={"season_length": season_length})


def forecast_moving_average(y_train: pd.Series, h: int = 1, window: int = 3) -> ForecastResult:
    check_min_length(y_train, min(window, len(y_train)))
    avg = float(y_train.tail(window).mean())
    idx = pd.date_range(y_train.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    return ForecastResult("moving_average", pd.Series([avg] * h, index=idx, dtype=float), details={"window": window})


def forecast_ses(y_train: pd.Series, h: int = 1) -> ForecastResult:
    model = SimpleExpSmoothing(y_train, initialization_method="estimated")
    fit = model.fit(optimized=True)
    idx = pd.date_range(y_train.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    yhat = pd.Series(fit.forecast(h), index=idx, dtype=float)
    fitted = pd.Series(fit.fittedvalues, index=y_train.index, dtype=float)
    return ForecastResult("ses", yhat, fitted=fitted, details={"params": dict(fit.params)})


def forecast_holt(y_train: pd.Series, h: int = 1, damped_trend: bool = False) -> ForecastResult:
    model = Holt(y_train, initialization_method="estimated", damped_trend=damped_trend)
    fit = model.fit(optimized=True)
    idx = pd.date_range(y_train.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    yhat = pd.Series(fit.forecast(h), index=idx, dtype=float)
    fitted = pd.Series(fit.fittedvalues, index=y_train.index, dtype=float)
    return ForecastResult("holt", yhat, fitted=fitted, details={"params": dict(fit.params), "damped_trend": damped_trend})


def forecast_holt_winters(
    y_train: pd.Series,
    h: int = 1,
    season_length: int = 12,
    trend: str = "add",
    seasonal: str = "add",
    damped_trend: bool = False,
) -> ForecastResult:
    # regla práctica: >= 2 ciclos estacionales
    check_min_length(y_train, season_length * 2)
    model = ExponentialSmoothing(
        y_train,
        trend=trend,
        damped_trend=damped_trend,
        seasonal=seasonal,
        seasonal_periods=season_length,
        initialization_method="estimated",
    )
    fit = model.fit(optimized=True)
    idx = pd.date_range(y_train.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    yhat = pd.Series(fit.forecast(h), index=idx, dtype=float)
    fitted = pd.Series(fit.fittedvalues, index=y_train.index, dtype=float)
    return ForecastResult(
        "holt_winters",
        yhat,
        fitted=fitted,
        details={"params": dict(fit.params), "season_length": season_length, "trend": trend, "seasonal": seasonal, "damped_trend": damped_trend},
    )


def forecast_arima(y_train: pd.Series, h: int = 1, order: Tuple[int, int, int] = (1, 1, 1)) -> ForecastResult:
    model = SARIMAX(
        y_train,
        order=order,
        seasonal_order=(0, 0, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    idx = pd.date_range(y_train.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    yhat = pd.Series(fit.forecast(steps=h), index=idx, dtype=float)
    fitted = pd.Series(fit.fittedvalues, index=y_train.index, dtype=float)
    return ForecastResult("arima", yhat, fitted=fitted, details={"order": order})


def forecast_sarima(
    y_train: pd.Series,
    h: int = 1,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (0, 1, 1, 12),
) -> ForecastResult:
    check_min_length(y_train, seasonal_order[3] * 2)
    model = SARIMAX(
        y_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    idx = pd.date_range(y_train.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    yhat = pd.Series(fit.forecast(steps=h), index=idx, dtype=float)
    fitted = pd.Series(fit.fittedvalues, index=y_train.index, dtype=float)
    return ForecastResult("sarima", yhat, fitted=fitted, details={"order": order, "seasonal_order": seasonal_order})


# =========================
# Walk-forward evaluation
# =========================

def walk_forward_mse(
    y: pd.Series,
    forecaster: Callable[[pd.Series, int], ForecastResult],
    h: int = 1,
    initial_train: int = 12,
) -> float:
    y = y.dropna().astype(float)
    check_min_length(y, initial_train + h)

    errors = []
    for t in range(initial_train, len(y) - h + 1):
        y_train = y.iloc[:t]
        y_true = y.iloc[t : t + h].values.astype(float)

        try:
            res = forecaster(y_train, h)
            y_pred = res.y_hat.values.astype(float)
            errors.append(np.mean((y_true - y_pred) ** 2))
        except Exception:
            return float("inf")

    return float(np.mean(errors)) if errors else float("inf")


def evaluate_models(
    y: pd.Series,
    initial_train: int = 12,
    h: int = 1,
    ma_window: int = 3,
    arima_order: Tuple[int, int, int] = (1, 1, 1),
    sarima_order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_candidates: Tuple[int, ...] = (12, 6),
    seasonal_threshold: float = 0.30,
    seasonal_min_cycles: int = 2,
) -> Tuple[pd.DataFrame, Dict[str, float | bool | int]]:
    """
    Evalúa modelos y retorna:
      - ranking DataFrame (model, mse)
      - dict con info de estacionalidad detectada
    """
    y = y.dropna().astype(float)
    check_min_length(y, initial_train + h)

    # Detectar estacionalidad (automático)
    s_info = detect_best_seasonality(
        y,
        candidate_periods=seasonal_candidates,
        threshold=seasonal_threshold,
        min_cycles=seasonal_min_cycles,
    )
    is_seasonal = bool(s_info["is_seasonal"])
    s = int(s_info["seasonal_period"])

    models: List[Tuple[str, Callable[[pd.Series, int], ForecastResult]]] = [
        ("naive", lambda yt, hh: forecast_naive(yt, hh)),
        ("moving_average", lambda yt, hh: forecast_moving_average(yt, hh, window=ma_window)),
        ("ses", lambda yt, hh: forecast_ses(yt, hh)),
        ("holt", lambda yt, hh: forecast_holt(yt, hh)),
        ("arima", lambda yt, hh: forecast_arima(yt, hh, order=arima_order)),
    ]

    # Solo incluir estacionales si detecta estacionalidad
    if is_seasonal:
        models.extend([
            ("seasonal_naive", lambda yt, hh: forecast_seasonal_naive(yt, hh, season_length=s)),
            ("holt_winters", lambda yt, hh: forecast_holt_winters(yt, hh, season_length=s)),
            ("sarima", lambda yt, hh: forecast_sarima(yt, hh, order=sarima_order, seasonal_order=(0, 1, 1, s))),
        ])

    rows = []
    for name, fn in models:
        mse = walk_forward_mse(y, fn, h=h, initial_train=initial_train)
        rows.append({"model": name, "mse": mse})

    out = pd.DataFrame(rows).sort_values("mse", ascending=True).reset_index(drop=True)
    return out, s_info


def fit_best_and_forecast(
    y: pd.Series,
    ranking: pd.DataFrame,
    seasonality_info: Dict[str, float | bool | int],
    h: int = 1,
    ma_window: int = 3,
    arima_order: Tuple[int, int, int] = (1, 1, 1),
    sarima_order: Tuple[int, int, int] = (1, 1, 1),
) -> ForecastResult:
    """
    Ajusta el mejor modelo (según ranking) sobre toda la serie y pronostica h pasos.
    """
    y = y.dropna().astype(float)

    best = str(ranking.iloc[0]["model"])
    s = int(seasonality_info.get("seasonal_period", 12))
    is_seasonal = bool(seasonality_info.get("is_seasonal", False))

    if best == "naive":
        return forecast_naive(y, h)
    if best == "moving_average":
        return forecast_moving_average(y, h, window=ma_window)
    if best == "ses":
        return forecast_ses(y, h)
    if best == "holt":
        return forecast_holt(y, h)
    if best == "arima":
        return forecast_arima(y, h, order=arima_order)

    # estacionales (deberían estar en ranking solo si is_seasonal=True)
    if best == "seasonal_naive":
        if not is_seasonal:
            raise ValueError("Se solicitó seasonal_naive pero no se detectó estacionalidad.")
        return forecast_seasonal_naive(y, h, season_length=s)

    if best == "holt_winters":
        if not is_seasonal:
            raise ValueError("Se solicitó holt_winters pero no se detectó estacionalidad.")
        return forecast_holt_winters(y, h, season_length=s)

    if best == "sarima":
        if not is_seasonal:
            raise ValueError("Se solicitó sarima pero no se detectó estacionalidad.")
        return forecast_sarima(y, h, order=sarima_order, seasonal_order=(0, 1, 1, s))

    raise ValueError(f"Modelo desconocido: {best}")

def residual_quantile_bands(
    y: pd.Series,
    model_name: str,
    seasonality_info: dict,
    alpha: float = 0.05,
    ma_window: int = 3,
    arima_order: Tuple[int, int, int] = (1, 1, 1),
    sarima_order: Tuple[int, int, int] = (1, 1, 1),
) -> Dict[str, float]:
    """
    Calcula cuantiles de residuales (e_t = y_t - yhat_in_sample_t) para construir bandas.
    Retorna:
      - q_low, q_high (cuantiles de residuales)
    """
    y = y.dropna().astype(float)
    s = int(seasonality_info.get("seasonal_period", 12))
    is_seasonal = bool(seasonality_info.get("is_seasonal", False))

    # In-sample fitted según modelo ganador
    if model_name == "naive":
        fitted = y.shift(1)
    elif model_name == "seasonal_naive":
        fitted = y.shift(s)
    elif model_name == "moving_average":
        fitted = y.rolling(ma_window).mean().shift(1)
    elif model_name == "ses":
        fit = SimpleExpSmoothing(y, initialization_method="estimated").fit(optimized=True)
        fitted = pd.Series(fit.fittedvalues, index=y.index)
    elif model_name == "holt":
        fit = Holt(y, initialization_method="estimated").fit(optimized=True)
        fitted = pd.Series(fit.fittedvalues, index=y.index)
    elif model_name == "holt_winters":
        fit = ExponentialSmoothing(
            y,
            trend="add",
            seasonal="add",
            seasonal_periods=s,
            initialization_method="estimated",
        ).fit(optimized=True)
        fitted = pd.Series(fit.fittedvalues, index=y.index)
    elif model_name == "arima":
        fit = SARIMAX(y, order=arima_order, seasonal_order=(0,0,0,0),
                     enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fitted = pd.Series(fit.fittedvalues, index=y.index)
    elif model_name == "sarima":
        if not is_seasonal:
            # fallback: tratar como ARIMA si no hay estacionalidad
            fit = SARIMAX(y, order=arima_order, seasonal_order=(0,0,0,0),
                         enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        else:
            fit = SARIMAX(y, order=sarima_order, seasonal_order=(0,1,1,s),
                         enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fitted = pd.Series(fit.fittedvalues, index=y.index)
    else:
        raise ValueError(f"Modelo no soportado para residuales: {model_name}")

    # Residuales (alinear y limpiar)
    e = (y - fitted).dropna()
    if e.empty:
        return {"q_low": 0.0, "q_high": 0.0}

    # =========================
    # Solo meses activos (y_t > 0)
    # =========================
    e_active = e[y.loc[e.index] > 0]

    # Fallback: si quedan pocos puntos activos, usar todos los residuales
    min_active = 8
    if len(e_active) >= min_active:
        e_use = e_active
    else:
        e_use = e

    q_low = float(e_use.quantile(alpha))
    q_high = float(e_use.quantile(1 - alpha))
    return {"q_low": q_low, "q_high": q_high}

