"""
src/evaluation.py
==================

Funciones para evaluar modelos y generar tablas comparativas.

La métrica principal del taller es MAE (Mean Absolute Error), pero conviene
reportar también el MAE por activo y opcionalmente el hit rate (% de aciertos
en signo) para entender mejor el comportamiento del modelo.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def mae_global(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error global (un solo número).

    Es la métrica oficial del taller.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Arrays de la misma forma.

    Returns
    -------
    float
        El MAE como un escalar.
    """
    return float(np.mean(np.abs(y_pred - y_true)))


def mae_por_activo(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nombres_activos: Sequence[str] | None = None,
) -> pd.Series:
    """
    MAE para cada activo (cada columna del output).

    Útil para identificar qué activos predice mejor el modelo.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Arrays de forma (N, num_activos).
    nombres_activos : Sequence[str] | None
        Lista de nombres de los activos. Si es None, se usan índices numéricos.

    Returns
    -------
    pd.Series
        MAE por activo, indexado por nombre.
    """
    errores = np.mean(np.abs(y_pred - y_true), axis=0)
    if nombres_activos is None:
        nombres_activos = [f"activo_{i}" for i in range(len(errores))]
    return pd.Series(errores, index=nombres_activos, name="MAE")


def hit_rate_signo(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Proporción de aciertos en el signo de la predicción.

    En forecasting financiero, predecir el signo correcto a veces es más útil
    que predecir el valor exacto: te dice si subirá o bajará.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Arrays con los retornos reales y predichos.

    Returns
    -------
    float
        Proporción entre 0 y 1.
    """
    aciertos = np.sign(y_true) == np.sign(y_pred)
    return float(np.mean(aciertos))


def evaluar_modelo(
    modelo,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    nombre: str = "modelo",
    incluir_hit_rate: bool = False,
) -> dict:
    """
    Evalúa un modelo en train/val/test y devuelve un dict con todas las métricas.

    Parameters
    ----------
    modelo : objeto con método ``predict``
        Puede ser un modelo de Keras, scikit-learn o cualquier wrapper.
    X_train, y_train, X_val, y_val, X_test, y_test : np.ndarray
        Conjuntos de datos.
    nombre : str
        Nombre del modelo (para identificarlo en tablas comparativas).
    incluir_hit_rate : bool
        Si True, calcula también el hit rate del signo.

    Returns
    -------
    dict con claves: nombre, MAE_train, MAE_val, MAE_test, [hit_rate_test]
    """
    pred_train = modelo.predict(X_train, verbose=0) if hasattr(modelo, "predict") else None
    pred_val = modelo.predict(X_val, verbose=0) if hasattr(modelo, "predict") else None
    pred_test = modelo.predict(X_test, verbose=0) if hasattr(modelo, "predict") else None

    # Algunos modelos (sklearn) no aceptan ``verbose``. Reintentamos sin él.
    if pred_train is None:
        pred_train = modelo.predict(X_train)
        pred_val = modelo.predict(X_val)
        pred_test = modelo.predict(X_test)

    resultado = {
        "nombre": nombre,
        "MAE_train": mae_global(y_train, pred_train),
        "MAE_val": mae_global(y_val, pred_val),
        "MAE_test": mae_global(y_test, pred_test),
    }

    if incluir_hit_rate:
        resultado["hit_rate_test"] = hit_rate_signo(y_test, pred_test)

    return resultado


def construir_matriz_resultados(
    resultados: list[dict],
    input_windows: Sequence[int] = (5, 10, 30, 90),
    output_windows: Sequence[int] = (1, 5, 30, 90),
    metrica: str = "MAE_test",
) -> np.ndarray:
    """
    A partir de una lista de resultados (uno por combinación de ventanas),
    construye la matriz 4x4 con la métrica solicitada.

    Cada elemento de ``resultados`` debe ser un dict que incluya 'V', 'H' y
    la métrica indicada.

    Parameters
    ----------
    resultados : list[dict]
        Cada dict tiene al menos las claves V, H y la métrica.
    input_windows : Sequence[int]
        Tamaños de ventana de entrada (filas).
    output_windows : Sequence[int]
        Tamaños de ventana de salida (columnas).
    metrica : str
        Clave del dict que contiene el valor a usar.

    Returns
    -------
    np.ndarray
        Matriz de forma (len(input_windows), len(output_windows)).
    """
    matriz = np.full((len(input_windows), len(output_windows)), np.nan)

    for r in resultados:
        if r.get("V") in input_windows and r.get("H") in output_windows:
            i = list(input_windows).index(r["V"])
            j = list(output_windows).index(r["H"])
            matriz[i, j] = r[metrica]

    return matriz
