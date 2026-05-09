"""
src/preprocessing.py
=====================

Técnicas avanzadas de preprocesado para series temporales financieras.
Implementa la diferenciación fraccionaria de López de Prado (Capítulo 5
de "Advances in Financial Machine Learning"), que es la técnica trabajada
en el Taller B3-T1 y que aplicaremos en el notebook 07 de investigación.

¿Por qué diferenciación fraccionaria?
--------------------------------------
El problema clásico en series financieras:

- Los **precios** tienen memoria (son no estacionarios): el precio de hoy
  depende del de ayer, que depende del de antes... La serie tiene una raíz
  unitaria y los modelos estadísticos no funcionan bien con ella.

- Los **retornos logarítmicos** (diferenciación entera d=1) son estacionarios
  pero pierden toda la memoria de largo plazo. La información sobre tendencias
  de semanas o meses queda destruida.

La diferenciación fraccionaria con orden 0 < d < 1 encuentra el equilibrio:
la serie resultante es estacionaria (apta para modelos) pero conserva parte
de la memoria de largo plazo.

Funciones principales
---------------------
- ``calcular_pesos``: calcula los pesos de la diferenciación fraccionaria.
- ``diferenciacion_fraccionaria``: aplica la transformación a una serie.
- ``diferenciacion_fraccionaria_df``: aplica a un DataFrame completo.
- ``encontrar_d_minimo``: busca el d mínimo que hace estacionaria la serie
  (test ADF).
- ``comparar_retornos_vs_fraccionaria``: compara ambas transformaciones
  visualmente.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def calcular_pesos(d: float, umbral: float = 1e-3) -> np.ndarray:
    """
    Calcula los pesos de la diferenciación fraccionaria de orden d.

    La diferenciación fraccionaria expande los retornos con pesos que
    decaen lentamente según la fórmula binomial generalizada:

        w_k = -w_{k-1} * (d - k + 1) / k

    Los pesos se truncan cuando su valor absoluto cae por debajo de
    ``umbral``, lo que controla cuánto "pasado" considera la transformación.

    Parameters
    ----------
    d : float
        Orden de diferenciación. d=0 → sin cambio. d=1 → diferencia entera
        (equivalente a retornos logarítmicos). 0 < d < 1 → fraccionaria.
    umbral : float
        Truncar los pesos cuando |w_k| < umbral. Valores más pequeños
        conservan más memoria pero hacen la operación más lenta.

    Returns
    -------
    np.ndarray
        Vector de pesos ordenados del más reciente (w_0=1) al más antiguo.

    Examples
    --------
    >>> pesos = calcular_pesos(d=0.4)
    >>> len(pesos)  # cuántos pasos hacia atrás considera
    """
    pesos = [1.0]
    k = 1
    while True:
        w = -pesos[-1] * (d - k + 1) / k
        if abs(w) < umbral:
            break
        pesos.append(w)
        k += 1
    return np.array(pesos[::-1])  # del más antiguo al más reciente


def diferenciacion_fraccionaria(
    serie: pd.Series,
    d: float,
    umbral: float = 1e-3,
) -> pd.Series:
    """
    Aplica diferenciación fraccionaria de orden d a una serie temporal.

    Parameters
    ----------
    serie : pd.Series
        Serie temporal (ej. precios de cierre de un activo).
    d : float
        Orden de diferenciación (0 < d < 1 para fraccionaria).
    umbral : float
        Umbral de truncamiento de pesos.

    Returns
    -------
    pd.Series
        Serie transformada, con el mismo índice que la original pero con
        NaN en las primeras posiciones (por la ventana de pesos).

    Notes
    -----
    - Aplicar sobre precios (no sobre retornos ya diferenciados).
    - Los primeros len(pesos)-1 valores serán NaN porque necesitan
      historia previa para calcularse.
    """
    pesos = calcular_pesos(d, umbral)
    n_pesos = len(pesos)
    valores = serie.values
    n = len(valores)

    resultado = np.full(n, np.nan)

    for i in range(n_pesos - 1, n):
        ventana = valores[i - n_pesos + 1 : i + 1]
        resultado[i] = np.dot(pesos, ventana)

    return pd.Series(resultado, index=serie.index, name=serie.name)


def diferenciacion_fraccionaria_df(
    df: pd.DataFrame,
    d: float,
    umbral: float = 1e-3,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Aplica diferenciación fraccionaria a todas las columnas de un DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con series temporales (ej. precios de 23 activos).
    d : float
        Orden de diferenciación.
    umbral : float
        Umbral de truncamiento de pesos.
    dropna : bool
        Si True, elimina las filas con NaN del inicio (recomendado).

    Returns
    -------
    pd.DataFrame
        DataFrame transformado con las mismas columnas.

    Examples
    --------
    >>> from src.data import cargar_precios
    >>> precios = cargar_precios()
    >>> frac_diff = diferenciacion_fraccionaria_df(precios, d=0.4)
    """
    resultado = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    for col in df.columns:
        resultado[col] = diferenciacion_fraccionaria(df[col], d, umbral)

    if dropna:
        resultado.dropna(inplace=True)

    return resultado


def test_estacionariedad(
    serie: pd.Series,
    nivel_significancia: float = 0.05,
) -> dict:
    """
    Aplica el test ADF (Augmented Dickey-Fuller) para comprobar si una
    serie es estacionaria.

    H0: la serie tiene raíz unitaria (no es estacionaria).
    Si p-value < nivel_significancia → rechazamos H0 → la serie ES estacionaria.

    Parameters
    ----------
    serie : pd.Series
        Serie temporal a testear. Se eliminan NaN automáticamente.
    nivel_significancia : float
        Nivel de significancia para el test (por defecto 0.05 = 5%).

    Returns
    -------
    dict con claves:
        - estadistico: el estadístico ADF.
        - p_value: el p-value del test.
        - es_estacionaria: bool, True si p_value < nivel_significancia.
        - valores_criticos: dict con umbrales al 1%, 5% y 10%.
    """
    serie_limpia = serie.dropna()
    resultado = adfuller(serie_limpia, autolag="AIC")

    return {
        "estadistico":     resultado[0],
        "p_value":         resultado[1],
        "es_estacionaria": resultado[1] < nivel_significancia,
        "valores_criticos": resultado[4],
    }


def encontrar_d_minimo(
    serie: pd.Series,
    d_min: float = 0.0,
    d_max: float = 1.0,
    paso: float = 0.05,
    umbral_adf: float = 0.05,
    umbral_pesos: float = 1e-3,
    verbose: bool = True,
) -> float:
    """
    Busca el orden de diferenciación d mínimo que hace estacionaria la serie.

    Itera sobre valores de d desde d_min hasta d_max en pasos de ``paso``.
    Para cada d aplica la diferenciación fraccionaria y comprueba con ADF
    si la serie resultante es estacionaria. Devuelve el primer d que lo logra.

    El d mínimo es el óptimo de López de Prado: conserva la máxima memoria
    posible (d pequeño) mientras garantiza estacionariedad (ADF significativo).

    Parameters
    ----------
    serie : pd.Series
        Serie de precios (no retornos) de un activo.
    d_min, d_max : float
        Rango de búsqueda de d.
    paso : float
        Incremento de d en cada iteración.
    umbral_adf : float
        Nivel de significancia para el test ADF.
    umbral_pesos : float
        Umbral de truncamiento de pesos en la diferenciación.
    verbose : bool
        Si True, imprime el resultado de cada iteración.

    Returns
    -------
    float
        El d mínimo que hace estacionaria la serie.
        Devuelve d_max si ningún valor en el rango lo consigue.

    Examples
    --------
    >>> from src.data import cargar_precios
    >>> precios = cargar_precios()
    >>> d_opt = encontrar_d_minimo(precios["IBM"], verbose=True)
    >>> print(f"d óptimo para IBM: {d_opt}")
    """
    valores_d = np.arange(d_min, d_max + paso, paso)

    for d in valores_d:
        d = round(d, 4)
        serie_diff = diferenciacion_fraccionaria(serie, d, umbral_pesos)
        test = test_estacionariedad(serie_diff, umbral_adf)

        if verbose:
            estado = "✓ estacionaria" if test["es_estacionaria"] else "✗ no estacionaria"
            print(f"  d={d:.2f}  p-value={test['p_value']:.4f}  {estado}")

        if test["es_estacionaria"]:
            return d

    if verbose:
        print(f"  Ningún d en [{d_min}, {d_max}] logró estacionariedad.")
    return d_max


def encontrar_d_minimo_df(
    df: pd.DataFrame,
    d_min: float = 0.0,
    d_max: float = 1.0,
    paso: float = 0.05,
    umbral_adf: float = 0.05,
    verbose: bool = False,
) -> pd.Series:
    """
    Aplica encontrar_d_minimo a todas las columnas de un DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de precios con una columna por activo.
    d_min, d_max, paso, umbral_adf : float
        Parámetros del test (ver encontrar_d_minimo).
    verbose : bool
        Si True, imprime el progreso por activo.

    Returns
    -------
    pd.Series
        Serie con el d óptimo por activo (índice = nombre del activo).

    Examples
    --------
    >>> from src.data import cargar_precios
    >>> precios = cargar_precios()
    >>> d_optimos = encontrar_d_minimo_df(precios)
    >>> print(d_optimos.describe())
    """
    d_optimos = {}

    for col in df.columns:
        if verbose:
            print(f"\nBuscando d mínimo para {col}:")
        d_opt = encontrar_d_minimo(
            df[col], d_min, d_max, paso, umbral_adf, verbose=verbose
        )
        d_optimos[col] = d_opt
        if not verbose:
            print(f"  {col}: d_opt = {d_opt:.2f}")

    return pd.Series(d_optimos, name="d_optimo")


def preparar_datos_fraccionarios(
    precios: pd.DataFrame,
    d: float | pd.Series,
    umbral: float = 1e-3,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Pipeline completo: aplica diferenciación fraccionaria a los precios
    y devuelve un DataFrame listo para usar como input de las redes.

    Acepta un d global (mismo para todos los activos) o un d por activo
    (pd.Series devuelta por encontrar_d_minimo_df).

    Parameters
    ----------
    precios : pd.DataFrame
        Precios de cierre de los activos.
    d : float o pd.Series
        Orden de diferenciación. Si es float, se aplica el mismo a todos.
        Si es pd.Series, cada activo usa su propio d óptimo.
    umbral : float
        Umbral de truncamiento de pesos.
    verbose : bool
        Si True, imprime el shape del resultado.

    Returns
    -------
    pd.DataFrame
        Series diferenciadas fraccionariamente, con NaN eliminados.
        Listo para pasarse a crear_ventanas() de src/data.py.

    Examples
    --------
    >>> from src.data import cargar_precios
    >>> precios = cargar_precios()
    >>>
    >>> # Opción A: d global
    >>> frac_returns = preparar_datos_fraccionarios(precios, d=0.4)
    >>>
    >>> # Opción B: d óptimo por activo
    >>> d_optimos = encontrar_d_minimo_df(precios)
    >>> frac_returns = preparar_datos_fraccionarios(precios, d=d_optimos)
    """
    if isinstance(d, (int, float)):
        # Mismo d para todos los activos
        resultado = diferenciacion_fraccionaria_df(precios, d, umbral)
    else:
        # d distinto por activo (pd.Series con índice = nombre de activo)
        resultado = pd.DataFrame(index=precios.index, dtype=float)
        for col in precios.columns:
            d_col = d.get(col, 0.4) if hasattr(d, 'get') else d[col]
            resultado[col] = diferenciacion_fraccionaria(
                precios[col], d_col, umbral
            )
        resultado.dropna(inplace=True)

    if verbose:
        print(f"[preparar_datos_fraccionarios] Shape: {resultado.shape}")
        print(f"  Rango: {resultado.index.min()} → "
              f"{resultado.index.max()}")

    return resultado
