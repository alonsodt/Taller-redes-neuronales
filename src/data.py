"""
src/data.py
============

Funciones para la carga, generación de ventanas y partición temporal de los datos
del taller. Este módulo encapsula toda la "fontanería" de datos para que los
notebooks queden limpios y centrados en el análisis.

Funciones principales
---------------------
- ``descargar_datos``: descarga retornos del SP500 desde Yahoo Finance y los cachea
  en disco.
- ``cargar_returns``: lee los retornos ya cacheados sin volver a descargar.
- ``crear_ventanas``: convierte una serie temporal en pares (X, y) con la ventana
  de entrada y la ventana de salida especificadas.
- ``split_temporal``: realiza el split train/val/test ordenado por tiempo
  (sin shuffle), siguiendo la convención del taller.

Notas
-----
- Los datos se almacenan en formato Parquet por velocidad y para preservar tipos.
- La semilla por defecto es 42, igual que en los notebooks oficiales del profesor.
- El split por defecto es 85/5/10 (train/val/test) sobre el total.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# Ruta donde se cachean los datos. Asume que el script se ejecuta desde la raíz
# del proyecto o desde la carpeta notebooks/. Esto ajusta la ruta automáticamente.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Lista de tickers del taller. Son blue chips del SP500 con histórico desde 1945.
TICKERS = [
    "AEP", "BA", "CAT", "CNP", "CVX", "DIS", "DTE", "ED", "GD", "GE",
    "HON", "HPQ", "IBM", "IP", "JNJ", "KO", "KR", "MMM", "MO", "MRK",
    "MSI", "PG", "XOM",
]

START_DATE = "1945-01-01"
RANDOM_SEED = 42


def descargar_datos(force: bool = False, verbose: bool = True) -> pd.DataFrame:
    """
    Descarga los retornos logarítmicos diarios de los 23 activos del SP500.

    Si los datos ya están cacheados en ``data/returns.parquet``, los lee de disco
    en lugar de volver a descargar (a menos que ``force=True``).

    Parameters
    ----------
    force : bool, default=False
        Si True, vuelve a descargar incluso si hay caché.
    verbose : bool, default=True
        Si True, muestra mensajes de progreso.

    Returns
    -------
    pd.DataFrame
        DataFrame con índice de fechas y 23 columnas (una por activo). Cada
        valor es el retorno logarítmico diario.
    """
    cache_path = DATA_DIR / "returns.parquet"

    if cache_path.exists() and not force:
        if verbose:
            print(f"[descargar_datos] Cargando datos cacheados desde {cache_path.name}")
        return pd.read_parquet(cache_path)

    # Importamos yfinance solo cuando hace falta descargar (acelera imports).
    import yfinance as yf

    if verbose:
        print(f"[descargar_datos] Descargando {len(TICKERS)} tickers desde {START_DATE}...")

    precios = yf.download(
        TICKERS,
        start=START_DATE,
        auto_adjust=True,
        progress=verbose,
    )["Close"]

    # Eliminamos columnas con NaN: aseguramos que todos los tickers tienen
    # histórico completo. Si algún ticker se incorporó al mercado más tarde,
    # quedaría descartado (en este caso los 23 elegidos sí tienen histórico).
    precios.dropna(axis=1, inplace=True)

    # Calculamos retornos logarítmicos.
    returns = np.log(precios).diff().dropna()

    # Guardamos en disco.
    DATA_DIR.mkdir(exist_ok=True)
    precios.to_parquet(DATA_DIR / "precios_close.parquet")
    returns.to_parquet(cache_path)

    if verbose:
        print(f"[descargar_datos] Guardado en {cache_path.name}")
        print(f"[descargar_datos] Forma de los retornos: {returns.shape}")

    return returns


def cargar_returns(verbose: bool = True) -> pd.DataFrame:
    """
    Carga los retornos cacheados sin volver a descargar.

    Si no hay caché, lanza un FileNotFoundError con un mensaje claro indicando
    que hay que ejecutar primero el notebook 00.

    Returns
    -------
    pd.DataFrame
        DataFrame con los retornos logarítmicos diarios.
    """
    cache_path = DATA_DIR / "returns.parquet"

    if not cache_path.exists():
        raise FileNotFoundError(
            f"No se encontró {cache_path}. Ejecuta primero el notebook "
            f"00_descarga_datos.ipynb para generar los datos."
        )

    returns = pd.read_parquet(cache_path)
    if verbose:
        print(f"[cargar_returns] Cargados {returns.shape[0]} días, {returns.shape[1]} activos")

    return returns


def cargar_precios(verbose: bool = True) -> pd.DataFrame:
    """
    Carga los precios de cierre cacheados.

    Returns
    -------
    pd.DataFrame
        DataFrame con los precios de cierre ajustados.
    """
    cache_path = DATA_DIR / "precios_close.parquet"

    if not cache_path.exists():
        raise FileNotFoundError(
            f"No se encontró {cache_path}. Ejecuta primero el notebook "
            f"00_descarga_datos.ipynb para generar los datos."
        )

    precios = pd.read_parquet(cache_path)
    if verbose:
        print(f"[cargar_precios] Cargados {precios.shape[0]} días, {precios.shape[1]} activos")

    return precios


def crear_ventanas(
    data: pd.DataFrame | np.ndarray,
    input_window: int,
    output_window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convierte una serie temporal en pares (X, y) usando ventanas deslizantes.

    Para cada índice ``i``, construye:
        - X[i] = datos en el rango ``[i, i + input_window)``
        - y[i] = promedio de los datos en ``[i + input_window, i + input_window + output_window)``

    Parameters
    ----------
    data : pd.DataFrame o np.ndarray
        Serie temporal con forma (T, num_features).
    input_window : int
        Tamaño de la ventana de entrada (V).
    output_window : int
        Tamaño de la ventana de salida (H). El target es el promedio sobre
        esta ventana.

    Returns
    -------
    X : np.ndarray
        Tensor de forma (N, V, num_features).
    y : np.ndarray
        Tensor de forma (N, num_features). Cada fila es el promedio de los
        ``output_window`` valores que siguen a la ventana de entrada.

    Notas
    -----
    - Esta función replica la lógica de ``create_time_series_data`` del notebook
      del profesor.
    - Si ``output_window == 0``, el target es el último valor de la ventana de
      entrada (caso degenerado que el taller no usa pero soportamos por compat).
    """
    data_array = data.values if isinstance(data, pd.DataFrame) else np.asarray(data)

    if data_array.ndim != 2:
        raise ValueError(
            f"Se esperaba un array 2D (T, num_features), pero se recibió "
            f"forma {data_array.shape}"
        )

    T = len(data_array)
    n_samples = T - input_window - output_window + 1

    if n_samples <= 0:
        raise ValueError(
            f"No hay suficientes datos para input_window={input_window} y "
            f"output_window={output_window}. T={T}"
        )

    X = np.empty((n_samples, input_window, data_array.shape[1]), dtype=data_array.dtype)
    y = np.empty((n_samples, data_array.shape[1]), dtype=data_array.dtype)

    for i in range(n_samples):
        X[i] = data_array[i : i + input_window]

        if output_window > 0:
            output_slice = data_array[i + input_window : i + input_window + output_window]
            y[i] = output_slice.mean(axis=0)
        else:
            y[i] = data_array[i + input_window - 1]

    return X, y


def split_temporal(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.05,
    test_size: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Realiza el split train/val/test ordenado por tiempo (sin shuffle).

    El test se toma de los últimos datos (futuros), el val justo antes del test,
    y el train del resto. Replicamos la convención del profesor: 5% val sobre
    el train original.

    Parameters
    ----------
    X : np.ndarray
        Features con forma (N, ...).
    y : np.ndarray
        Targets con forma (N, ...).
    val_size : float, default=0.05
        Fracción de validación SOBRE EL TRAIN (no sobre el total). Esto replica
        ``test_size=0.05`` en el segundo split del profesor.
    test_size : float, default=0.10
        Fracción de test SOBRE EL TOTAL.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test : tuple of np.ndarray
    """
    n = len(X)

    # Primer split: separamos el test del resto (90/10).
    n_test = int(round(n * test_size))
    n_trainval = n - n_test

    X_trainval = X[:n_trainval]
    X_test = X[n_trainval:]
    y_trainval = y[:n_trainval]
    y_test = y[n_trainval:]

    # Segundo split: separamos val del train. val_size es relativo al train.
    n_val = int(round(n_trainval * val_size))
    n_train = n_trainval - n_val

    X_train = X_trainval[:n_train]
    X_val = X_trainval[n_train:]
    y_train = y_trainval[:n_train]
    y_val = y_trainval[n_train:]

    return X_train, X_val, X_test, y_train, y_val, y_test


def preparar_datos(
    returns: pd.DataFrame,
    input_window: int,
    output_window: int,
    val_size: float = 0.05,
    test_size: float = 0.10,
    verbose: bool = False,
) -> dict:
    """
    Pipeline completo: ventanas + split temporal.

    Es la función "todo en uno" que se llama desde los notebooks. Devuelve un
    dict con todos los conjuntos para no tener que hacer unpacking de muchas
    variables.

    Parameters
    ----------
    returns : pd.DataFrame
        Retornos logarítmicos.
    input_window : int
        Tamaño de la ventana de entrada (V).
    output_window : int
        Tamaño de la ventana de salida (H).
    val_size : float, default=0.05
        Fracción de validación sobre el train original.
    test_size : float, default=0.10
        Fracción de test sobre el total.
    verbose : bool, default=False
        Si True, imprime las formas de los conjuntos.

    Returns
    -------
    dict con claves: X_train, X_val, X_test, y_train, y_val, y_test
    """
    X, y = crear_ventanas(returns, input_window, output_window)
    X_train, X_val, X_test, y_train, y_val, y_test = split_temporal(
        X, y, val_size=val_size, test_size=test_size
    )

    if verbose:
        print(f"[preparar_datos] V={input_window}, H={output_window}")
        print(f"  X_train: {X_train.shape}   y_train: {y_train.shape}")
        print(f"  X_val:   {X_val.shape}   y_val:   {y_val.shape}")
        print(f"  X_test:  {X_test.shape}   y_test:  {y_test.shape}")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }
