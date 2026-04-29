"""
src/baselines.py
=================

Modelos baseline no neuronales contra los que comparar las redes. Si una red
neuronal no bate consistentemente a estos baselines, no aporta valor.

Baselines incluidos
-------------------
- ``predecir_persistencia``: el target predicho es el último valor visto en X.
- ``predecir_media_historica``: el target predicho es la media histórica del activo.
- ``predecir_buy_and_hold``: estrategia "comprar y mantener" (predicción = 0 cambio).
- ``RegresionLinealMultiple``: regresión lineal multivariante con scikit-learn.

Notas
-----
- En el contexto del taller, "predecir" significa estimar el promedio de retornos
  durante la ventana de salida.
- Estos baselines NO se entrenan con épocas; son cálculos directos o ajustes
  cerrados (mínimos cuadrados).

NOTA: este módulo se completará en la Fase 2 del proyecto. La estructura está
preparada pero las funciones se implementarán cuando trabajemos en
01_baselines.ipynb.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression


def predecir_persistencia(X: np.ndarray) -> np.ndarray:
    """
    Baseline: la predicción es el último valor de la ventana de entrada.

    En el contexto del taller (predicción del promedio de retornos futuros),
    esto equivale a decir "el promedio futuro será igual al último retorno
    observado".

    Parameters
    ----------
    X : np.ndarray
        Tensor de forma (N, V, num_features).

    Returns
    -------
    np.ndarray
        Predicciones de forma (N, num_features).
    """
    # Tomamos el último timestep de cada ventana.
    return X[:, -1, :]


def predecir_media_historica(X: np.ndarray) -> np.ndarray:
    """
    Baseline: la predicción es la media de la ventana de entrada.

    Parameters
    ----------
    X : np.ndarray
        Tensor de forma (N, V, num_features).

    Returns
    -------
    np.ndarray
        Predicciones de forma (N, num_features), siendo la media a lo largo
        del eje temporal.
    """
    return X.mean(axis=1)


def predecir_buy_and_hold(X: np.ndarray) -> np.ndarray:
    """
    Baseline: la predicción es retorno cero (estrategia pasiva).

    Para retornos logarítmicos, "buy and hold" implica que no esperamos cambio
    sistemático, por lo que la mejor predicción del retorno medio futuro es 0.

    Parameters
    ----------
    X : np.ndarray
        Tensor de forma (N, V, num_features).

    Returns
    -------
    np.ndarray
        Predicciones de forma (N, num_features), todas a cero.
    """
    n_samples, _, n_features = X.shape
    return np.zeros((n_samples, n_features))


class RegresionLinealMultiple:
    """
    Wrapper sobre scikit-learn LinearRegression que acepta tensores 3D
    aplanándolos automáticamente, y soporta múltiples salidas.

    Esta clase replica exactamente el baseline de regresión lineal del notebook
    del profesor (``Lectura_datos_Taller_B3_T4_modelo_lineal.ipynb``).

    Examples
    --------
    >>> modelo = RegresionLinealMultiple()
    >>> modelo.fit(X_train, y_train)  # X_train shape: (N, V, 23), y_train: (N, 23)
    >>> predicciones = modelo.predict(X_test)
    """

    def __init__(self):
        self.modelo = LinearRegression()

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        """Aplana de (N, V, F) a (N, V*F). Si ya es 2D, no hace nada."""
        if X.ndim == 3:
            n = X.shape[0]
            return X.reshape(n, -1)
        return X

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Entrena la regresión lineal sobre los datos aplanados."""
        X_flat = self._flatten(X)
        self.modelo.fit(X_flat, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice sobre datos nuevos."""
        X_flat = self._flatten(X)
        return self.modelo.predict(X_flat)

    @property
    def num_parametros(self) -> int:
        """Número de parámetros del modelo (pesos + bias)."""
        coef = self.modelo.coef_
        # coef tiene forma (n_outputs, n_features). Más el intercept (n_outputs).
        n_pesos = coef.size
        n_bias = self.modelo.intercept_.size
        return int(n_pesos + n_bias)
