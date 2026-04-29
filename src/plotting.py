"""
src/plotting.py
================

Funciones de visualización reutilizables para el taller. Genera gráficas
consistentes en estilo entre todos los notebooks.

Funciones principales
---------------------
- ``plot_returns``: visualiza los retornos de los 23 activos en el tiempo.
- ``plot_curva_entrenamiento``: curva de loss train/val a lo largo de las épocas.
- ``plot_heatmap_mae``: matriz 4x4 de MAE para las distintas combinaciones de ventanas.
- ``plot_mae_por_activo``: barras horizontales con el MAE por activo.
- ``plot_comparacion_modelos``: barras agrupadas comparando varios modelos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Estilo por defecto para todas las figuras del proyecto.
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 100,
})


def plot_returns(
    returns: pd.DataFrame,
    title: str = "Retornos logarítmicos diarios de los 23 activos",
    save_path: str | Path | None = None,
):
    """
    Visualiza los retornos de los 23 activos en el tiempo.

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame con índice de fechas y una columna por activo.
    title : str
        Título de la gráfica.
    save_path : str | Path | None
        Si se especifica, guarda la figura en esa ruta.
    """
    fig, ax = plt.subplots(figsize=(15, 7))

    # Una línea por activo, con transparencia para que se solapen sin ocultar.
    for col in returns.columns:
        ax.plot(returns.index, returns[col], alpha=0.5, linewidth=0.5)

    ax.set_title(title)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Retorno logarítmico")
    ax.legend(returns.columns, loc="upper left", bbox_to_anchor=(1, 1), ncol=2)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax


def plot_curva_entrenamiento(
    history,
    title: str = "Curva de entrenamiento",
    metric: str = "loss",
    save_path: str | Path | None = None,
):
    """
    Plotea la curva de loss durante el entrenamiento (train y val).

    Parameters
    ----------
    history : keras.callbacks.History o dict
        Objeto devuelto por ``model.fit()`` o un dict con las listas.
    title : str
        Título de la gráfica.
    metric : str
        Métrica a plotear (por defecto 'loss').
    save_path : str | Path | None
        Si se especifica, guarda la figura.
    """
    # Acepta tanto objeto History como dict directo.
    if hasattr(history, "history"):
        hist_dict = history.history
    else:
        hist_dict = history

    fig, ax = plt.subplots(figsize=(10, 6))

    train_metric = hist_dict.get(metric)
    val_metric = hist_dict.get(f"val_{metric}")

    if train_metric is None:
        raise ValueError(f"La métrica '{metric}' no está en el history.")

    epochs = np.arange(1, len(train_metric) + 1)
    ax.plot(epochs, train_metric, label=f"Train {metric}", linewidth=2)

    if val_metric is not None:
        ax.plot(epochs, val_metric, label=f"Val {metric}", linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("Época")
    ax.set_ylabel(metric.upper())
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax


def plot_heatmap_mae(
    mae_matrix: np.ndarray,
    input_windows: Sequence[int] = (5, 10, 30, 90),
    output_windows: Sequence[int] = (1, 5, 30, 90),
    title: str = "MAE por combinación de ventanas",
    cmap: str = "viridis_r",
    fmt: str = ".4f",
    save_path: str | Path | None = None,
):
    """
    Heatmap de la matriz 4x4 de MAE para distintas combinaciones de ventanas.

    Parameters
    ----------
    mae_matrix : np.ndarray
        Matriz de forma (len(input_windows), len(output_windows)) con valores de MAE.
    input_windows : Sequence[int]
        Tamaños de ventana de entrada (filas).
    output_windows : Sequence[int]
        Tamaños de ventana de salida (columnas).
    title : str
        Título de la gráfica.
    cmap : str
        Mapa de colores. 'viridis_r' (invertido) hace que valores bajos sean
        más oscuros, lo que es más intuitivo para errores (menos = mejor = oscuro).
    fmt : str
        Formato para los valores en cada celda.
    save_path : str | Path | None
        Si se especifica, guarda la figura.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(mae_matrix, cmap=cmap, origin="upper", aspect="auto")

    # Anotar cada celda con su valor numérico.
    for i in range(len(input_windows)):
        for j in range(len(output_windows)):
            value = mae_matrix[i, j]
            if not np.isnan(value):
                # Color del texto según el fondo.
                text_color = "white" if value > np.nanmedian(mae_matrix) else "black"
                ax.text(j, i, f"{value:{fmt}}", ha="center", va="center",
                        color=text_color, fontsize=10, fontweight="bold")

    ax.set_xticks(np.arange(len(output_windows)))
    ax.set_yticks(np.arange(len(input_windows)))
    ax.set_xticklabels(output_windows)
    ax.set_yticklabels(input_windows)
    ax.set_xlabel("Ventana de salida (H, días)")
    ax.set_ylabel("Ventana de entrada (V, días)")
    ax.set_title(title)

    plt.colorbar(im, ax=ax, label="MAE")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax


def plot_mae_por_activo(
    mae_por_activo: np.ndarray,
    nombres_activos: Sequence[str],
    title: str = "MAE por activo",
    save_path: str | Path | None = None,
):
    """
    Barras horizontales con el MAE para cada activo.

    Útil para identificar qué activos predice mejor o peor el modelo.

    Parameters
    ----------
    mae_por_activo : np.ndarray
        Vector de longitud N con el MAE para cada activo.
    nombres_activos : Sequence[str]
        Lista de N tickers.
    title : str
        Título.
    save_path : str | Path | None
        Si se especifica, guarda la figura.
    """
    # Ordenamos por MAE para que sea más legible.
    orden = np.argsort(mae_por_activo)[::-1]
    valores = mae_por_activo[orden]
    etiquetas = [nombres_activos[i] for i in orden]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(etiquetas, valores, color="steelblue", alpha=0.8)
    ax.set_xlabel("MAE")
    ax.set_title(title)
    ax.invert_yaxis()  # Activo con más error arriba.
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax


def plot_comparacion_modelos(
    resultados: dict,
    metrica: str = "MAE test",
    title: str = "Comparación de modelos",
    save_path: str | Path | None = None,
):
    """
    Barras comparando varios modelos en una métrica concreta.

    Parameters
    ----------
    resultados : dict
        Diccionario {nombre_modelo: valor_metrica}.
    metrica : str
        Nombre de la métrica (para el eje y).
    title : str
        Título.
    save_path : str | Path | None
        Si se especifica, guarda la figura.
    """
    nombres = list(resultados.keys())
    valores = list(resultados.values())

    # Ordenamos de menor a mayor para que el mejor quede a la izquierda.
    orden = np.argsort(valores)
    nombres = [nombres[i] for i in orden]
    valores = [valores[i] for i in orden]

    fig, ax = plt.subplots(figsize=(10, 6))
    colores = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(nombres)))
    bars = ax.bar(nombres, valores, color=colores, alpha=0.85)

    # Anotar valor encima de cada barra.
    for bar, valor in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{valor:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel(metrica)
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax
