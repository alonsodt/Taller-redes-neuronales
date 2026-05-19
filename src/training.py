"""
src/training.py
================

Loop de entrenamiento estandarizado para redes Keras.

La función principal es entrenar_modelo(), que:
1. Configura los callbacks (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau).
2. Lanza model.fit con la configuración estándar del taller.
3. Guarda los pesos del mejor modelo en results/checkpoints/.
4. Devuelve el history para plotear curvas.

También incluye entrenar_todos_los_modelos(), que recorre las 16
combinaciones de ventanas y varios modelos, guardando todos los resultados
en una lista para generar tablas y heatmaps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import keras
from keras import Model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = PROJECT_ROOT / "results" / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

def fijar_semilla(seed: int = 42):
    """
    Fija todas las semillas de aleatoriedad para reproducibilidad.
    Llama a esta función antes de instanciar y entrenar cualquier modelo.
    """
    import os
    import random
    import tensorflow as tf

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def entrenar_modelo(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 20,
    nombre: str = "modelo",
    verbose: int = 0,
    seed: int = 42,
) -> keras.callbacks.History:
    """
    Entrena un modelo Keras con callbacks estándar del taller.

    Callbacks configurados
    ----------------------
    - ModelCheckpoint: guarda el mejor modelo (menor val_loss) en disco.
      Ruta: results/checkpoints/{nombre}.keras
    - EarlyStopping: para el entrenamiento si val_loss no mejora en
      'patience' épocas consecutivas. Restaura los mejores pesos al final.
    - ReduceLROnPlateau: reduce el learning rate a la mitad cuando val_loss
      se estanca durante 5 épocas. Ayuda a salir de mínimos locales.

    Parameters
    ----------
    model : Model
        Modelo Keras ya compilado (salida de cualquier build_* de models.py).
    X_train, y_train : np.ndarray
        Datos de entrenamiento.
    X_val, y_val : np.ndarray
        Datos de validación.
    epochs : int
        Número máximo de épocas. EarlyStopping puede parar antes.
    batch_size : int
        Tamaño del batch. 32 es el default razonable.
    patience : int
        Épocas sin mejora en val_loss antes de parar. 20 es conservador
        pero evita parar demasiado pronto cuando la curva es ruidosa.
    nombre : str
        Nombre del modelo, usado para el archivo de checkpoint.
    verbose : int
        0 = silencioso, 1 = barra de progreso, 2 = una línea por época.
    seed : int
        Semilla de aleatoriedad para reproducibilidad.

    Returns
    -------
    keras.callbacks.History
        El history del entrenamiento. Úsalo con plot_curva_entrenamiento()
        para visualizar cómo evolucionó la loss.

    Examples
    --------
    >>> from src.models import build_dense_model
    >>> from src.training import entrenar_modelo
    >>> model = build_dense_model(input_shape=(30, 23), hidden_units=[64])
    >>> hist = entrenar_modelo(model, X_train, y_train, X_val, y_val,
    ...                        epochs=100, nombre="dense_V30_H5")
    >>> # Ahora puedes plotear la curva:
    >>> from src.plotting import plot_curva_entrenamiento
    >>> plot_curva_entrenamiento(hist, title="Dense V=30 H=5")
    """
    fijar_semilla(seed)  # Asegura reproducibilidad en cada entrenamiento

    checkpoint_path = CHECKPOINTS_DIR / f"{nombre}.keras"

    callbacks = [
        # Guarda el mejor modelo en disco según val_loss
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
        # Para el entrenamiento si no mejora en 'patience' épocas
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,  # al parar, recupera el mejor modelo
            verbose=1,
        ),
        # Reduce el lr cuando val_loss se estanca
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,        # divide el lr a la mitad
            patience=5,        # espera 5 épocas antes de bajar el lr
            min_lr=1e-6,       # lr mínimo, nunca baja de aquí
            verbose=0,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )

    return history


def entrenar_todos_los_modelos(
    builders: dict,
    returns,
    input_windows: list = [5, 10, 30, 90],
    output_windows: list = [1, 5, 30, 90],
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 15,
    verbose: int = 0,
) -> list[dict]:
    """
    Entrena múltiples modelos en todas las combinaciones de ventanas.

    Recorre las 16 combinaciones (V, H) y para cada una instancia y entrena
    cada modelo definido en builders. Guarda todos los resultados en una
    lista de dicts compatible con construir_matriz_resultados() de evaluation.py.

    Parameters
    ----------
    builders : dict
        Diccionario {nombre: función_builder} donde cada función_builder
        acepta input_shape como primer argumento y devuelve un modelo compilado.

        Ejemplo:
        builders = {
            "MLP_pequeño":  lambda s: build_dense_model(s, [32]),
            "MLP_mediano":  lambda s: build_dense_model(s, [64, 32]),
            "LSTM":         lambda s: build_lstm_model(s, [64]),
        }

    returns : pd.DataFrame
        Los retornos logarítmicos cargados con cargar_returns().
    input_windows : list
        Ventanas de entrada a probar.
    output_windows : list
        Ventanas de salida a probar.
    epochs : int
        Épocas máximas por entrenamiento.
    batch_size : int
        Tamaño del batch.
    patience : int
        Patience del EarlyStopping.
    verbose : int
        Verbosidad del entrenamiento (0 = silencioso).

    Returns
    -------
    list[dict]
        Lista de resultados. Cada dict tiene:
        - modelo: nombre del modelo
        - V: ventana de entrada
        - H: ventana de salida
        - MAE_train, MAE_val, MAE_test: métricas
        - n_params: número de parámetros del modelo
        - epocas_entrenadas: cuántas épocas antes del early stopping

    Notes
    -----
    Esta función puede tardar bastante (varios minutos) dependiendo del número
    de modelos y combinaciones. Usa verbose=1 si quieres ver el progreso.
    """
    from src.data import preparar_datos
    from src.evaluation import mae_global

    resultados = []
    combinaciones = [(V, H) for V in input_windows for H in output_windows]
    total = len(combinaciones) * len(builders)
    contador = 0

    for V, H in combinaciones:
        datos = preparar_datos(returns, V, H, verbose=False)
        X_tr  = datos["X_train"];  y_tr  = datos["y_train"]
        X_val = datos["X_val"];    y_val = datos["y_val"]
        X_te  = datos["X_test"];   y_te  = datos["y_test"]

        input_shape = (V, X_tr.shape[2])  # (V, 23)

        for nombre_modelo, builder_fn in builders.items():
            contador += 1
            nombre_completo = f"{nombre_modelo}_V{V}_H{H}"
            print(f"[{contador}/{total}] Entrenando {nombre_completo}...")

            # Instanciar modelo fresco (pesos aleatorios nuevos)
            model = builder_fn(input_shape)

            # Entrenar
            hist = entrenar_modelo(
                model, X_tr, y_tr, X_val, y_val,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
                nombre=nombre_completo,
                verbose=verbose,
            )

            epocas = len(hist.history["loss"])

            resultados.append({
                "modelo":            nombre_modelo,
                "V":                 V,
                "H":                 H,
                "MAE_train":         mae_global(y_tr,  model.predict(X_tr,  verbose=0)),
                "MAE_val":           mae_global(y_val, model.predict(X_val, verbose=0)),
                "MAE_test":          mae_global(y_te,  model.predict(X_te,  verbose=0)),
                "n_params":          int(model.count_params()),
                "epocas_entrenadas": epocas,
            })

            print(f"    MAE test={resultados[-1]['MAE_test']:.5f}  "
                  f"params={resultados[-1]['n_params']}  "
                  f"épocas={epocas}")

    return resultados


class EarlyStoppingConMinimo(keras.callbacks.Callback):
    """
    Early stopping que garantiza un mínimo de épocas antes de parar.

    A diferencia del EarlyStopping estándar, no para antes de
    ``min_epochs`` épocas. Útil con lr=1e-5 donde el modelo
    tarda más en converger.

    Parameters
    ----------
    monitor : str       Métrica a monitorizar (default 'val_loss').
    patience : int      Épocas sin mejora antes de parar (tras min_epochs).
    min_epochs : int    Épocas mínimas antes de activar el ES.
    min_delta : float   Mejora mínima que cuenta como mejora real.
    """
    def __init__(self, monitor="val_loss", patience=30,
                 min_epochs=100, min_delta=1e-5):
        super().__init__()
        self.monitor    = monitor
        self.patience   = patience
        self.min_epochs = min_epochs
        self.min_delta  = min_delta
        self.mejor_val  = np.inf
        self.contador   = 0
        self.mejor_pesos = None

    def on_epoch_end(self, epoch, logs=None):
        val_actual = logs.get(self.monitor, np.inf)
        if val_actual < self.mejor_val - self.min_delta:
            self.mejor_val   = val_actual
            self.contador    = 0
            self.mejor_pesos = self.model.get_weights()
        else:
            self.contador += 1
        if epoch + 1 >= self.min_epochs and self.contador >= self.patience:
            print(f"\nEarlyStopping en época {epoch+1} "
                  f"(min={self.min_epochs}, patience={self.patience})")
            if self.mejor_pesos is not None:
                self.model.set_weights(self.mejor_pesos)
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.mejor_pesos is not None:
            self.model.set_weights(self.mejor_pesos)


def entrenar_modelo_lr5(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 300,
    batch_size: int = 32,
    patience: int = 30,
    min_epochs: int = 100,
    nombre: str = "modelo",
    verbose: int = 0,
    seed: int = 42,
    lr: float = 1e-5,
):
    """
    Entrena con lr bajo (1e-5) y EarlyStoppingConMinimo.

    Garantiza min_epochs de entrenamiento antes de activar el ES.
    Recompila el modelo con el lr especificado.
    """
    import tensorflow as tf
    fijar_semilla(seed)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mean_absolute_error",
        metrics=["mean_absolute_error"],
    )

    checkpoint_path = CHECKPOINTS_DIR / f"{nombre}.keras"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
        EarlyStoppingConMinimo(
            monitor="val_loss",
            patience=patience,
            min_epochs=min_epochs,
            min_delta=1e-5,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=0,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )

    return history