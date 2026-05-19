"""
src/models.py
==============

Definición de las arquitecturas de redes neuronales para el taller.

Cada función build_* devuelve un modelo de Keras ya compilado con
MAE como loss y Adam como optimizador.

Arquitecturas disponibles
--------------------------
Densas:     build_dense_model
Recurrentes: build_lstm_model, build_gru_model
Conv:       build_conv1d_model
Inception:  build_inception_1d_model
Mixtas:     build_conv_lstm_model, build_conv_dense_model

Notas de diseño
---------------
- La capa de salida siempre es Dense(n_outputs) SIN activación.
  Los retornos pueden ser positivos o negativos sin rango acotado.
- Dropout solo en capas intermedias, nunca en la capa de salida.
- Conv1D usa padding='causal': solo mira el pasado, nunca el futuro.
"""

from __future__ import annotations
import keras
from keras import layers, Input, Model


def _compilar(model: Model, lr: float) -> Model:
    """Compila el modelo con MAE y Adam. Uso interno."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mean_absolute_error",
        metrics=["mean_absolute_error"],
    )
    return model


def contar_parametros(model: Model) -> int:
    """Devuelve el número total de parámetros entrenables."""
    return int(model.count_params())


# ─────────────────────────────────────────────────────────────
# MODELOS DENSOS (MLP)
# ─────────────────────────────────────────────────────────────

def build_dense_model(
    input_shape: tuple,
    hidden_units: list = [64, 32],
    dropout: float = 0.0,
    n_outputs: int = 23,
    lr: float = 1e-3,
) -> Model:
    """
    MLP configurable en número de capas y neuronas.

    Aplana la entrada (V, 23) a un vector de V*23 features y pasa
    por capas densas con activación ReLU.

    Parameters
    ----------
    input_shape : tuple  ej. (30, 23)
    hidden_units : list  neuronas por capa, ej. [64, 32] → 2 capas
    dropout : float      tasa de dropout entre capas (0.0 = sin dropout)
    n_outputs : int      número de salidas (por defecto 23)
    lr : float           learning rate de Adam
    """
    inputs = Input(shape=input_shape, name="input")
    x = layers.Flatten(name="flatten")(inputs)

    for i, units in enumerate(hidden_units):
        x = layers.Dense(units, activation="relu", name=f"dense_{i+1}")(x)
        if dropout > 0.0:
            x = layers.Dropout(dropout, name=f"dropout_{i+1}")(x)

    outputs = layers.Dense(n_outputs, name="output")(x)
    model = Model(inputs=inputs, outputs=outputs, name="MLP")
    return _compilar(model, lr)


# ─────────────────────────────────────────────────────────────
# MODELOS RECURRENTES
# ─────────────────────────────────────────────────────────────

def build_lstm_model(
    input_shape: tuple,
    units: list = [64],
    dropout: float = 0.0,
    recurrent_dropout: float = 0.0,
    n_outputs: int = 23,
    lr: float = 1e-3,
) -> Model:
    """
    Red LSTM apilable en varias capas.

    Parameters
    ----------
    input_shape : tuple  ej. (30, 23)
    units : list         neuronas por capa LSTM, ej. [64] o [64, 32]
    dropout : float      dropout sobre la entrada de cada capa
    recurrent_dropout    dropout sobre la conexión recurrente (úsalo bajo)
    n_outputs : int      número de salidas
    lr : float           learning rate

    Notes
    -----
    Si hay varias capas, todas menos la última usan return_sequences=True
    para pasar la secuencia completa a la siguiente capa LSTM.
    """
    inputs = Input(shape=input_shape, name="input")
    x = inputs

    for i, n_units in enumerate(units):
        return_seq = (i < len(units) - 1)
        x = layers.LSTM(
            n_units,
            return_sequences=return_seq,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            name=f"lstm_{i+1}",
        )(x)

    outputs = layers.Dense(n_outputs, name="output")(x)
    model = Model(inputs=inputs, outputs=outputs, name="LSTM")
    return _compilar(model, lr)


def build_gru_model(
    input_shape: tuple,
    units: list = [64],
    dropout: float = 0.0,
    recurrent_dropout: float = 0.0,
    n_outputs: int = 23,
    lr: float = 1e-3,
) -> Model:
    """
    Red GRU: alternativa a LSTM con menos parámetros (2 gates vs 3).
    Suele dar resultados comparables entrenando más rápido.

    Parameters
    ----------
    input_shape : tuple  ej. (30, 23)
    units : list         neuronas por capa GRU
    dropout : float      dropout sobre la entrada
    recurrent_dropout    dropout sobre la conexión recurrente
    n_outputs : int      número de salidas
    lr : float           learning rate
    """
    inputs = Input(shape=input_shape, name="input")
    x = inputs

    for i, n_units in enumerate(units):
        return_seq = (i < len(units) - 1)
        x = layers.GRU(
            n_units,
            return_sequences=return_seq,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            name=f"gru_{i+1}",
        )(x)

    outputs = layers.Dense(n_outputs, name="output")(x)
    model = Model(inputs=inputs, outputs=outputs, name="GRU")
    return _compilar(model, lr)


# ─────────────────────────────────────────────────────────────
# MODELOS CONVOLUCIONALES
# ─────────────────────────────────────────────────────────────

def build_conv1d_model(
    input_shape: tuple,
    filters: list = [64, 32],
    kernel_size: int = 3,
    dropout: float = 0.0,
    n_outputs: int = 23,
    lr: float = 1e-3,
) -> Model:
    """
    Red convolucional 1D para series temporales.

    Conv1D aplica filtros deslizantes sobre la dimensión temporal para
    detectar patrones locales (momentum de 3 días, reversión de 5 días).
    GlobalAveragePooling agrega la info temporal antes de la capa final.

    Parameters
    ----------
    input_shape : tuple  ej. (30, 23)
    filters : list       filtros por capa Conv1D, ej. [64, 32]
    kernel_size : int    tamaño del kernel (ventana local que mira cada filtro)
    dropout : float      dropout entre capas convolucionales
    n_outputs : int      número de salidas
    lr : float           learning rate

    Notes
    -----
    padding='causal' garantiza que cada posición temporal solo ve el pasado.
    Esto evita leakage temporal, crítico en datos financieros.
    """
    inputs = Input(shape=input_shape, name="input")
    x = inputs

    for i, n_filters in enumerate(filters):
        x = layers.Conv1D(
            n_filters,
            kernel_size=kernel_size,
            activation="relu",
            padding="same", 
            name=f"conv1d_{i+1}",
        )(x)
        x = layers.MaxPooling1D(pool_size=2, padding="same", name=f"maxpool_{i+1}")(x)
        if dropout > 0.0:
            x = layers.Dropout(dropout, name=f"dropout_{i+1}")(x)

    x = layers.Flatten(name="flatten")(x)
    outputs = layers.Dense(n_outputs, name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Conv1D")
    return _compilar(model, lr)


# ─────────────────────────────────────────────────────────────
# MODELOS MIXTOS
# ─────────────────────────────────────────────────────────────

def build_conv_lstm_model(
    input_shape: tuple,
    conv_filters: list = [32],
    kernel_size: int = 3,
    lstm_units: list = [32],
    dropout: float = 0.0,
    n_outputs: int = 23,
    lr: float = 1e-3,
) -> Model:
    """
    Arquitectura mixta Conv1D → LSTM.

    Conv1D extrae patrones locales de corto plazo.
    LSTM captura dependencias temporales más largas sobre esas features.
    Es una de las arquitecturas más comunes en forecasting financiero.

    Parameters
    ----------
    input_shape : tuple    ej. (30, 23)
    conv_filters : list    filtros por capa Conv1D
    kernel_size : int      tamaño del kernel convolucional
    lstm_units : list      neuronas por capa LSTM
    dropout : float        dropout entre bloques
    n_outputs : int        número de salidas
    lr : float             learning rate
    """
    inputs = Input(shape=input_shape, name="input")
    x = inputs

    for i, n_filters in enumerate(conv_filters):
        x = layers.Conv1D(
            n_filters,
            kernel_size=kernel_size,
            activation="relu",
            padding="causal",
            name=f"conv1d_{i+1}",
        )(x)
        if dropout > 0.0:
            x = layers.Dropout(dropout, name=f"dropout_conv_{i+1}")(x)

    for i, n_units in enumerate(lstm_units):
        return_seq = (i < len(lstm_units) - 1)
        x = layers.LSTM(
            n_units,
            return_sequences=return_seq,
            name=f"lstm_{i+1}",
        )(x)
        if dropout > 0.0 and return_seq:
            x = layers.Dropout(dropout, name=f"dropout_lstm_{i+1}")(x)

    outputs = layers.Dense(n_outputs, name="output")(x)
    model = Model(inputs=inputs, outputs=outputs, name="Conv1D_LSTM")
    return _compilar(model, lr)


def build_conv_dense_model(
    input_shape: tuple,
    conv_filters: list = [32],
    kernel_size: int = 3,
    dense_units: list = [32],
    dropout: float = 0.0,
    n_outputs: int = 23,
    lr: float = 1e-3,
) -> Model:
    """
    Arquitectura mixta Conv1D → Dense.

    Conv1D extrae features locales y GlobalAveragePooling las agrega.
    Capas densas refinan la predicción final.

    Parameters
    ----------
    input_shape : tuple    ej. (30, 23)
    conv_filters : list    filtros por capa Conv1D
    kernel_size : int      tamaño del kernel
    dense_units : list     neuronas por capa densa posterior
    dropout : float        dropout entre capas
    n_outputs : int        número de salidas
    lr : float             learning rate
    """
    inputs = Input(shape=input_shape, name="input")
    x = inputs

    for i, n_filters in enumerate(conv_filters):
        x = layers.Conv1D(
            n_filters,
            kernel_size=kernel_size,
            activation="relu",
            padding="causal",
            name=f"conv1d_{i+1}",
        )(x)

    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)

    for i, units in enumerate(dense_units):
        x = layers.Dense(units, activation="relu", name=f"dense_{i+1}")(x)
        if dropout > 0.0:
            x = layers.Dropout(dropout, name=f"dropout_{i+1}")(x)

    outputs = layers.Dense(n_outputs, name="output")(x)
    model = Model(inputs=inputs, outputs=outputs, name="Conv1D_Dense")
    return _compilar(model, lr)

def build_inception_1d_model(
    input_shape: tuple,
    filters_b1: int = 32,
    filters_b2: int = 32,
    kernel_size_b1: int = 3,
    kernel_size_b2: int = 7,
    dropout: float = 0.2,
    noise: float = 0.01,
    n_outputs: int | None = None,
    lr: float = 1e-3,
) -> Model:
    """
    Arquitectura Inception 1D con dos ramas convolucionales paralelas.

    Rama 1 usa un kernel pequeño para capturar patrones de corto plazo,
    rama 2 usa un kernel grande para tendencias de largo plazo. Ambas
    se fusionan mediante concatenación (estilo Inception de Szegedy et al.).

    Parameters
    ----------
    input_shape : tuple       ej. (30, 23)
    filters_b1 : int          filtros de la rama de corto plazo
    filters_b2 : int          filtros de la rama de largo plazo
    kernel_size_b1 : int      kernel de la rama 1 (corto plazo)
    kernel_size_b2 : int      kernel de la rama 2 (largo plazo)
    dropout : float           tasa de dropout antes de la capa de salida
    noise : float             desviación del GaussianNoise (0 = desactivado)
    n_outputs : int | None    número de salidas (None = input_shape[1])
    lr : float                learning rate de Adam
    """
    if n_outputs is None:
        n_outputs = input_shape[1]

    inputs = Input(shape=input_shape, name="input")

    # Inyección de ruido para evitar que memorice datos exactos
    x_input = inputs
    if noise > 0:
        x_input = layers.GaussianNoise(noise, name="gaussian_noise")(x_input)

    # Rama 1: Corto plazo
    branch1 = layers.Conv1D(filters=filters_b1, kernel_size=kernel_size_b1, padding='same', name="conv1d_short")(x_input)
    branch1 = layers.LayerNormalization(name="layernorm_short")(branch1)
    branch1 = layers.Activation('relu', name="relu_short")(branch1)
    branch1 = layers.MaxPooling1D(pool_size=2, name="maxpool_short")(branch1)

    # Rama 2: Largo plazo
    branch2 = layers.Conv1D(filters=filters_b2, kernel_size=kernel_size_b2, padding='same', name="conv1d_long")(x_input)
    branch2 = layers.LayerNormalization(name="layernorm_long")(branch2)
    branch2 = layers.Activation('relu', name="relu_long")(branch2)
    branch2 = layers.MaxPooling1D(pool_size=2, name="maxpool_long")(branch2)

    # Fusionar ramificaciones (Inception)
    merged = layers.Concatenate(axis=-1, name="concat")([branch1, branch2])

    # GlobalAveragePooling1D: más eficiente que Flatten y reduce sobreajuste
    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(merged)

    if dropout > 0:
        x = layers.Dropout(dropout, name="dropout")(x)

    outputs = layers.Dense(n_outputs, name="output")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='Inception_1D_Model')
    
    return _compilar(model, lr)
