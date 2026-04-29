# Taller B3-T4: Redes Neuronales para Forecasting Financiero

Este repositorio contiene el desarrollo del Taller B3-T4 (T5, T6) del Máster en Inteligencia Artificial Aplicada a los Mercados Financieros (BME). El objetivo es comparar distintas arquitecturas de redes neuronales (densas, recurrentes, convolucionales y mixtas) para tareas de *forecasting* sobre retornos diarios de 23 activos del SP500.

## Estructura del repositorio

```
taller-redes-neuronales/
├── data/                    # Datos descargados (no versionados en Git)
├── notebooks/               # Notebooks de análisis y entrenamiento
│   ├── 00_descarga_datos.ipynb
│   ├── 01_baselines.ipynb
│   ├── 02_competicion_densas.ipynb
│   ├── 03_competicion_rnn.ipynb
│   ├── 04_competicion_cnn.ipynb
│   ├── 05_competicion_mixto.ipynb
│   ├── 06_resumen_competicion.ipynb
│   ├── 07_investigacion.ipynb
│   └── 08_carteras_2025.ipynb
├── src/                     # Código reutilizable (módulos Python)
│   ├── data.py              # Carga, ventaneo y splits temporales
│   ├── baselines.py         # Modelos no neuronales (persistencia, regresión...)
│   ├── models.py            # Definición de arquitecturas neuronales
│   ├── training.py          # Loop de entrenamiento estandarizado
│   ├── evaluation.py        # Métricas y comparación de modelos
│   ├── plotting.py          # Gráficas reutilizables
│   └── preprocessing.py     # Diferenciación fraccionaria, etc.
├── results/                 # Resultados generados
│   ├── tables/              # CSVs con MAE por combinación
│   ├── figures/             # PNGs de curvas, heatmaps, etc.
│   └── checkpoints/         # Pesos de modelos (no versionados)
└── presentacion/            # PDF final de la presentación
```

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/alonsodt/taller-redes-neuronales.git
cd taller-redes-neuronales
```

### 2. Crear entorno virtual

En **Windows (PowerShell)**:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

Si PowerShell te bloquea la ejecución de scripts, antes ejecuta:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

La instalación de TensorFlow puede tardar varios minutos.

### 4. Verificar instalación

```python
python -c "import tensorflow as tf; import keras; print('TF:', tf.__version__); print('Keras:', keras.__version__)"
```

### 5. Lanzar Jupyter

```bash
jupyter notebook
```

## Flujo de trabajo

1. **Primero**: ejecutar el notebook `00_descarga_datos.ipynb` para descargar los datos de Yahoo Finance y guardarlos localmente. Solo hace falta hacer esto una vez.
2. **Segundo**: ejecutar `01_baselines.ipynb` para tener los baselines (regresión lineal, Buy & Hold, persistencia) sobre los que comparar.
3. **Después**: los notebooks `02` a `05` entrenan los distintos tipos de redes en las 16 combinaciones de ventanas. Pueden ejecutarse en paralelo (cada uno en su propia ejecución).
4. **`06_resumen_competicion.ipynb`** lee los resultados de los anteriores y genera la matriz final 4×4 con el mejor modelo por casilla.
5. **`07_investigacion.ipynb`** aplica diferenciación fraccionaria (López de Prado) y compara.
6. **`08_carteras_2025.ipynb`** construye las dos carteras pedidas en el enunciado y las compara en 2025.

## Configuración del taller

- **Activos**: 23 tickers del SP500 con histórico desde 1945.
- **Variable objetivo**: promedio de retornos logarítmicos diarios sobre la ventana de salida.
- **Ventanas de entrada (V)**: 5, 10, 30, 90 días.
- **Ventanas de salida (H)**: 1, 5, 30, 90 días.
- **Loss**: MAE (Mean Absolute Error).
- **Partición**: 85% train / 5% val / 10% test, ordenada por tiempo (no shuffle).
- **Semilla**: `RANDOM_SEED = 42` para reproducibilidad.

## Autores

- Alonso (alonsodt)
- *(añadir compañeros del grupo)*

## Notas

- Los datos descargados de Yahoo Finance pueden variar ligeramente entre ejecuciones por correcciones retroactivas (dividendos, splits). El notebook `00` cachea los datos en disco en formato Parquet para garantizar reproducibilidad dentro del proyecto.
- Los pesos de los modelos entrenados (`results/checkpoints/`) no se suben a Git. Para reproducir, ejecutar los notebooks correspondientes localmente.
