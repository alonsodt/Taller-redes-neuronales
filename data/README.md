# Carpeta `data/`

Esta carpeta contiene los datos del proyecto. Los archivos `.parquet` **no se versionan en Git** (están listados en `.gitignore`).

## Cómo regenerar los datos

Ejecutar el notebook `notebooks/00_descarga_datos.ipynb`. Este notebook:

1. Descarga el cierre ajustado de los 23 tickers desde Yahoo Finance (`yfinance`).
2. Calcula retornos logarítmicos diarios.
3. Guarda los archivos en formato Parquet:
   - `data/precios_close.parquet`: precios diarios de cierre.
   - `data/returns.parquet`: retornos logarítmicos diarios.

## Tickers usados

```
AEP, BA, CAT, CNP, CVX, DIS, DTE, ED, GD, GE, HON, HPQ, IBM,
IP, JNJ, KO, KR, MMM, MO, MRK, MSI, PG, XOM
```

## Forma esperada de los datos

- `precios_close.parquet`: DataFrame con índice de fechas y 23 columnas (un activo por columna). ~16.000 filas (días desde 1960).
- `returns.parquet`: similar, pero con retornos logarítmicos. Una fila menos que precios (por la diferenciación).

## Reproducibilidad

Los datos de Yahoo Finance pueden variar ligeramente entre ejecuciones (correcciones retroactivas de dividendos, splits, etc.). Una vez generados los Parquet, **no es necesario volver a descargarlos** salvo que se quiera actualizar a fechas más recientes.
