# 🏡 California Housing Analysis

Proyecto de análisis exploratorio de datos y regresión lineal utilizando el dataset de California Housing.

---

## 📂 Estructura del Proyecto

- `main.py`: Script principal de ejecución.
- `data_loading.py`: Carga del dataset.
- `preprocessing.py`: Limpieza y tratamiento de valores nulos.
- `feature_engineering.py`: Creación de nuevas variables.
- `correlation_analysis.py`: Análisis de correlación.
- `visualizations.py`: Visualización de correlaciones.
- `model.py`: Entrenamiento y evaluación del modelo.
- `Figures/`: Carpeta con gráficos generados automáticamente.

---

## ⚙️ Características

✅ Ingeniería de variables  
✅ Gráfico de correlación  
✅ Modelo de Regresión Lineal Múltiple  
✅ Código modular, limpio y comentado  
✅ Logging informativo paso a paso  

---

## 📉 Resultados del Modelo

- **Modelo:** Regresión lineal múltiple
- **Features:** `median_income`, `housing_median_age`, `rooms_per_household`, `bedrooms_per_room`
- **MSE:** ~4.99e9
- **RMSE:** ~70,642

---

## ▶️ Ejecución

1. Instala las dependencias:

```bash
pip install -r requirements.txt
