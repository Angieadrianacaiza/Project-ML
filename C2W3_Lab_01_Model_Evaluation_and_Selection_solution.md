# Solución guiada: Optional Lab — Model Evaluation and Selection

Este documento recorre y resuelve los ejercicios del laboratorio "Model Evaluation and Selection" con código completo (listo para ejecutar) y explicaciones en español. Está diseñado para ejecutarse en un Jupyter Notebook o como script (adaptando la parte de gráficas si es necesario).

Dependencias:
- Python 3.7+
- numpy, matplotlib, scikit-learn, tensorflow
- (Opcional) jupyter, jupyterlab

Instalación rápida (si no están instaladas):
```bash
pip install numpy matplotlib scikit-learn tensorflow
```

Estructura:
1. Imports y utilidades (funciones de plotting y `build_models`)
2. Parte de regresión
   - Carga de datos
   - División en train/cv/test
   - Regresión lineal (escalado y evaluación)
   - Añadir características polinómicas y elegir grado
   - Redes neuronales (comparar arquitecturas)
3. Parte de clasificación
   - Carga y preparación de datos
   - Entrenamiento de redes y evaluación de errores de clasificación
4. Comentarios finales

---

## 1) Imports y utilidades

Primero importamos librerías y definimos funciones auxiliares (plots y `build_models`) para que el notebook sea autocontenido.

```python
# imports y utilidades
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf

# reproducibilidad básica
np.random.seed(1)
tf.random.set_seed(1)

# -------------------------
# Funciones auxiliares (utils)
# -------------------------
def plot_dataset(x, y, title="dataset"):
    plt.figure(figsize=(6,4))
    plt.scatter(x.squeeze(), y.squeeze(), c='C0', edgecolor='k')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_train_cv_test(x_train, y_train, x_cv, y_cv, x_test, y_test, title="train/cv/test"):
    plt.figure(figsize=(6,4))
    plt.scatter(x_train.squeeze(), y_train.squeeze(), label='train', marker='o', s=50)
    plt.scatter(x_cv.squeeze(), y_cv.squeeze(), label='cv', marker='s', s=50)
    plt.scatter(x_test.squeeze(), y_test.squeeze(), label='test', marker='^', s=50)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_train_cv_mses(degrees, train_mses, cv_mses, title="degree vs mse"):
    plt.figure(figsize=(7,4))
    plt.plot(degrees, train_mses, marker='o', label='Train MSE')
    plt.plot(degrees, cv_mses, marker='o', label='CV MSE')
    plt.xlabel('Degree of polynomial')
    plt.ylabel('MSE (note: divided by 2 as in lab)')
    plt.title(title)
    plt.xticks(degrees)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_bc_dataset(x, y, title="binary classification"):
    plt.figure(figsize=(6,5))
    y_flat = y.squeeze()
    plt.scatter(x[y_flat==0,0], x[y_flat==0,1], label='class 0', alpha=0.7, edgecolor='k')
    plt.scatter(x[y_flat==1,0], x[y_flat==1,1], label='class 1', alpha=0.7, edgecolor='k')
    plt.xlabel('x1'); plt.ylabel('x2')
    plt.title(title)
    plt.legend(); plt.grid(True)
    plt.show()

def build_models():
    """Construye y devuelve una lista de modelos Keras (regresión/ clasificación con salida lineal).
    Usamos varios tamaños para comparar. Nombres legibles asignados manualmente.
    """
    models = []

    # Modelo 1: pequeño (1 hidden layer)
    m1 = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(None,)) , # placeholder, set más adelante por input_shape real
    ])
    # En lugar de Input(shape=(None,)) lo ajustaremos al compilar/usar: crearemos con input_shape dinámico abajo.
    # Para facilitar la construcción, definimos versiones con input_shape=(n_features,)
    def _mk_model(hidden_units_list, name):
        model = tf.keras.Sequential(name=name)
        model.add(tf.keras.layers.InputLayer(input_shape=(None,)))  # placeholder, se remplaza en build por Keras
        # We'll rebuild below; to avoid complexity, we'll recreate with proper shapes when called.
        return model

    # Para evitar problemas con InputLayer placeholder, construiremos modelos concretos
    # en el caller donde sabemos la dimensión. Aquí devolvemos funciones que crean modelos dados input_dim.
    def model_factory_1(input_dim):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(8, activation='tanh'),
            tf.keras.layers.Dense(1, activation='linear')
        ], name='nn_small')
        return model

    def model_factory_2(input_dim):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(32, activation='tanh'),
            tf.keras.layers.Dense(8, activation='tanh'),
            tf.keras.layers.Dense(1, activation='linear')
        ], name='nn_medium')
        return model

    def model_factory_3(input_dim):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(32, activation='tanh'),
            tf.keras.layers.Dense(8, activation='tanh'),
            tf.keras.layers.Dense(1, activation='linear')
        ], name='nn_large')
        return model

    # Devolvemos las "factorías" para construir con input_dim conocido
    return [model_factory_1, model_factory_2, model_factory_3]
```

Explicación:
- Creamos funciones de plot usadas en el lab.
- `build_models()` devuelve fábricas (funciones) que crean modelos dados el número de features (input_dim). Hacemos esto para poder usar tanto en regresión como en clasificación (mismo número de entradas). Los modelos tienen capa de salida lineal (tal como recomienda el laboratorio: usar `from_logits=True` en la pérdida para clasificación).

---

## 2) Regresión

### 2.1 Carga de datos
Asegúrate que el archivo `./data/data_w3_ex1.csv` exista en la ruta. Si lo bajaste del curso, ponlo en `./data/`.

```python
# Cargar datos
data = np.loadtxt('./data/data_w3_ex1.csv', delimiter=',')
x = data[:,0]
y = data[:,1]
# convertir a 2D
x = np.expand_dims(x, axis=1)
y = np.expand_dims(y, axis=1)

print("x shape:", x.shape)
print("y shape:", y.shape)
plot_dataset(x, y, title="input vs target (raw)")
```

Explicación:
- El dataset tiene 50 ejemplos (según el enunciado original). Visualizamos la relación x→y.

---

### 2.2 División train / cross-validation / test (60/20/20)

```python
x_train, x_, y_train, y_ = train_test_split(x, y, test_size=0.40, random_state=1)
x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)
del x_, y_

print("train:", x_train.shape, y_train.shape)
print("cv:", x_cv.shape, y_cv.shape)
print("test:", x_test.shape, y_test.shape)

plot_train_cv_test(x_train, y_train, x_cv, y_cv, x_test, y_test, title="train / cv / test split")
```

Explicación:
- `train_test_split` se usa dos veces para obtener 60/20/20. `random_state=1` fija la semilla de la división.

---

### 2.3 Regresión lineal (escalado y entrenamiento)

Escalamos usando la media y desviación calculadas en el training set.

```python
# Escalado
scaler_linear = StandardScaler()
X_train_scaled = scaler_linear.fit_transform(x_train)
print("train mean:", scaler_linear.mean_.squeeze(), "scale:", scaler_linear.scale_.squeeze())
plot_dataset(X_train_scaled, y_train, title="scaled input vs target (train)")

# Entrenamiento modelo lineal
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Predicción y MSE (dividimos por 2 como en el lab)
yhat_train = linear_model.predict(X_train_scaled)
train_mse_sklearn = mean_squared_error(y_train, yhat_train) / 2
print("Training MSE (sklearn/2):", train_mse_sklearn)

# Implementación por bucle (comprobación)
total_squared_error = np.sum((yhat_train - y_train)**2)
mse_loop = total_squared_error / (2 * len(yhat_train))
print("Training MSE (loop):", mse_loop.squeeze())
```

Explicación:
- Escalamos x con `StandardScaler`.
- Entrenamos `LinearRegression()` usando los valores escalados.
- Calculamos MSE y lo dividimos por 2 (convenio del lab).

---

### 2.4 Evaluación en CV (usar scaler del training)

```python
X_cv_scaled = scaler_linear.transform(x_cv)
yhat_cv = linear_model.predict(X_cv_scaled)
cv_mse = mean_squared_error(y_cv, yhat_cv) / 2
print("Cross validation MSE (linear):", cv_mse)
```

Explicación:
- Es crítico usar la media/desviación del training set para transformar CV y test.

---

### 2.5 Añadir características polinómicas (grado 2 como ejemplo)

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_mapped = poly.fit_transform(x_train)
print("Primeros 5 (x, x^2):")
print(X_train_mapped[:5])

# Escalado sobre las nuevas características
scaler_poly = StandardScaler()
X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)
print("Primeros 5 escalados:")
print(X_train_mapped_scaled[:5])

# Entrenar y evaluar
model_poly2 = LinearRegression()
model_poly2.fit(X_train_mapped_scaled, y_train)
yhat_train_poly2 = model_poly2.predict(X_train_mapped_scaled)
print("Training MSE (poly deg 2):", mean_squared_error(y_train, yhat_train_poly2)/2)

# CV: transformar y evaluar
X_cv_mapped = poly.transform(x_cv)
X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)
yhat_cv_poly2 = model_poly2.predict(X_cv_mapped_scaled)
print("CV MSE (poly deg 2):", mean_squared_error(y_cv, yhat_cv_poly2)/2)
```

Explicación:
- `PolynomialFeatures` crea columnas x, x^2, ..., x^degree.
- Escalamos las columnas polinómicas (usamos scaler ajustado al training set).

---

### 2.6 Experimentar grados 1..10 y graficar train/CV MSEs

```python
train_mses = []
cv_mses = []
models = []
polys = []
scalers = []

for degree in range(1, 11):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_mapped = poly.fit_transform(x_train)
    polys.append(poly)
    scaler_poly = StandardScaler()
    X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)
    scalers.append(scaler_poly)

    model = LinearRegression()
    model.fit(X_train_mapped_scaled, y_train)
    models.append(model)

    # train mse
    yhat_train = model.predict(X_train_mapped_scaled)
    train_mse = mean_squared_error(y_train, yhat_train) / 2
    train_mses.append(train_mse)

    # cv mse
    X_cv_mapped = poly.transform(x_cv)
    X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)
    yhat_cv = model.predict(X_cv_mapped_scaled)
    cv_mse = mean_squared_error(y_cv, yhat_cv) / 2
    cv_mses.append(cv_mse)

degrees = list(range(1, 11))
plot_train_cv_mses(degrees, train_mses, cv_mses, title="degree vs train/CV MSE")
```

Explicación:
- Repetimos pipeline (mapear, escalar, entrenar, evaluar) para grados 1 a 10.
- Graficamos curvas de aprendizaje por grado para comparar bias/variance implícito.

---

### 2.7 Elegir el mejor modelo por CV y evaluar en test

```python
best_degree = int(np.argmin(cv_mses) + 1)
print("Mejor grado (min CV MSE):", best_degree)

# obtener test MSE del modelo elegido
best_idx = best_degree - 1
X_test_mapped = polys[best_idx].transform(x_test)
X_test_mapped_scaled = scalers[best_idx].transform(X_test_mapped)
yhat_test = models[best_idx].predict(X_test_mapped_scaled)
test_mse = mean_squared_error(y_test, yhat_test) / 2

print(f"Training MSE (grado {best_degree}): {train_mses[best_idx]:.4f}")
print(f"CV MSE (grado {best_degree}): {cv_mses[best_idx]:.4f}")
print(f"Test MSE (grado {best_degree}): {test_mse:.4f}")
```

Explicación:
- Se elige el modelo con menor error en CV. Test MSE estima la generalización del modelo elegido.

---

## 3) Redes neuronales (regresión)

Usaremos las factorías de modelos devueltas por `build_models()`. Necesitamos construir modelos con `input_dim` correcto.

```python
# Preparación: usar grado=1 (sin polinomios extra) como en el lab (las redes pueden aprender no linealidad)
degree = 1
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_train_mapped = poly.fit_transform(x_train)
X_cv_mapped = poly.transform(x_cv)
X_test_mapped = poly.transform(x_test)

# Escalado
scaler = StandardScaler()
X_train_mapped_scaled = scaler.fit_transform(X_train_mapped)
X_cv_mapped_scaled = scaler.transform(X_cv_mapped)
X_test_mapped_scaled = scaler.transform(X_test_mapped)

# Obtener las factorías
model_factories = build_models()

nn_train_mses = []
nn_cv_mses = []
nn_models = []

# Entrenar cada arquitectura
for factory in model_factories:
    input_dim = X_train_mapped_scaled.shape[1]
    model = factory(input_dim)  # construir con input_dim
    # Compilación (regresión con MSE)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.1))
    print(f"Entrenando {model.name} ...")
    model.fit(X_train_mapped_scaled, y_train, epochs=300, verbose=0)
    print("Hecho.")

    # registrar
    yhat_train = model.predict(X_train_mapped_scaled)
    train_mse = mean_squared_error(y_train, yhat_train) / 2
    nn_train_mses.append(train_mse)

    yhat_cv = model.predict(X_cv_mapped_scaled)
    cv_mse = mean_squared_error(y_cv, yhat_cv) / 2
    nn_cv_mses.append(cv_mse)

    nn_models.append(model)

# Mostrar resultados
for i, (tr, cv) in enumerate(zip(nn_train_mses, nn_cv_mses), start=1):
    print(f"Model {i} ({nn_models[i-1].name}): Train MSE={tr:.4f}, CV MSE={cv:.4f}")
```

Explicación:
- Probamos varias arquitecturas; las redes usan `activation='tanh'` en internas y salida lineal.
- Ajusta el `learning_rate` o `epochs` si convergencia es inestable.
- Comparamos train y cv MSE para decidir arquitectura.

Seleccionamos la mejor por CV y calculamos test MSE:

```python
best_model_idx = int(np.argmin(nn_cv_mses))
best_model = nn_models[best_model_idx]
print("Modelo seleccionado (por CV):", best_model.name, "index:", best_model_idx+1)

# calcular test MSE
yhat_test = best_model.predict(X_test_mapped_scaled)
test_mse_nn = mean_squared_error(y_test, yhat_test) / 2
print(f"Train MSE: {nn_train_mses[best_model_idx]:.4f}")
print(f"CV MSE: {nn_cv_mses[best_model_idx]:.4f}")
print(f"Test MSE: {test_mse_nn:.4f}")
```

Observaciones:
- Si un modelo sobreajusta, verá train MSE bajo y CV alto.
- Si hay underfitting ambos errores serán altos.

---

## 4) Clasificación (binary) — evaluación y selección

### 4.1 Carga y visualización del dataset

```python
data_bc = np.loadtxt('./data/data_w3_ex2.csv', delimiter=',')
x_bc = data_bc[:,:-1]
y_bc = data_bc[:,-1]
y_bc = np.expand_dims(y_bc, axis=1)

print("x_bc shape:", x_bc.shape)
print("y_bc shape:", y_bc.shape)
plot_bc_dataset(x_bc, y_bc, title="Binary classification dataset")
```

Explicación:
- Dataset con 200 ejemplos y 2 features (x1,x2) y etiqueta 0/1.

---

### 4.2 División y escalado (60/20/20)

```python
x_bc_train, x_, y_bc_train, y_ = train_test_split(x_bc, y_bc, test_size=0.40, random_state=1)
x_bc_cv, x_bc_test, y_bc_cv, y_bc_test = train_test_split(x_, y_, test_size=0.50, random_state=1)
del x_, y_

scaler_bc = StandardScaler()
x_bc_train_scaled = scaler_bc.fit_transform(x_bc_train)
x_bc_cv_scaled = scaler_bc.transform(x_bc_cv)
x_bc_test_scaled = scaler_bc.transform(x_bc_test)
```

Explicación:
- Igual procedimiento de particionado y escalado que en regresión.

---

### 4.3 Evaluar error de clasificación — ejemplo simple

```python
# ejemplo de thresholding
probabilities = np.array([0.2, 0.6, 0.7, 0.3, 0.8])
predictions = np.where(probabilities >= 0.5, 1, 0)
ground_truth = np.array([1,1,1,1,1])
fraction_error = np.mean(predictions != ground_truth)
print("fraction_error:", fraction_error)
```

Explicación:
- Para clasificación binaria, usamos la fracción de ejemplos mal clasificados como métrica (error de clasificación).

---

### 4.4 Entrenar modelos (clasificación) y calcular errores

Usamos las mismas factorías de modelos, esta vez con la salida lineal y entrenando con `BinaryCrossentropy(from_logits=True)`. Tras predecir aplicamos sigmoid y threshold.

```python
# Obtener factorías
model_factories = build_models()

nn_train_error = []
nn_cv_error = []
models_bc = []

threshold = 0.5

for factory in model_factories:
    input_dim = x_bc_train_scaled.shape[1]
    model = factory(input_dim)
    # loss recomendada: BinaryCrossentropy(from_logits=True) porque la salida es 'linear'
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)
    )
    print(f"Entrenando (clasificación) {model.name} ...")
    model.fit(x_bc_train_scaled, y_bc_train, epochs=200, verbose=0)
    print("Hecho.")

    # predecir train
    yhat_train_logits = model.predict(x_bc_train_scaled)
    yhat_train_prob = tf.math.sigmoid(yhat_train_logits).numpy()
    yhat_train_pred = np.where(yhat_train_prob >= threshold, 1, 0)
    train_error = np.mean(yhat_train_pred != y_bc_train)
    nn_train_error.append(train_error)

    # predecir CV
    yhat_cv_logits = model.predict(x_bc_cv_scaled)
    yhat_cv_prob = tf.math.sigmoid(yhat_cv_logits).numpy()
    yhat_cv_pred = np.where(yhat_cv_prob >= threshold, 1, 0)
    cv_error = np.mean(yhat_cv_pred != y_bc_cv)
    nn_cv_error.append(cv_error)

    models_bc.append(model)

# mostrar resultados
for i, (tr, cv) in enumerate(zip(nn_train_error, nn_cv_error), start=1):
    print(f"Model {i} ({models_bc[i-1].name}): Train err={tr:.4f}, CV err={cv:.4f}")
```

Explicación:
- La convención del lab es salida lineal + `from_logits=True`. Luego aplica sigmoid al output para obtener probabilidades.
- El umbral es 0.5 por defecto.

---

### 4.5 Elegir mejor modelo y medir test error

```python
best_model_idx = int(np.argmin(nn_cv_error))
best_model = models_bc[best_model_idx]
print("Modelo seleccionado (clasificación):", best_model.name, "index:", best_model_idx+1)

# Test error
yhat_test_logits = best_model.predict(x_bc_test_scaled)
yhat_test_prob = tf.math.sigmoid(yhat_test_logits).numpy()
yhat_test_pred = np.where(yhat_test_prob >= threshold, 1, 0)
test_error = np.mean(yhat_test_pred != y_bc_test)

print(f"Train error: {nn_train_error[best_model_idx]:.4f}")
print(f"CV error: {nn_cv_error[best_model_idx]:.4f}")
print(f"Test error: {test_error:.4f}")
```

Explicación:
- Seleccionamos por menor error en CV; si existe empate, se podrían aplicar criterios adicionales (menor train error o modelo más pequeño).

---

## 5) Comentarios finales y cómo ejecutar

- Este notebook/markdown incluye todo el pipeline y funciones de utilidad. Para obtener los valores numéricos (MSEs, errores, gráficos) ejecuta las celdas en un Jupyter Notebook donde estén presentes los archivos:
  - `./data/data_w3_ex1.csv`
  - `./data/data_w3_ex2.csv`
- Si no tienes `data/`, descarga los CSV desde el recurso del curso y colócalos en `./data/`.
- Ajustes recomendados:
  - Si ves que las redes no convergen bien, reduce la learning rate (ej. 0.01 o 0.001) o aumenta epochs.
  - Para más robustez, se puede guardar los modelos y/o usar callbacks (EarlyStopping) para no sobreentrenar.
- Interpretación:
  - La comparación entre modelos se hace mirando train vs CV: si CV >> train → overfitting; si ambos altos → underfitting.
  - Finalmente, usar test set solo para reporte final del modelo elegido.

Si quieres, puedo:
- Ejecutar el notebook localmente por ti y pegar aquí los números y plots (pero necesitarás subir los archivos CSV o darme los datos).
- Convertir esto a un archivo .ipynb listo para descargar (te lo puedo mostrar como contenido JSON o como un archivo que puedas copiar/pegar).
- Ajustar arquitecturas, lr, o añadir regularización (L2, dropout) y re-ejecutar.

¿Cómo prefieres continuar? ¿Quieres que genere y entregue el archivo .ipynb listo para descargar, o prefieres que ejecute el código y devuelva los resultados si subes los CSV? 