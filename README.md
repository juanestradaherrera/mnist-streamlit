# mnist-streamlit
Clasificación de dígitos MNIST con Streamlit

# Clasificación de Dígitos MNIST con Streamlit

Este proyecto implementa una aplicación web interactiva desarrollada en **Streamlit** para la clasificación de dígitos manuscritos utilizando el dataset **Digits de sklearn (tipo MNIST)**.  
La aplicación permite evaluar diferentes modelos de clasificación, comparar su desempeño con y sin reducción de dimensionalidad mediante **PCA**, y realizar predicciones a partir de un dígito dibujado manualmente por el usuario.

---

## 🎯 Objetivo del proyecto

Desarrollar y desplegar una aplicación de Machine Learning que permita:

- Verificar la **calidad de los datos**
- Entrenar y evaluar **múltiples modelos de clasificación**
- Comparar el desempeño **con y sin PCA**
- Probar diferentes **porcentajes de entrenamiento y prueba**
- Evaluar modelos con **validación cruzada**
- Permitir la **predicción interactiva** de dígitos dibujados con el mouse

---

## 📊 Dataset

- **Fuente:** `sklearn.datasets.load_digits`
- **Descripción:** Imágenes de dígitos manuscritos (0–9)
- **Resolución:** 8×8 píxeles
- **Número de muestras:** 1,797
- **Número de clases:** 10
- **Número de características:** 64 (flatten de 8×8)

El dataset no contiene valores faltantes y presenta una distribución balanceada entre clases.

---

## 🤖 Modelos de clasificación implementados

La aplicación permite entrenar y comparar los siguientes modelos:

- Naive Bayes (GaussianNB)
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM – kernel RBF)
- Random Forest
- Decision Tree
- Logistic Regression

Cada modelo es entrenado usando un **pipeline** que incluye estandarización y, opcionalmente, reducción de dimensionalidad con PCA.

---

## 🔎 Reducción de dimensionalidad (PCA)

La aplicación permite activar o desactivar **PCA (Principal Component Analysis)** mediante un checkbox.  
Cuando está activo:
- Se selecciona el número de componentes según el **porcentaje de varianza explicada**
- Se evalúa el impacto del PCA sobre el desempeño y el sobreajuste del modelo

---

## 🔁 Validación cruzada

Se implementan distintas estrategias de validación cruzada para evaluar la estabilidad de los modelos:

- Stratified K-Fold
- K-Fold
- Repeated Stratified K-Fold
- Stratified Shuffle Split

Se reportan métricas promedio y desviación estándar para:
- Accuracy
- F1-score (ponderado)

---

## 📈 Métricas y visualizaciones

La aplicación muestra:

- Accuracy en entrenamiento y prueba
- Gráficas comparativas Train vs Test
- Matriz de confusión
- Reporte de clasificación
- Resultados por fold en validación cruzada

---

## ✏️ Predicción interactiva

El usuario puede dibujar un dígito usando el mouse directamente en la aplicación.  
El dibujo es:
1. Convertido a escala de grises
2. Redimensionado a 8×8 píxeles
3. Normalizado para coincidir con el formato del dataset
4. Clasificado por el modelo entrenado

---

## 🌐 Despliegue

La aplicación está desplegada usando **Streamlit Cloud**, conectada directamente a este repositorio de GitHub.

🔗 **Link de la aplicación:**  
