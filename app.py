import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image, ImageOps

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from streamlit_drawable_canvas import st_canvas


# -------------------------------
# Configuración general
# -------------------------------
st.set_page_config(
    page_title="Clasificación Digits (sklearn) | Streamlit",
    page_icon="🔢",
    layout="wide"
)

sns.set_style("whitegrid")


# -------------------------------
# Carga de datos (cache)
# -------------------------------
@st.cache_data
def load_data():
    digits = load_digits()
    X = digits.data.astype(np.float32)   # (n_samples, 64)
    y = digits.target.astype(int)        # (n_samples,)
    images = digits.images               # (n_samples, 8, 8)
    return X, y, images


# -------------------------------
# Modelos disponibles (>=5)
# -------------------------------
def get_models(random_state=42):
    models = {
        "Naive Bayes (GaussianNB)": GaussianNB(),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "SVM (RBF)": SVC(kernel="rbf", C=10, gamma="scale", probability=False),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Logistic Regression": LogisticRegression(max_iter=2000, solver="lbfgs")
    }
    return models


# -------------------------------
# Construcción de pipeline con/sin PCA
# -------------------------------
def make_pipeline(model, use_pca=False, n_components=0.95):
    steps = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=n_components, random_state=42)))
    steps.append(("clf", model))
    return Pipeline(steps)


# -------------------------------
# Estrategias de validación cruzada
# -------------------------------
def make_cv(strategy_name, n_splits=5, n_repeats=2, test_size=0.2, random_state=42):
    if strategy_name == "StratifiedKFold":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    elif strategy_name == "KFold":
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    elif strategy_name == "RepeatedStratifiedKFold":
        return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    elif strategy_name == "StratifiedShuffleSplit":
        return StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    else:
        return StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)


# -------------------------------
# Visualizaciones
# -------------------------------
def plot_class_distribution(y):
    fig, ax = plt.subplots(figsize=(6, 3))
    counts = pd.Series(y).value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, color="#4c78a8")
    ax.set_title("Distribución de clases (dígitos)")
    ax.set_xlabel("Clase")
    ax.set_ylabel("Cantidad")
    st.pyplot(fig, clear_figure=True)


def plot_sample_images(images, y, n=12):
    idx = np.random.choice(len(images), size=n, replace=False)
    cols = 6
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3.5))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            ax.imshow(images[idx[i]], cmap="gray")
            ax.set_title(f"y={y[idx[i]]}", fontsize=10)
    fig.suptitle("Muestras aleatorias de Digits (8x8)", y=1.02)
    st.pyplot(fig, clear_figure=True)


def plot_train_test_bars(train_acc, test_acc):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["Train", "Test"], [train_acc, test_acc], color=["#59a14f", "#e15759"])
    ax.set_ylim(0, 1.0)
    ax.set_title("Accuracy: Train vs Test")
    for i, v in enumerate([train_acc, test_acc]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
    st.pyplot(fig, clear_figure=True)


def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_title("Matriz de Confusión (Test)")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    st.pyplot(fig, clear_figure=True)


# -------------------------------
# Preprocesamiento del dibujo (Canvas -> 8x8)
# -------------------------------
def canvas_to_digits_vector(canvas_rgba, invert=True):
    """
    Convierte el canvas RGBA a vector 64 (8x8) aproximando el formato de sklearn digits.
    sklearn digits usa intensidades 0..16 en 8x8.
    """
    # canvas_rgba: (H, W, 4)
    img = Image.fromarray(canvas_rgba.astype(np.uint8), mode="RGBA").convert("L")  # a escala de grises
    # Fondo negro + trazo blanco usual en el canvas, lo ajustamos:
    if invert:
        img = ImageOps.invert(img)

    # Normalizar y redimensionar a 8x8
    img = img.resize((8, 8), resample=Image.Resampling.BILINEAR)

    arr = np.array(img).astype(np.float32)  # 0..255
    # Mapear 0..255 -> 0..16 (como digits)
    arr = (arr / 255.0) * 16.0
    vec = arr.reshape(1, -1)
    return vec, arr


# =========================================================
# UI
# =========================================================
st.title("🔢 Clasificación de Dígitos - sklearn Digits (tipo MNIST) + Streamlit")

st.markdown(
    """
Este dashboard permite:
- Revisar **calidad de datos**
- Probar **≥ 5 modelos de clasificación**
- Comparar desempeño **con/sin PCA**
- Ajustar el **porcentaje Train/Test**
- Evaluar con **validación cruzada**
- Visualizar **métricas + matriz de confusión**
- Dibujar un dígito con el mouse y **predecir**
"""
)

X, y, images = load_data()
labels = list(range(10))

# -------------------------------
# Sidebar - Controles
# -------------------------------
st.sidebar.header("⚙️ Configuración")

train_size = st.sidebar.slider(
    "Porcentaje de datos para entrenamiento",
    min_value=0.5, max_value=0.9, value=0.8, step=0.05
)

use_pca = st.sidebar.checkbox("Usar PCA", value=False)
pca_variance = st.sidebar.slider(
    "Varianza explicada (PCA n_components)",
    min_value=0.70, max_value=0.99, value=0.95, step=0.01,
    help="Si PCA está activo, se elige el número de componentes para explicar esta varianza."
)

model_name = st.sidebar.selectbox("Modelo", list(get_models().keys()))

st.sidebar.subheader("Validación Cruzada")
cv_strategy = st.sidebar.selectbox(
    "Estrategia",
    ["StratifiedKFold", "KFold", "RepeatedStratifiedKFold", "StratifiedShuffleSplit"]
)
n_splits = st.sidebar.slider("n_splits", 2, 10, 5)
n_repeats = st.sidebar.slider("n_repeats (solo RepeatedStratifiedKFold)", 1, 5, 2)
cv_test_size = st.sidebar.slider("test_size (solo StratifiedShuffleSplit)", 0.1, 0.4, 0.2, 0.05)

random_state = st.sidebar.number_input("random_state", value=42, step=1)

run_train = st.sidebar.button("🚀 Entrenar y Evaluar", use_container_width=True)

# -------------------------------
# 1) Calidad de Datos
# -------------------------------
tab1, tab2, tab3 = st.tabs(["1️⃣ Calidad de datos", "2️⃣ Entrenamiento / Evaluación", "3️⃣ Dibujar y predecir"])

with tab1:
    st.subheader("1) Verificación de calidad de los datos")

    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("N. muestras", X.shape[0])
    with colB:
        st.metric("N. features", X.shape[1])
    with colC:
        st.metric("Clases", len(np.unique(y)))

    # Missing values
    missing = np.isnan(X).sum()
    st.write(f"**Valores faltantes (NaN) en X:** {missing}")

    # Duplicados (opcional)
    dfX = pd.DataFrame(X)
    dup = dfX.duplicated().sum()
    st.write(f"**Filas duplicadas en X:** {dup}")

    # Distribución de clases
    st.markdown("### Distribución de clases")
    plot_class_distribution(y)

    st.markdown("### Muestras de imágenes (8x8)")
    plot_sample_images(images, y, n=12)

    st.info(
        "Notas: `load_digits()` contiene dígitos en 8x8 con intensidades similares a 0..16 (re-escaladas internamente). "
        "No suele tener NaNs. La estandarización ayuda mucho a SVM, LR y KNN."
    )


# -------------------------------
# 2) Entrenar y Evaluar (con/sin PCA, CV, métricas)
# -------------------------------
with tab2:
    st.subheader("2) Entrenamiento y evaluación del modelo")

    if not run_train:
        st.warning("Configura los parámetros en la barra lateral y presiona **Entrenar y Evaluar**.")
    else:
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, stratify=y, random_state=random_state
        )

        st.write(f"**Train:** {X_train.shape}  |  **Test:** {X_test.shape}")

        # Model + pipeline
        models = get_models(random_state=random_state)
        model = models[model_name]
        pipe = make_pipeline(model, use_pca=use_pca, n_components=pca_variance)

        # Fit
        pipe.fit(X_train, y_train)

        # Predictions
        y_pred_train = pipe.predict(X_train)
        y_pred_test = pipe.predict(X_test)

        # Metrics
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)

        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred_test, average="weighted", zero_division=0)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy (Train)", f"{train_acc:.4f}")
        col2.metric("Accuracy (Test)", f"{test_acc:.4f}")
        col3.metric("F1 (weighted)", f"{f1:.4f}")
        col4.metric("Precision (weighted)", f"{prec:.4f}")

        st.markdown("### Gráfico desempeño Train vs Test")
        plot_train_test_bars(train_acc, test_acc)

        st.markdown("### Matriz de Confusión (Test)")
        plot_confusion_matrix(y_test, y_pred_test, labels)

        st.markdown("### Reporte de Clasificación (Test)")
        st.code(classification_report(y_test, y_pred_test, digits=4), language="text")

        # 4.1 Cross validation
        st.markdown("## 4.1 Validación Cruzada (sobre el conjunto de entrenamiento)")
        cv = make_cv(cv_strategy, n_splits=n_splits, n_repeats=n_repeats, test_size=cv_test_size, random_state=random_state)

        scoring = {"acc": "accuracy", "f1w": "f1_weighted"}
        cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1)

        cv_df = pd.DataFrame({
            "train_accuracy": cv_res["train_acc"],
            "valid_accuracy": cv_res["test_acc"],
            "train_f1_weighted": cv_res["train_f1w"],
            "valid_f1_weighted": cv_res["test_f1w"],
        })

        st.dataframe(cv_df.style.format("{:.4f}"), use_container_width=True)

        colA, colB = st.columns(2)
        with colA:
            st.success(f"CV Valid Accuracy: **{cv_df['valid_accuracy'].mean():.4f} ± {cv_df['valid_accuracy'].std():.4f}**")
        with colB:
            st.success(f"CV Valid F1 (w): **{cv_df['valid_f1_weighted'].mean():.4f} ± {cv_df['valid_f1_weighted'].std():.4f}**")

        # Plot CV curves
        st.markdown("### Gráfica por fold (train vs valid)")
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(cv_df["train_accuracy"].values, marker="o", label="Train Acc")
        ax.plot(cv_df["valid_accuracy"].values, marker="o", label="Valid Acc")
        ax.set_title("Accuracy por fold")
        ax.set_xlabel("Fold")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.0)
        ax.legend()
        st.pyplot(fig, clear_figure=True)

        # Guardar pipeline en session_state para usar en el canvas
        st.session_state["trained_pipe"] = pipe
        st.session_state["use_pca"] = use_pca
        st.session_state["model_name"] = model_name

        st.info(
            "Tip: activa/desactiva PCA para comparar desempeño. "
            "PCA puede ayudar a acelerar o mejorar modelos lineales; en KNN a veces mejora, a veces empeora."
        )


# -------------------------------
# 3) Canvas: Dibujar y predecir
# -------------------------------
with tab3:
    st.subheader("3) Dibuja un dígito y reconócelo")

    if "trained_pipe" not in st.session_state:
        st.warning("Primero entrena un modelo en la pestaña **Entrenamiento / Evaluación**.")
    else:
        st.write(
            f"Modelo actual: **{st.session_state['model_name']}** | PCA: **{st.session_state['use_pca']}**"
        )

        colL, colR = st.columns([1, 1])

        with colL:
            st.markdown("### 🎨 Canvas (dibuja con el mouse)")
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=18,
                stroke_color="#FFFFFF",
                background_color="#000000",
                width=280,
                height=280,
                drawing_mode="freedraw",
                key="canvas",
            )

            predict_btn = st.button("🔍 Predecir dígito", use_container_width=True)

        with colR:
            st.markdown("### 🧪 Preprocesamiento (8x8) y Predicción")

            if canvas_result.image_data is not None:
                vec, img8 = canvas_to_digits_vector(canvas_result.image_data, invert=True)

                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(img8, cmap="gray")
                ax.set_title("Imagen 8x8 (entrada al modelo)")
                ax.axis("off")
                st.pyplot(fig, clear_figure=True)

                if predict_btn:
                    pipe = st.session_state["trained_pipe"]
                    pred = int(pipe.predict(vec)[0])

                    st.success(f"✅ Predicción: **{pred}**")

                    # Probabilidades (solo si el modelo las soporta)
                    if hasattr(pipe.named_steps["clf"], "predict_proba"):
                        probs = pipe.predict_proba(vec)[0]
                        prob_df = pd.DataFrame({"clase": list(range(10)), "prob": probs})
                        st.bar_chart(prob_df.set_index("clase"))
                    else:
                        st.info("Este modelo no expone `predict_proba` (ej. SVM sin probabilidad).")

            st.caption(
                "Nota: El dataset `digits` es 8x8, por eso reducimos el dibujo del canvas de 280x280 a 8x8."
            )
