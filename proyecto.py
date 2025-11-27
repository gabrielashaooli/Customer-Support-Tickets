import streamlit as st
st.set_page_config(layout="wide")
import spacy
import tensorflow as tf
import tensorflow.keras as tf_keras
import re
import nltk
import numpy as np
import pickle
import os
import sys
import types
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.initializers import Orthogonal
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
import plotly.express as px

# Usamos directamente la librería oficial de OpenAI 
from openai import OpenAI

try:
    import keras 
except ModuleNotFoundError:
    keras = types.ModuleType("keras")
    keras.__dict__.update(tf_keras.__dict__)
    sys.modules['keras'] = keras
    sys.modules['keras.src'] = tf_keras
    if hasattr(tf_keras, "preprocessing"):
        sys.modules['keras.src.preprocessing'] = tf_keras.preprocessing
        if hasattr(tf_keras.preprocessing, "text"):
            sys.modules['keras.src.preprocessing.text'] = tf_keras.preprocessing.text

# ---------------------------------------------------------------------
# RECURSOS NLTK
# ---------------------------------------------------------------------
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

try:
    stopwords_en = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    stopwords_en = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()

# ---------------------------------------------------------------------
# CARGA DEL MODELO DE spaCy 
# ---------------------------------------------------------------------
try:
    nlp = spacy.load('en_core_web_lg')
except Exception:
    nlp = None

# ---------------------------------------------------------------------
# CARGA DEL MODELO WORD2VEC
# ---------------------------------------------------------------------
w2v_model = Word2Vec.load('word2vec.model')
DIM_W2V = w2v_model.vector_size  # dimensión real del embedding (seguro ~100)

CATEGORIAS = [
    'ACCOUNT', 'ORDER', 'REFUND', 'INVOICE', 'PAYMENT',
    'FEEDBACK', 'CONTACT', 'SHIPPING_ADDRESS', 'DELIVERY',
    'CANCELLATION_FEE', 'NEWSLETTER'
]

# =====================================================================
# FUNCIONES DE PLN
# =====================================================================

def vectorizar(texto: str) -> np.ndarray:
    texto = texto.lower()
    texto = re.sub(r'([^0-9A-Za-z \t])', ' ', texto)
    tokens = word_tokenize(texto)
    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stopwords_en
    ]
    vectores = [w2v_model.wv[token] for token in tokens if token in w2v_model.wv]
    if len(vectores) == 0:
        return np.zeros(DIM_W2V)
    return np.mean(vectores, axis=0)


def ajustar_dim(vec: np.ndarray, dim_objetivo: int) -> np.ndarray:
    """
    Ajusta un vector 1D al tamaño dim_objetivo:
    - si es más grande, lo recorta,
    - si es más pequeño, lo rellena con ceros.
    """
    actual = vec.shape[0]
    if actual == dim_objetivo:
        return vec
    if actual > dim_objetivo:
        return vec[:dim_objetivo]
    # actual < dim_objetivo
    padding = np.zeros(dim_objetivo - actual)
    return np.concatenate([vec, padding])

def predecir(modelo, texto: str) -> dict:
    """
    Devuelve un diccionario {categoría: probabilidad} para un comentario.
    - Para modelos Keras (FNN/CNN) se adapta al input_shape del modelo.
    - Para modelos sklearn se usa directamente el vector Word2Vec.
    """
    base_vec = vectorizar(texto)  # 1D, largo = DIM_W2V

    # Modelos Keras (FNN / CNN)
    if isinstance(modelo, tf.keras.Model):
        input_shape = modelo.input_shape  # p.ej. (None, 300) o (None, 100, 1)
        rank = len(input_shape)

        # FNN
        if rank == 2:
            input_dim = input_shape[1]
            x = np.zeros((1, input_dim), dtype=np.float32)
            L = min(base_vec.size, input_dim)
            x[0, :L] = base_vec[:L]
            probs = modelo.predict(x, verbose=0)[0]

        # CNN
        elif rank == 3:
            seq_len = input_shape[1]
            channels = input_shape[2]
            x = np.zeros((1, seq_len, channels), dtype=np.float32)

            flat = base_vec  
            if channels == 1:
                L = min(flat.size, seq_len)
                x[0, :L, 0] = flat[:L]
            else:
                needed = seq_len * channels
                tmp = np.zeros(needed, dtype=np.float32)
                L = min(flat.size, needed)
                tmp[:L] = flat[:L]
                x[0, :, :] = tmp.reshape(seq_len, channels)
            probs = modelo.predict(x, verbose=0)[0]

        else:
            # Caso raro: cualquier otra cosa, usamos fallback simple
            x = base_vec.reshape(1, -1)
            probs = modelo.predict(x, verbose=0)[0]

    # Modelos sklearn con predict_proba (SVM, etc.)
    elif hasattr(modelo, "predict_proba"):
        x = base_vec.reshape(1, -1)
        probs = modelo.predict_proba(x)[0]

    # Fallback genérico
    else:
        x = base_vec.reshape(1, -1)
        y = modelo.predict(x)
        if isinstance(y, np.ndarray):
            cls = int(np.argmax(y))
        else:
            cls = int(y[0])
        probs = np.zeros(len(CATEGORIAS))
        probs[cls] = 1.0

    probs = np.array(probs, dtype=float)
    if probs.sum() > 0:
        probs = probs / probs.sum()

    scores = dict(zip(CATEGORIAS, probs))
    return scores


def predecir_batch(modelo, texto: str) -> str:
    """
    Devuelve la categoría final para un comentario (para clasificación en CSV).
    Usa la misma lógica de construcción de input que predecir().
    """
    base_vec = vectorizar(texto)

    # 🔹 Modelos Keras
    if isinstance(modelo, tf.keras.Model):
        input_dim = modelo.input_shape[-1]
        x = np.zeros((1, input_dim), dtype=np.float32)
        L = min(base_vec.size, input_dim)
        x[0, :L] = base_vec[:L]
        probs = modelo.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        return CATEGORIAS[idx]

    # 🔹 Modelos sklearn con probas
    if hasattr(modelo, "predict_proba"):
        x = base_vec.reshape(1, -1)
        probs = modelo.predict_proba(x)[0]
        idx = int(np.argmax(probs))
        return CATEGORIAS[idx]

    # 🔹 Fallback
    x = base_vec.reshape(1, -1)
    y = modelo.predict(x)[0]
    if isinstance(y, str):
        return y
    try:
        return CATEGORIAS[int(y)]
    except Exception:
        return str(y)

class CustomLSTM(tf.keras.layers.LSTM):
    """LSTM custom para cargar modelos antiguos que usaban time_major."""
    def __init__(self, *args, **kwargs):
        if 'time_major' in kwargs:
            kwargs.pop('time_major')
        super(CustomLSTM, self).__init__(*args, **kwargs)


def generar_texto(semilla, n_lineas, modelo, tokenizer, max_len=20):
    """Generador simple de texto a partir de una semilla."""
    resultado = []
    for _ in range(n_lineas):
        texto = []
        semilla_actual = semilla
        for _ in range(max_len):
            codificado = tokenizer.texts_to_sequences([semilla_actual])
            codificado = pad_sequences(codificado, maxlen=max_len - 1, padding='pre')
            y_pred = np.argmax(modelo.predict(codificado, verbose=0), axis=-1)[0]
            palabra_predicha = ''
            for palabra, indice in tokenizer.word_index.items():
                if indice == y_pred:
                    palabra_predicha = palabra
                    break
            if not palabra_predicha:
                break
            semilla_actual += ' ' + palabra_predicha
            texto.append(palabra_predicha)
        resultado.append(' '.join(texto))
    return resultado

# =====================================================================
# INTERFAZ STREAMLIT
# =====================================================================

st.title("Sistema NLP para Clasificación de Mensajes de Clientes")

opcion = st.sidebar.selectbox(
    "Selecciona una opción",
    (
        "Inicio",
        "N-Gramas",
        "T-SNE",
        "Modelos",
        "ChatGPT",
        "Marco legal y técnico"
    )
)

# ---------------------------------------------------------------------
# INICIO
# ---------------------------------------------------------------------
if opcion == "Inicio":
    st.header("Proyecto de Integración Tecnológica Atención a Clientes con IA")
    st.markdown("""Luis Atristain Alfaro, Efren Flores Porras, Gabriela Shaooli Cassab, Carlo Folgori Jacobo, Patricio Fernández Paillés, Oscar Rodríguez Alcántara y Miguel Angel Zamora del Castillo 
""")
    st.subheader("Descripción general")
    st.write("""
Este proyecto implementa un sistema de **Procesamiento de Lenguaje Natural (PLN)** para clasificar
mensajes de soporte al cliente en 11 categorías (ACCOUNT, ORDER, REFUND, INVOICE, PAYMENT,
FEEDBACK, CONTACT, SHIPPING_ADDRESS, DELIVERY, CANCELLATION_FEE, NEWSLETTER).

La idea central es que la empresa pueda:
- Priorizar casos críticos,
- Canalizar correctamente cada ticket,
- Mejorar la experiencia del usuario,
- Y todo esto **bajo supervisión humana**, respetando el marco jurídico y ético.
    """)

    st.subheader("Componentes de la solución")
    st.markdown("""
1. **Preprocesamiento de texto**  
   Limpieza, tokenización, eliminación de stopwords y lematización (NLTK + spaCy).

2. **Vectorización**  
   Representación de mensajes con **Word2Vec (300 dimensiones)** entrenado en el dataset.

3. **Modelos de clasificación**  
   - Modelos clásicos (SVM, Árbol de decisión, Random Forest, Regresión Logística – entrenados externamente).  
   - Modelos de redes neuronales (**FNN**, **CNN**) sobre embeddings.

4. **Evaluación**  
   - División 70/15/15 estratificada.  
   - Validación cruzada (modelos clásicos).  
   - Early stopping (redes neuronales).  
   - Métrica principal: **F1 macro** (objetivo ≥ 0.80, ninguna categoría < 0.70).

5. **Interfaz con Streamlit (esta app)**  
   - Clasificación de un mensaje individual.  
   - Clasificación masiva vía CSV.  
   - Visualización de N-Gramas y T-SNE.    
   - Módulo de ChatGPT especializado en atención al cliente.
    """)

    st.subheader("Hipótesis de trabajo")
    st.write("""
La hipótesis es que un sistema de clasificación automática, integrado en el flujo de atención al cliente,
permite **administrar mejor las solicitudes**, disminuir la carga de trabajo manual y **aumentar la
satisfacción de los usuarios**, siempre respetando la normatividad en materia de datos personales y
las buenas prácticas de IA responsable.
    """)

# ---------------------------------------------------------------------
# N-GRAMAS
# ---------------------------------------------------------------------
elif opcion == "N-Gramas":
    st.header("✨ Visualización de N-Gramas ✨")
    
    st.subheader("Unigramas")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image('unigrama_account.png', caption='Unigrama – ACCOUNT', use_container_width=True)
    with col2:
        st.image('unigrama_order.png', caption='Unigrama – ORDER', use_container_width=True)
    with col3:
        st.image('unigrama_contact.png', caption='Unigrama – CONTACT', use_container_width=True)

    st.subheader("Unigramas y Bigramas")
    col4, col5 = st.columns(2)
    with col4:
        st.image('unigrama_refund.png', caption='Unigrama – REFUND', use_container_width=True)
    with col5:
        st.image('bigrama_newsletter.png', caption='Bigrama – NEWSLETTER', use_container_width=True)

    st.subheader("Bigramas")
    col6, col7, col8 = st.columns(3)
    with col6:
        st.image('bigrama_payment.png', caption='Bigrama – PAYMENT', use_container_width=True)
    with col7:
        st.image('bigrama_feedback.png', caption='Bigrama – FEEDBACK', use_container_width=True)
    with col8:
        st.image('bigrama_cancellation.png', caption='Bigrama – CANCELLATION_FEE', use_container_width=True)

    st.subheader("Trigramas")
    col9, col10, col11 = st.columns(3)
    with col9:
        st.image('trigrama_delivery.png', caption='Trigrama – DELIVERY', use_container_width=True)
    with col10:
        st.image('trigrama_invoice.png', caption='Trigrama – INVOICE', use_container_width=True)
    with col11:
        st.image('trigrama_shipping.png', caption='Trigrama – SHIPPING_ADDRESS / DELIVERY', use_container_width=True)

# ---------------------------------------------------------------------
# T-SNE
# ---------------------------------------------------------------------
elif opcion == "T-SNE":
    st.header("Visualización T-SNE 🪄")
    st.write("Distribución en 2D de las representaciones de los mensajes por categoría.")
    st.image('tsne.png', caption='Visualización T-SNE de las categorías de mensajes')

# ---------------------------------------------------------------------
# MODELOS
# ---------------------------------------------------------------------
elif opcion == "Modelos":
    st.header("Modelos de Clasificación")
    st.write("Selecciona el modelo con el que quieres clasificar los mensajes:")

    opciones_modelo = {
        'FNN (Red neuronal densa)': 'ProyectoFnn.h5',
        'CNN (Red neuronal convolucional)': 'ProyectoCnn.h5',
        'Árbol de decisión (DT)': 'modeloDT.pkl',
        'SVM': 'modeloSVM.pkl'
    }

    nombre_modelo = st.selectbox('🤖 Modelo a utilizar', list(opciones_modelo.keys()))
    ruta_modelo = opciones_modelo[nombre_modelo]

    modelo_cargado = None

# Modelos de redes neuronales (FNN / CNN)
    if "Red neuronal" in nombre_modelo:
        modelo_cargado = tf.keras.models.load_model(ruta_modelo)

# Árbol de decisión (DT)
    elif "Árbol" in nombre_modelo:
        try:
            modelo_cargado = joblib.load(ruta_modelo)
        except ValueError:
            st.warning(
            "⚠️ El modelo de Árbol de decisión (.pkl) fue entrenado con otra versión de scikit-learn.\n\n"
            "Para esta demo, nos concentraremos en FNN, CNN y SVM."
        )
            st.stop()

# SVM 
    elif "SVM" in nombre_modelo:
        try:
            modelo_cargado = joblib.load(ruta_modelo)
        except Exception as e:
            st.warning(
            "⚠️ El modelo SVM no se pudo cargar en este entorno.\n"
            "Para la demo puedes usar FNN o CNN y explicar que SVM requiere reentrenarse aquí."
        )
            st.stop()


    st.subheader('🌟 Clasificador de comentarios 🌟')
    st.write('🔍 **Escribe un comentario para clasificarlo:**')

    comentario = st.text_input('Comentario', 'Escribe aquí el mensaje del cliente')

    if st.button('Clasificar') and modelo_cargado is not None:
        scores = predecir(modelo_cargado, comentario)
        st.balloons()
        st.write('### Resultados de clasificación 🎉')
        cols = st.columns(len(scores))
        for col, (categoria, score) in zip(cols, scores.items()):
            col.metric(categoria, f"{round(score*100, 2)}%")

        indice = ['Probabilidad']
        df = pd.DataFrame(scores, index=indice)
        st.dataframe(df)

        df_plot = df.transpose().reset_index()
        df_plot.columns = ['Categoría', 'Probabilidad']
        fig = px.bar(
            df_plot,
            x='Categoría',
            y='Probabilidad',
            color='Categoría',
            title='Probabilidad por categoría'
        )
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Clasificación de archivo CSV")
    archivo = st.file_uploader("📄 Carga un archivo CSV con una columna 'Text'", type='csv')
    if archivo is not None and modelo_cargado is not None:
        df_csv = pd.read_csv(archivo)
        st.write('### Contenido del archivo 📑')
        st.dataframe(df_csv.head())

        if st.button('Clasificar archivo'):
            if 'Text' not in df_csv.columns:
                st.error("El CSV debe contener una columna llamada 'Text' con los mensajes de los clientes.")
            else:
                df_csv['Category'] = df_csv['Text'].apply(lambda x: predecir_batch(modelo_cargado, x))
                st.write('### Resultados de clasificación del archivo 📊')
                st.dataframe(df_csv)

# ---------------------------------------------------------------------
# GENERADOR DE TEXTO
# ---------------------------------------------------------------------
elif opcion == "Generador de texto":
    st.header('Generador de texto')

    ruta_modelo_rnn = "ProyectoRN.h5"
    objetos_personalizados = {
        'Orthogonal': Orthogonal,
        'LSTM': CustomLSTM
    }

    modelo_rnn = tf.keras.models.load_model(
        ruta_modelo_rnn,
        custom_objects=objetos_personalizados,
        compile=False
    )

    try:
        with open('tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
    except ModuleNotFoundError:
        st.error(
            "⚠️ El archivo tokenizer.pkl fue creado con otra versión de Keras.\n"
            "Instala una versión compatible o vuelve a generar el tokenizer en este entorno."
        )
        st.stop()

    st.subheader("Generador de respuestas automáticas (demo)")
    st.sidebar.header('Opciones de generación')

    categoria = st.sidebar.selectbox(
        'Selecciona una categoría (simbólica, solo para contexto)',
        ['customer', 'order', 'payment', 'feedback']
    )
    num_mensajes = st.sidebar.number_input(
        'Número de mensajes a generar',
        min_value=1,
        max_value=20,
        value=5
    )
    semilla = st.sidebar.text_input('⌨️ Texto inicial (seed)')

    if st.sidebar.button('Generar texto'):
        if not semilla:
            st.warning("Por favor escribe un texto inicial (semilla).")
        else:
            with st.spinner('Generando texto...'):
                textos_generados = generar_texto(semilla, num_mensajes, modelo_rnn, tokenizer)
            st.success('Texto generado 😃')
            st.write('### Resultados:')
            for i, texto in enumerate(textos_generados, 1):
                st.write(f"{i}. {texto}")

# ---------------------------------------------------------------------
# CHATGPT
# ---------------------------------------------------------------------
elif opcion == "ChatGPT":
    st.header('ChatGPT para atención a clientes')

    archivo_key = st.file_uploader("Sube tu API key de OpenAI (archivo .txt)", type=['txt'])
    cliente = None

    if archivo_key is not None:
        openai_api_key = archivo_key.read().decode('utf-8').strip()
        os.environ['OPENAI_API_KEY'] = openai_api_key
        cliente = OpenAI(api_key=openai_api_key)

        modelo = st.selectbox(
            "Selecciona el modelo de OpenAI",
            ("gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"),
            index=0
        )

        system_prompt = """
        Eres un asistente virtual experto en atención a clientes de una empresa.
        Atiendes dudas sobre: cuentas, pedidos, reembolsos, facturas, pagos, comentarios,
        contacto, direcciones de envío, entregas, tarifas de cancelación y newsletters.

        Debes:
        1) Contestar de forma clara, empática y útil.
        2) Clasificar la consulta del cliente en una de estas categorías en inglés:
           'ACCOUNT', 'ORDER', 'REFUND', 'INVOICE', 'PAYMENT', 'FEEDBACK', 'CONTACT',
           'SHIPPING_ADDRESS', 'DELIVERY', 'CANCELLATION_FEE', 'NEWSLETTER'.

        Si te escriben en español, también puedes mencionar la categoría equivalente en español:
           'CUENTA', 'PEDIDO', 'REEMBOLSO', 'FACTURA', 'PAGO', 'COMENTARIO', 'CONTACTO',
           'DIRECCIÓN_DE_ENVÍO', 'ENTREGA', 'TARIFA_DE_CANCELACIÓN', 'BOLETÍN'.

        Siempre indica explícitamente al final la categoría asignada.
        Termina con una línea amable para el cliente.

        Si te piden algo totalmente ajeno a este dominio de soporte al cliente,
        responde únicamente: 'Sorry, it won't be possible!'.
        """

        consulta = st.text_input('Escribe aquí la consulta del cliente:')

        if st.button('Generar respuesta'):
            if not consulta:
                st.warning("Por favor escribe una consulta primero.")
            else:
                with st.spinner('Generando respuesta...'):
                    respuesta = cliente.chat.completions.create(
                        model=modelo,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": consulta}
                        ]
                    )
                    contenido = respuesta.choices[0].message.content
                    st.success('Respuesta generada ✅')
                    st.write(contenido)
    else:
        st.info("Sube un archivo .txt con tu API key de OpenAI para activar este módulo.")

    st.markdown(
        """
        <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 20px;
            border-radius: 12px;
        }
        .stTextInput>div>div>input {
            border: 2px solid #4CAF50;
            font-size: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------------------------------------------
# MARCO LEGAL Y TÉCNICO (PARCIAL 3)
# ---------------------------------------------------------------------
elif opcion == "Marco legal y técnico":
    st.header("Marco legal y técnico – Evaluación integral del proyecto")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Naturaleza jurídica",
        "Datos personales (LFPDPPP)",
        "Transparencia y AI Act",
        "Sesgos y gobernanza",
        "Ética, riesgos y conclusiones"
    ])

    # ---------------- TAB 1: Naturaleza jurídica ----------------
    with tab1:
        st.subheader("Naturaleza Jurídica del Sistema")
        st.markdown("""
- El sistema **clasifica mensajes**, no toma decisiones autónomas.
- Carece de voluntad: **no puede obligarse, consentir ni generar efectos jurídicos directos**.
- Su uso requiere **validación humana constante**.
- Es un sistema **asistencial**, no operativo ni decisorio.
- No puede generar obligaciones a favor de terceros ni responsabilidad por actos propios de manera autónoma.
        """)

        st.subheader("Clasificación regulatoria (UE – AI Act)")
        st.markdown("""
- Clasificado como sistema de **“riesgo limitado”** (art. 25 AI Act).
- No impacta derechos legales o económicos de forma autónoma.
- Obligaciones clave:
  - **Transparencia** ante usuarios y operadores.
  - **Información suficiente** para entender la salida del sistema.
  - **Gobernanza proporcional** al riesgo.
        """)

        st.subheader("Marco mexicano aplicable")
        st.markdown("""
Ante la ausencia de una ley específica de IA en México, se toma como base:

- **Ley Federal de Protección de Datos Personales en Posesión de los Particulares (LFPDPPP)**  
- **Responsabilidad civil y de producto**
- **Principios constitucionales de derechos humanos**

Esto permite:
- Delimitar la responsabilidad del **proveedor**, del **operador** y de la **empresa usuaria**.
- Enmarcar el uso del sistema dentro de deberes de **cuidado, diligencia y no discriminación**.
        """)

    # ---------------- TAB 2: Protección de datos (LFPDPPP) ----------------
    with tab2:
        st.subheader("Protección de Datos Personales (LFPDPPP)")

        st.markdown("### Principios esenciales aplicables")
        st.markdown("""
- **Aviso de privacidad** claro y accesible.
- **Consentimiento expreso** o mediante signos inequívocos.
- Garantía plena de **Derechos ARCO** (Acceso, Rectificación, Cancelación y Oposición).
- **Trazabilidad** del uso del sistema para:
  - Proteger al desarrollador y operadores.
  - Evidenciar buen uso frente a autoridades y usuarios.
        """)

        st.markdown("### Obligaciones del artículo 19 LFPDPPP")
        st.markdown("""
**Medidas administrativas, técnicas y físicas**, por ejemplo:

- Controles de acceso por **roles**.
- **Cifrado** de datos en tránsito y en reposo.
- Eliminación **segura** de datos tras su uso.
- **Minimización** de datos y, cuando sea posible, procesamiento local o pseudonimizado.

El aviso de privacidad debe informar:

- El **uso de IA** para clasificar mensajes.
- La existencia de un módulo de **XAI** (explicabilidad) para atender solicitudes de información.
- La base de licitud del tratamiento:
  - Ordenamiento jurídico válido,
  - Consentimiento informado,
  - O relación jurídica previa con la persona usuaria.
        """)

    # ---------------- TAB 3: Transparencia y documentación (AI Act) ----------------
    with tab3:
        st.subheader("Transparencia y Documentación (AI Act)")

        st.markdown("### Requisitos internacionales (art. 13, 14, 52 AI Act)")
        st.markdown("""
El sistema debe garantizar:

- Identidad del **proveedor** y del responsable.
- **Finalidad** y límites del sistema de clasificación.
- **Métricas de exactitud y desempeño** comunicadas de forma comprensible.
- **Revisión humana garantizada** en el flujo operativo.

        """)

        st.markdown("### Implementación en nuestro sistema")
        st.markdown("""
- Interfaz **intuitiva**, con indicadores de contexto (palabras clave, categorías).
- Información clara sobre:
  - Qué hace el sistema,
  - Qué no hace,
  - Y cómo debe usarse correctamente.
- **Manual de uso** para operadores y auditores, que incluya:
  - Propósito y alcance,
  - Limitaciones técnicas,
  - Requisitos del sistema,
  - Mecanismos de revisión humana,
  - Métricas y umbrales de desempeño aceptable.
- Un módulo de **XAI** previsto para:
  - Dar trazabilidad de cada clasificación,
  - Explicar factores que influyeron en la decisión del modelo.
        """)

    # ---------------- TAB 4: Sesgos, actualización y gobernanza ----------------
    with tab4:
        st.subheader("Mitigación de Sesgos")

        st.markdown("### Riesgos potenciales")
        st.markdown("""
- Sesgos **lingüísticos**: regionalismos, variaciones dialectales, expresiones coloquiales.
- Riesgos de **discriminación indirecta** (trato desigual a ciertos grupos).
- Riesgos legales:
  - **Responsabilidad penal** en casos extremos,
  - **Daño moral**,
  - **Responsabilidad civil** frente a personas afectadas.
        """)

        st.markdown("### Medidas de mitigación implementadas")
        st.markdown("""
- **Auditoría algorítmica semestral**.
- Validación cruzada para **subgrupos lingüísticos**.
- Métricas de **equidad** y tasas de error balanceadas entre categorías.
- Identificación y análisis de **outliers** (casos atípicos).
- **Reporte interno** para TEVV (Testing, Evaluation, Verification and Validation) y mejora continua.
- Recomendación de contar con una **póliza de responsabilidad civil** frente a terceros.
        """)

        st.subheader("Actualización y Gobernanza del Sistema")
        st.markdown("### Control de versiones")
        st.markdown("""
- Registro de cada **iteración del modelo**.
- Validaciones técnicas, legales y éticas antes del despliegue.
- Análisis de impacto y evidencia de pruebas.
- Aprobación por parte de la figura de **AI Compliance Officer**.
        """)

        st.markdown("### Post-despliegue")
        st.markdown("""
- Monitoreo de **drift** (cambio en patrones de lenguaje y datos).
- Detección de comportamientos anómalos.
- Mecanismo de **rollback** para regresar a versiones estables si alguna actualización:
  - Degrada la exactitud,
  - Afecta la seguridad,
  - O reduce la transparencia.
        """)

    # ---------------- TAB 5: Ética, responsabilidad y conclusiones ----------------
    with tab5:
        st.subheader("Enfoque Ético Integral")

        st.markdown("### Documento ético accesible a usuarios")
        st.markdown("""
El proyecto contempla un documento ético que explique:

- La **visión ética** del sistema y los valores que lo guían:
  - Dignidad,
  - Igualdad,
  - No discriminación,
  - Transparencia,
  - Responsabilidad social.
- Basado en:
  - La **Constitución Mexicana** (parte dogmática),
  - Tratados internacionales de **derechos humanos**.
        """)

        st.markdown("### Marco ético operativo")
        st.markdown("""
- Uso **responsable** de datos.
- Límites funcionales del sistema (solo apoyo a clasificación, sin decisiones finales).
- **Explicabilidad mínima** garantizada hacia usuarios y auditores.
- **Revisión humana obligatoria** en casos de baja confianza o alto impacto.
- Evaluación ética **anual** sobre el impacto real del sistema.
- Restricciones estrictas frente a:
  - Usos prohibidos,
  - Desvíos de finalidad,
  - O aplicaciones que comprometan derechos fundamentales.
        """)

        st.subheader("Responsabilidad y Riesgos Jurídicos")
        st.markdown("""
- El sistema **no tiene voluntad propia** → la responsabilidad recae en:
  - Quienes lo diseñan,
  - Quienes lo operan,
  - Y la empresa que decide implementarlo.
- Los riesgos se mitigan mediante:
  - **Transparencia reforzada**,
  - Auditorías semestrales,
  - Trazabilidad documentada,
  - Medidas de seguridad robustas,
  - Revisión humana constante.

En el contexto mexicano, la ausencia de una ley específica de IA se compensa con:

- Regulación de **datos personales**,
- **Derechos humanos**,
- **Responsabilidad civil**,
- Normativa de **protección al consumidor**.
        """)

        st.subheader("Conclusiones del Cumplimiento")
        st.markdown("""
- El sistema está razonablemente clasificado como de **riesgo limitado**.
- Cumple con las obligaciones del **AI Act** en:
  - Transparencia,
  - Gobernanza,
  - Documentación proporcional al riesgo.
- Se encuentra alineado con la **LFPDPPP** en:
  - Aviso de privacidad,
  - Seguridad de datos,
  - Ejercicio de derechos ARCO.
- Cuenta con estrategias sólidas contra sesgos, con **auditorías periódicas** y métricas de equidad.
- Integra un **marco ético operativo**, con revisión anual y enfoque en no discriminación.
- El proyecto refleja una visión **interdisciplinaria** Derecho + Ingeniería:
  - Se demuestra **cumplimiento**,  
  - **Responsabilidad**,  
  - Y **trazabilidad** técnica y jurídica del sistema.
        """)
