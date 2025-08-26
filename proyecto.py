import streamlit as st
st.set_page_config(layout="wide")
import spacy
import tensorflow as tf
import re
import nltk
import numpy as np
import pickle
import os
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
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
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

nlp = spacy.load('en_core_web_lg')

def vectorize(text):
    text = text.lower()
    text = re.sub(r'([^0-9A-Za-z \t])', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords_en]
    vectors = [w2v_model.wv[token] for token in tokens if token in w2v_model.wv]
    if len(vectors) == 0:
        return np.zeros(300) 
    return np.mean(vectors, axis=0)

def predecir(modelo, texto):
    texto = vectorize(texto)
    texto = texto.reshape(1, -1)
    if hasattr(modelo, 'predict'):
        texto = modelo.predict(texto)
        categories = ['ACCOUNT', 'ORDER', 'REFUND', 'INVOICE', 'PAYMENT', 'FEEDBACK', 'CONTACT', 'SHIPPING_ADDRESS', 'DELIVERY', 'CANCELLATION_FEE', 'NEWSLETTER']
        scores = {category: score for category, score in zip(categories, texto)}  # Corregir aquí
    else:
        texto = texto.reshape(1, 300, 1)
        texto = modelo.predict(texto)
        categories = ['ACCOUNT', 'ORDER', 'REFUND', 'INVOICE', 'PAYMENT', 'FEEDBACK', 'CONTACT', 'SHIPPING_ADDRESS', 'DELIVERY', 'CANCELLATION_FEE', 'NEWSLETTER']
        scores = {category: score for category, score in zip(categories, texto)}  # Corregir aquí
    return scores

w2v_model = Word2Vec.load('word2vec.model')


def predecirBatch(modelo, texto):
    texto = vectorize(texto)
    texto = texto.reshape(1, -1)
    resultado = modelo.predict(texto)
    categories = ['ACCOUNT', 'ORDER', 'REFUND', 'INVOICE', 'PAYMENT', 'FEEDBACK', 'CONTACT', 'SHIPPING_ADDRESS', 'DELIVERY', 'CANCELLATION_FEE', 'NEWSLETTER']
    return categories[resultado[0]]

class CustomLSTM(tf.keras.layers.LSTM):
    def __init__(self, *args, **kwargs):
        if 'time_major' in kwargs:
            kwargs.pop('time_major')
        super(CustomLSTM, self).__init__(*args, **kwargs)

def generate_text(seed_text, n_lines, model, tokenizer, max_len=20):
    result = []
    for _ in range(n_lines):
        text = []
        for _ in range(max_len):
            encoded = tokenizer.texts_to_sequences([seed_text])
            encoded = pad_sequences(encoded, maxlen=max_len-1, padding='pre')
            y_pred = np.argmax(model.predict(encoded, verbose=0), axis=-1)
            predicted_word = ''
            for word, index in tokenizer.word_index.items():
                if index == y_pred:
                    predicted_word = word
                    break
            seed_text += ' ' + predicted_word
            text.append(predicted_word)
            seed_text = text[-1]
        result.append(' '.join(text))
    return result

st.title("NLP Project")

option = st.sidebar.selectbox(
    "Select an option",
    ("Home", "NGrams", "TSNE", "Models", "Text Generator", "ChatGPT")
)

if option == "Home":
    st.header("Natural Language Processing Project for Customer Service")

    st.header("Introduction")
    st.write("""
This project uses Natural Language Processing (NLP) techniques to classify and analyze customer comments for a clothing company. 
The goal is to improve customer service by automating comment analysis, allowing the company to respond efficiently and accurately.
    """)

    st.header("Project Description")
    st.write("""
The project is divided into several sections:

- **NGrams**: Visualization of unigrams, bigrams, and trigrams to better understand the most common words and phrases in comments.
- **T-SNE**: High-dimensional visualization to understand the distribution of comment categories.
- **Models**: Comment classification using neural network models.
- **Text Generator**: Generation of automatic responses from a language model.
- **ChatGPT**: Interaction with an advanced language model to answer customer service questions.
    """)

    st.header("Technologies Used")
    st.write("""
The following technologies and tools were used to carry out this project:

- **Python**: Main programming language.
- **Streamlit**: Library for creating interactive web applications.
- **TensorFlow and Keras**: Frameworks for developing neural network models.
- **SpaCy**: Library for natural language processing.
- **NLTK**: Library for working with text and language.
- **Plotly**: Library for data visualization.
- **OpenAI API**: For integrating advanced language models like GPT-3 and GPT-4.
""")

    # Operation
    st.header("Operation")
    st.write("""
### 1. Text Preprocessing
Customer comments are preprocessed using tokenization, lemmatization, and stopword removal techniques.

### 2. Vectorization
SpaCy is used to convert text into vectors that can be processed by neural network models.

### 3. Classification Models
Several neural network models have been trained to classify comments into different categories such as 'ORDER', 'REFUND', 'PAYMENT', etc.

### 4. Text Generator
Using a language model, automatic responses are generated from an initial text string provided by the user.

### 5. ChatGPT
Integration of an advanced language model to interactively answer customer service questions.
    """)
    st.header("Conclusion")
    st.write("""
    This project demonstrates how natural language processing techniques and learning models can significantly improve customer service in a clothing company. 
    By automating the analysis and response to customer comments, faster and more accurate responses can be offered, improving customer satisfaction and operational efficiency.
    """)

elif option == "NGrams":
    st.header("✨Unigrams, Bigrams, and Trigrams✨")
    
    st.subheader("Unigrams")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image('unigrama_account.png', caption='Unigram Account', use_column_width=True)
    with col2:
        st.image('unigrama_order.png', caption='Unigram Order', use_column_width=True)
    with col3:
        st.image('unigrama_contact.png', caption='Unigram Contact', use_column_width=True)

    st.subheader("Unigrams and Bigrams")
    col4, col5 = st.columns(2)
    with col4:
        st.image('unigrama_refund.png', caption='Unigram Refund', use_column_width=True)
    with col5:
        st.image('bigrama_newsletter.png', caption='Bigram Newsletter', use_column_width=True)

    st.subheader("Bigrams")
    col6, col7, col8 = st.columns(3)
    with col6:
        st.image('bigrama_payment.png', caption='Bigram Payment', use_column_width=True)
    with col7:
        st.image('bigrama_feedback.png', caption='Bigram Feedback', use_column_width=True)
    with col8:
        st.image('bigrama_cancellation.png', caption='Bigram Cancellation Fee', use_column_width=True)

    st.subheader("Trigrams")
    col9, col10, col11 = st.columns(3)
    with col9:
        st.image('trigrama_delivery.png', caption='Trigram Delivery', use_column_width=True)
    with col10:
        st.image('trigrama_invoice.png', caption='Trigram Invoice', use_column_width=True)
    with col11:
        st.image('trigrama_shipping.png', caption='Trigram Shipping', use_column_width=True)

elif option == "TSNE":
    st.header("T-SNE🪄")
    st.image('tsne.png', caption='T-SNE Visualization')

elif option == "Models":
    st.header("Models")
    st.write("Choose an option")

    model_options = {
        'FNN': 'ProyectoFnn.h5', 
        'CNN': 'ProyectoCnn.h5', 
        'DT': 'modeloDT.pkl', 
        'SVM': 'modeloSVM.pkl'
    }

    selected_model = st.selectbox('🤖 Choose the model to use', list(model_options.keys()))
    

    # Load the model according to the type
    if selected_model in ['FNN', 'CNN']:
        modelo_cargado = tf.keras.models.load_model(model_options[selected_model])
    elif selected_model in ['DT', 'SVM']:
        modelo_cargado = joblib.load(model_options[selected_model])

    st.title('🌟 Comment Classifier 🌟')
    st.write('🔍 **Put your comment here to classify it:**')

    comentario = st.text_input('Comment', 'Put your comment here')

    if st.button('Classify'):
        scores = predecir(modelo_cargado, comentario)
        st.balloons()  # Balloons animation
        st.write('### Classification Results 🎉')
        cols = st.columns(len(scores))
        for col, (category, score) in zip(cols, scores.items()):
            col.metric(category, f"{round(score*100, 2)}%")

        index = ['prob']
        df = pd.DataFrame(scores, index=index)
        st.dataframe(df.style.background_gradient(cmap='coolwarm'))

        df = df.transpose().reset_index()
        df.columns = ['Category', 'Probability']
        fig = px.bar(df, x='Category', y='Probability', color='Category', title='Probability by Category')
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

    comments_file = st.file_uploader("📄 Choose a CSV file", type='csv')
    if comments_file is not None:
        comments_df = pd.read_csv(comments_file)
        st.write('### CSV File Content 📑')
        comments_df = pd.read_csv(comments_file)
        st.write('### CSV File Content 📑')
        st.dataframe(comments_df.style.background_gradient(cmap='viridis'))
        if st.button('Classify File'):
            comments_df['Category'] = comments_df['Text'].apply(lambda x: predecirBatch(modelo_cargado, x))
            st.write('### File Classification Results 📊')
            st.dataframe(comments_df.style.background_gradient(cmap='coolwarm'))


elif option == 'Text Generator':
    st.header('Text Generator')
    model_path = "ProyectoRN.h5"  
    
    custom_objects = {
        'Orthogonal': Orthogonal,
        'LSTM': CustomLSTM

    }

    model_cargado = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)

    st.title("Text Generator with Neural Network")
    st.sidebar.header('Text Generation Options')

    categoria = st.sidebar.selectbox('Select a category', ['customer', 'order', 'payment', 'feedback'])
    num_mensajes = st.sidebar.number_input('Number of messages to generate', min_value=1, max_value=20, value=10)
    start_string = st.sidebar.text_input('⌨️Enter the start of the text')

    if st.sidebar.button('Generate Text'):
        with st.spinner('Generating text...'):
            generated_texts = generate_text(start_string, num_mensajes, model_cargado, tokenizer)
        st.success('Text generated! 😃')
        st.write('### Results:')
        for i, text in enumerate(generated_texts, 1):
            st.write(f"{i}. {text}")

elif option == 'ChatGPT':
    st.header('ChatGPT for Customer Service')
    openai_key_file = st.file_uploader("Upload your OpenAI key", type=['txt'])
    if openai_key_file is not None:
        openai_api_key = openai_key_file.read().decode('utf-8').strip()
        os.environ['OPENAI_API_KEY'] = openai_api_key

        model_option = st.selectbox(
            "Select the model to use",
            ("gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-instruct"),
            index=0,
            placeholder="Select a model..."
        )

        llm = ChatOpenAI(model=model_option)

        system_prompt = """
        You are a virtual assistant expert in customer service for a clothing company. You respond to customer inquiries about topics such as accounts, orders, refunds, invoices, payments, feedback, contact, shipping addresses, delivery, cancellation fees, and newsletters.
        Classify customer inquiries into one of the following categories: 'ACCOUNT', 'ORDER', 'REFUND', 'INVOICE', 'PAYMENT', 'FEEDBACK', 'CONTACT', 'SHIPPING_ADDRESS', 'DELIVERY', 'CANCELLATION_FEE', 'NEWSLETTER'.
        If asked in Spanish, classify customer inquiries into one of the following categories: 'CUENTA', 'PEDIDO', 'REEMBOLSO', 'FACTURA', 'PAGO', 'COMENTARIO', 'CONTACTO', 'DIRECCIÓN_DE_ENVÍO', 'ENTREGA', 'TARIFA_DE_CANCELACIÓN', 'BOLETÍN'.
        Provide detailed and helpful answers for each inquiry, ensuring to provide excellent customer service, and always classify the question into the classifications.

        End with a nice comment for the customer.
        If you are asked to do something else, anything that is not related to the database you respond with 'Sorry, it won't be possible!'
        """

        user_query = st.text_input('Write your customer service query here:')

        if st.button('Generate response'):
            with st.spinner('Generating response...'):
                result = llm(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_query)
                    ]
                )
                st.success('Response successfully generated', icon='✅')
                st.write(result.content)

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
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
