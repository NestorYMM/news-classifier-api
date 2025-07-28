import os
from flask import Flask, request, jsonify
import joblib
import numpy as np
import re
import string
import unicodedata

# Importaciones de NLTK
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- Configuración de NLTK ---
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except:
    nltk.download('omw-1.4')

english_stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --- Función de limpieza de texto (IDÉNTICA a la usada en entrenamiento) ---
def clean_text_for_modeling(text):
    """
    Limpia un string de texto para preparar datos para modelado
    """
    # 1. Convertir a minúsculas, como parte de la estandarización
    text = text.lower()

    # 2. Eliminar URLs y patrones comunes en redes sociales
    text = re.sub(r'https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'\brt\b', '', text)
    

    # 3. Eliminar puntuación y números
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text) # Eliminar números

    # 4. Normalizar caracteres Unicode (acentos, ñ, etc.)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    # 5. Tokenizar el texto
    tokens = word_tokenize(text)

    # 6. Eliminar stopwords y lematizar
    cleaned_tokens = []
    for word in tokens:
        if word not in english_stopwords and word.strip() != '':
            cleaned_tokens.append(lemmatizer.lemmatize(word))

    # 7. Eliminamos las palabras 'photo', 'new', 'video'
    #Dado que se observan como de las más comunes en multiples categorias
    cleaned_tokens = [word for word in cleaned_tokens if word != 'new']
    cleaned_tokens = [word for word in cleaned_tokens if word != 'photo']
    cleaned_tokens = [word for word in cleaned_tokens if word != 'video']

    # 8. Unir las palabras limpias de nuevo en un string
    text_cleaned = ' '.join(cleaned_tokens)
    text_cleaned = re.sub(r'\s+', ' ', text_cleaned).strip() # Eliminar espacios extra

    return text_cleaned

# --- Cargar el modelo, TF-IDF Vectorizer y LabelEncoder ---
MODEL_PATH = 'lr_model.pkl'
TFIDF_PATH = 'tfidf_vectorizer.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pkl'

try:
    model = joblib.load(MODEL_PATH)
    tfidf_vectorizer = joblib.load(TFIDF_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    print("Modelos y transformadores cargados exitosamente.")
except FileNotFoundError as e:
    print(f"Error: Uno o más archivos no se encontraron. Asegúrate de que estén en '{os.getcwd()}'. Error: {e}")
    exit() # Salir si no se pueden cargar los archivos esenciales para que la API funcione
except Exception as e:
    print(f"Error al cargar modelos/transformadores: {e}")
    exit()

# --- Inicializar la aplicación Flask ---
app = Flask(__name__)

# --- Endpoint de predicción ---
@app.route('/predict', methods=['POST'])
def predict():
    # Validar que la solicitud contiene JSON y la clave 'texts'
    if not request.json or 'texts' not in request.json:
        return jsonify({
            "error": "Por favor, envía una lista de textos bajo la clave 'texts' en formato JSON."
        }), 400

    texts = request.json['texts']

    # Validar que 'texts' es una lista
    if not isinstance(texts, list):
        return jsonify({
            "error": "La clave 'texts' debe contener una lista de cadenas (strings)."
        }), 400

    # Validar que todos los elementos en 'texts' son cadenas
    if not all(isinstance(t, str) for t in texts):
        return jsonify({
            "error": "Todos los elementos en la lista 'texts' deben ser cadenas (strings)."
        }), 400

    # Preprocesar los textos de entrada usando la misma función de limpieza
    cleaned_texts = [clean_text_for_modeling(text) for text in texts]

    # Transformar los textos limpios usando el TF-IDF Vectorizer cargado
    texts_tfidf = tfidf_vectorizer.transform(cleaned_texts)

    # Realizar predicciones
    predictions_encoded = model.predict(texts_tfidf)
    predictions_proba = model.predict_proba(texts_tfidf) # Obtener probabilidades para la confianza

    # Decodificar las etiquetas numéricas a sus nombres de categoría originales
    predicted_labels = label_encoder.inverse_transform(predictions_encoded)

    # Preparar los resultados para la respuesta JSON
    results = []
    for i, text in enumerate(texts):
        # Obtener la probabilidad de la clase predicha
        # np.max(predictions_proba[i]) obtiene la probabilidad más alta para esa predicción
        confidence = np.max(predictions_proba[i])
        results.append({
            "original_text": text,
            "predicted_category": predicted_labels[i],
            "confidence": float(confidence) # Convertir a float para asegurar la serialización JSON
        })

    # Devolver la respuesta como JSON
    return jsonify({"predictions": results})

# --- Ejecutar la aplicación Flask ---
if __name__ == '__main__':
    # Para desarrollo local: app.run(debug=True) habilita el modo de depuración
    # que recarga el servidor automáticamente con cambios y muestra errores.
    # Para despliegue real, debug=False y usar un servidor WSGI como Gunicorn.
    print("Iniciando la API Flask...")
    print("Endpoint de predicción disponible en: http://127.0.0.1:5000/predict")
    print("Envía solicitudes POST con un JSON que contenga una lista de textos bajo la clave 'texts'.")
    print("Ejemplo de JSON: {'texts': ['Texto de ejemplo 1', 'Texto de ejemplo 2']}")
    app.run(debug=False, host='0.0.0.0', port=5000) # host='0.0.0.0' hace que el servidor sea accesible desde otras máquinas en tu red local