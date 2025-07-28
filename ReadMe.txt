# Clasificador de Titulares de Noticias - API REST

Este proyecto implementa una API REST para clasificar titulares de noticias en diversas categorías temáticas. 
El modelo utilizado es una Regresión Logística entrenada con características TF-IDF, 
diseñado para ser eficiente y preciso en la clasificación de texto.

El objetivo principal es proporcionar un servicio que permita a los usuarios enviar titulares de noticias y recibir la categoría predicha, 
facilitando la organización y búsqueda de información.

##Características

* **Clasificación de Texto:** Clasifica titulares de noticias en categorías predefinidas.
* **API REST:** Interfaz a través de solicitudes HTTP POST.
* **Tecnologías:** Desarrollado con Python, Flask, scikit-learn y NLTK.
* **Gestión de Datos:** Incorpora preprocesamiento de texto (lematización, eliminación de stop words) y manejo del desbalance de clases (SMOTE).

## Nota:
La carpeta creacion modelo contiene el archivo Seleccion_Modelo.ipynb el cuál muestra a detalle el proceso de creación del modelo 

## 📦 Requisitos del Sistema

Asegúrate de tener instalado lo siguiente en tu sistema:

* **Python:** Versión 3.8 o superior (recomendado).
* **Git:** Para clonar el repositorio.

## Configuración del Entorno y Ejecución

Sigue estos pasos para poner en marcha la API en tu máquina local:

### 1. Clonar el Repositorio
:

```bash
git clone [URL_DE_TU_REPOSITORIO_GITHUB]
cd [nombre-del-directorio-de-tu-repositorio] # Por ejemplo: cd api_clasificador_noticias

### 2. Crear y Activar un Entorno Virtual
python -m venv n_ambiente

Luego activa el ambiente
-\n_ambiente\Scripts\activate  

### 3. Instalar las Dependencias de Python
pip install -r requirements.txt

### 4. Descargar los Recursos de NLTK
Abre python con el ambiente creado
python
>>> import nltk
>>> nltk.download('punkt')
>>> nltk.download('stopwords')
>>> nltk.download('wordnet')
>>> nltk.download('omw-1.4')
>>> nltk.download('punkt_tab')
>>> exit()

### 5. Ejecutar la API Flask
python app.py


 Uso de la API

La API expone un único endpoint de tipo POST para realizar predicciones.

Endpoint: /predict
Método: POST

URL: http://127.0.0.1:5000/predict

Formato de Solicitud (JSON):
Debe enviar un cuerpo de solicitud JSON con una clave texts que contenga una lista de cadenas (strings).
{
    "texts": [
        "Titular de la primera noticia.",
        "Otro titular de ejemplo para clasificar.",
        "Un tercer titular sobre ciencia o tecnología."
    ]
}

Formato de Respuesta (JSON):
La API devolverá un JSON con una lista de predicciones. Cada predicción incluirá el texto original, la categoría predicha y la confianza (probabilidad) de esa predicción.

{
    "predictions": [
        {
            "original_text": "Titular de la primera noticia.",
            "predicted_category": "Categoria_Predicha_1",
            "confidence": 0.9567
        },
        {
            "original_text": "Otro titular de ejemplo para clasificar.",
            "predicted_category": "Categoria_Predicha_2",
            "confidence": 0.8812
        },
        {
            "original_text": "Un tercer titular sobre ciencia o tecnología.",
            "predicted_category": "Categoria_Predicha_3",
            "confidence": 0.7934
        }
    ]
}

Ejemplo de solicitud hecha desde terminal:

curl -X POST -H "Content-Type: application/json" -d "{\"texts\": [\"The new budget plan sparks debate in Congress\", \"Latest fashion trends for the summer season\", \"Tips for a healthy lifestyle and wellbeing\"]}" [http://127.0.0.1:5000/predict](http://127.0.0.1:5000/predict)

Ejemplo de solicitud desde cript python:

# test_api.py
import requests
import json

url = "[http://127.0.0.1:5000/predict](http://127.0.0.1:5000/predict)"
headers = {"Content-Type": "application/json"}

data = {
    "texts": [
        "President announces new economic policy",
        "Celebrity wedding takes place in Italy",
        "New research on artificial intelligence",
        "Sports headlines from the NBA finals",
        "Gardening tips for beginners"
    ]
}

try:
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response.raise_for_status() # Lanza una excepción para errores HTTP (4xx o 5xx)
    predictions = response.json()
    print(json.dumps(predictions, indent=4))
except requests.exceptions.RequestException as e:
    print(f"Error al conectar con la API: {e}")
except json.JSONDecodeError:
    print(f"Error al decodificar la respuesta JSON. Respuesta del servidor: {response.text}")

