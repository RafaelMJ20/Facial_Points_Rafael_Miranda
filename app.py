from flask import Flask, request, jsonify, render_template
import mediapipe as mp
import numpy as np
from flask_cors import CORS
from PIL import Image, ImageDraw
import io
import base64
import json
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import random

app = Flask(__name__)
CORS(app)

# ID de la carpeta donde deseas subir la imagen
FOLDER_ID = '1Z5oK0YBGg8HFsbpmsUWFwKqtczKXZxPX'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def detectar_puntos_faciales():
    if 'file' not in request.files:
        return jsonify({'error': 'No se recibió correctamente la imagen'})

    archivo = request.files['file']
    if archivo.filename == '':
        return jsonify({'error': 'No se cargó ninguna imagen'})

    # Leer el contenido de la imagen original
    imagen_original = archivo.read()
    archivo.seek(0)

    image_np = np.array(Image.open(archivo).convert('RGB'))
    mp_face_mesh = mp.solutions.face_mesh

    if image_np is None:
        return jsonify({'error': 'Error al cargar la imagen'})

    resultados = {}
    transformaciones = {
        "original": image_np,
        "horizontal_flip": np.flip(image_np, axis=1),
        "brightness_increased": np.clip(image_np * 1.8, 0, 255).astype(np.uint8),
        "vertical_flip": np.flip(image_np, axis=0)
    }

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        for nombre, imagen_np in transformaciones.items():
            imagen_gris = Image.fromarray(imagen_np).convert('L')
            results = face_mesh.process(imagen_np)
            puntos_deseados = [70, 55, 285, 300, 33, 468, 133, 362, 473, 263, 4, 185, 0, 306, 17]

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    draw = ImageDraw.Draw(imagen_gris)
                    for idx, landmark in enumerate(face_landmarks.landmark):
                        if idx in puntos_deseados:
                            h, w, _ = image_np.shape
                            x = int(landmark.x * w)
                            y = int(landmark.y * h)
                            draw.line((x - 5, y - 5, x + 5, y + 5), fill=255, width=2)
                            draw.line((x - 5, y + 5, x + 5, y - 5), fill=255, width=2)

            # Guardar la imagen procesada como base64
            buffered = io.BytesIO()
            imagen_gris.save(buffered, format="PNG")
            img_data = buffered.getvalue()

            # Subir a Google Drive
            service = obtener_servicio_drive()
            archivo_drive = MediaIoBaseUpload(io.BytesIO(img_data), mimetype='image/png')
            archivo_metadata = {
                'name': f'{nombre}_{archivo.filename}',
                'mimeType': 'image/png',
                'parents': [FOLDER_ID]
            }
            archivo_subido = service.files().create(body=archivo_metadata, media_body=archivo_drive).execute()

            resultados[nombre] = {
                'image_base64': base64.b64encode(img_data).decode('utf-8'),
                'drive_id': archivo_subido.get('id')
            }

    return jsonify(resultados)


def obtener_servicio_drive():
    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    creds_info = json.loads(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON'))
    creds = service_account.Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)
    return service

def transformar_imagen(image_np, transform_type):
    """Aplica una transformación específica a la imagen."""
    if transform_type == 'horizontal_flip':
        return np.flip(image_np, axis=1)
    elif transform_type == 'vertical_flip':
        return np.flip(image_np, axis=0)
    elif transform_type == 'increase_brightness':
        return np.clip(random.uniform(1.5, 2) * image_np, 0, 255).astype(np.uint8)
    return image_np

def guardar_imagen_pil(image_np):
    """Convierte un arreglo numpy a una imagen PIL."""
    return Image.fromarray(image_np)

def subir_imagen_google_drive(service, img_data, filename):
    """Sube una imagen a Google Drive."""
    media = MediaIoBaseUpload(io.BytesIO(img_data), mimetype='image/png')
    metadata = {
        'name': filename,
        'mimeType': 'image/png',
        'parents': [FOLDER_ID]
    }
    archivo_drive = service.files().create(body=metadata, media_body=media).execute()
    return archivo_drive.get('id')

@app.route('/upload', methods=['POST'])
def detectar_Puntos_Faciales():
    if 'file' not in request.files:
        return jsonify({'error': 'No se recibió correctamente la imagen'})

    archivo = request.files['file']
    if archivo.filename == '':
        return jsonify({'error': 'No se cargó ninguna imagen'})

    image_np = np.array(Image.open(archivo).convert('RGB'))
    if image_np is None:
        return jsonify({'error': 'Error al cargar la imagen'})

    # Transformaciones
    transforms = {
        'original': image_np,
        'horizontal_flip': transformar_imagen(image_np, 'horizontal_flip'),
        'increase_brightness': transformar_imagen(image_np, 'increase_brightness'),
        'vertical_flip': transformar_imagen(image_np, 'vertical_flip')
    }

    # Procesar imágenes y subirlas
    service = obtener_servicio_drive()
    results = {}
    for transform_name, transformed_np in transforms.items():
        img_pil = guardar_imagen_pil(transformed_np)
        buffered = io.BytesIO()
        img_pil.save(buffered, format="PNG")
        img_data = buffered.getvalue()
        drive_id = subir_imagen_google_drive(service, img_data, f"{transform_name}_{archivo.filename}")
        results[transform_name] = {
            'image_base64': base64.b64encode(img_data).decode('utf-8'),
            'drive_id': drive_id
        }

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
