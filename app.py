from flask import Flask, request, jsonify, render_template
import mediapipe as mp
import numpy as np
from flask_cors import CORS
from PIL import Image, ImageDraw
import io
import base64
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

app = Flask(__name__)
CORS(app)

# Configura las credenciales de Google Drive
CLIENT_SECRET_FILE = '/home/rafael/Documents/credenciales/secret.json'
SCOPES = ['https://www.googleapis.com/auth/drive.file']

# ID de la carpeta donde deseas subir la imagen
FOLDER_ID = '1X3aRgkGKUEuZXKNqDrOx2sbXvZsK3Lpi'

@app.route('/')
def index():
    return render_template('index.html')

# Inicializar el servicio de Google Drive
def obtener_servicio_drive():
    creds = service_account.Credentials.from_service_account_file(
        CLIENT_SECRET_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)
    return service

@app.route('/upload', methods=['POST'])
def detectar_Puntos_Faciales():
    if 'file' not in request.files:
        return jsonify({'error': 'No se recibió correctamente la imagen'})

    archivo = request.files['file']
    if archivo.filename == '':
        return jsonify({'error': 'No se cargó ninguna imagen'})

    # Leer el contenido de la imagen original
    imagen_original = archivo.read()
    archivo.seek(0)

    # Procesar la imagen para detectar puntos faciales
    image_np = np.array(Image.open(archivo).convert('RGB'))
    mp_face_mesh = mp.solutions.face_mesh

    if image_np is None:
        return jsonify({'error': 'Error al cargar la imagen'})

    # Crear una copia de la imagen en escala de grises
    imagen_con_puntos = Image.fromarray(image_np).convert('L')  # Convertir a escala de grises

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(image_np)
        puntos_deseados = [70, 55, 285, 300, 33, 468, 133, 362, 473, 263, 4, 185, 0, 306, 17]

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, landmark in enumerate(face_landmarks.landmark):
                    if idx in puntos_deseados:
                        h, w, _ = image_np.shape
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        size = 5
                        color = (255)  # Color blanco en escala de grises
                        thickness = 2

                        draw = ImageDraw.Draw(imagen_con_puntos)
                        draw.line((x - size, y - size, x + size, y + size), fill=color, width=thickness)
                        draw.line((x - size, y + size, x + size, y - size), fill=color, width=thickness)

    # Convertir la imagen procesada a formato base64
    buffered = io.BytesIO()
    imagen_con_puntos.save(buffered, format="PNG")
    img_data_con_puntos = buffered.getvalue()

    # Sube la imagen con puntos a Google Drive manteniendo su nombre
    service = obtener_servicio_drive()

    # Cambia esto para usar la imagen con puntos
    archivo_con_puntos = MediaIoBaseUpload(io.BytesIO(img_data_con_puntos), mimetype='image/png')

    # Cambiar el nombre del archivo si deseas
    archivo_metadata = {
        'name': f'puntos_{archivo.filename}',  # Usar un prefijo para distinguir
        'mimeType': 'image/png',
        'parents': [FOLDER_ID]  # Aquí especificas la carpeta
    }
    archivo_drive_subido = service.files().create(body=archivo_metadata, media_body=archivo_con_puntos).execute()

    return jsonify({
        'image_with_points_base64': base64.b64encode(img_data_con_puntos).decode('utf-8'),  # Imagen con puntos
        'drive_id': archivo_drive_subido.get('id')
    })

if __name__ == '__main__':
    app.run(debug=True)
