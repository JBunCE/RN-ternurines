import base64

import cv2
import keras
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI, APIRouter


class Image(BaseModel):
    image_base64: str


classes = [
    'gato_adulto_ropa_casual',
    'gato_infante_accesorio',
    'gato_infante_disfraz_galleta',
    'gato_infante_flor',
    'gato_infante_hongo',
    'jirafa_b',
    'jirafa_infante_accesorio_naranja',
    'jirafa_infante_roja_accesorio',
    'jirafa_ropa_de_campo',
    'lobo_adulto_ropa_formal',
    'lobo_explorador',
    'lobo_jugador_b',
    'jirafa_fantasma'
]
model = keras.models.load_model('model.h5')


app = FastAPI()
router = APIRouter()


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict_image(image: Image):
    # Decode the base64 string to bytes
    img_bytes = base64.b64decode(image.image_base64)
    # Convert bytes to a NumPy array
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    # Decode the image from the NumPy array
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "Error decoding image"
    else:
        # Proceed with image processing and prediction
        img = cv2.resize(img, (64, 128))
        img = np.array(img)
        img = img.reshape(1, 64, 128, 1)
        prediction = model.predict(img)
        predicted_class = classes[np.argmax(prediction)]
        return f"Predicted class is: {predicted_class}"


app.include_router(router)
