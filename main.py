import os
import gdown
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = FastAPI()

MODEL_PATH = 'skin_cancer_model.h5'
DRIVE_FILE_ID = '1-2V970NOplej2tN-JxLNLjoRuZekfgRV'  # faqat ID

if not os.path.exists(MODEL_PATH):
    url = f'https://drive.google.com/uc?id={DRIVE_FILE_ID}'
    gdown.download(url, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)


class_names = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']
class_descriptions = [
    "Quyosh nurlari ta'sirida yillar davomida paydo bo‘ladigan dog‘.",
    "Bazal hujayrali karsinoma, teri saratoni turi.",
    "Saratonsiz teri o‘sishi.",
    "Fibroz teri shishi.",
    "Xol yoki melanotsitlarning xavfsiz ko‘payishi.",
    "Qon tomirlari hosil bo‘lgan xavfsiz o‘sishlar.",
    "Melanotsitlardan kelib chiqadigan jiddiy teri saratoni."
]

@app.post("/api/predict/")
async def predict_image(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)

        if img_array.ndim == 2:
            img_array = np.stack((img_array,) * 3, axis=-1)

        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]

        return JSONResponse({
            "class": class_names[predicted_class],
            "description": class_descriptions[predicted_class]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
