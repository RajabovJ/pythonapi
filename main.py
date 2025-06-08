import os
import gdown
from tensorflow.keras.models import load_model
from skimage.morphology import remove_small_objects, remove_small_holes
from PIL import Image
import io
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from PIL import Image
import numpy as np
import io
import base64
# Fayl yo‘llari
CLASSIFICATION_MODEL_PATH = 'skin_cancer_model.h5'
SEGMENTATION_MODEL_PATH = 'unet_melanom_batch64.keras'

# Google Drive file IDlari
CLASSIFICATION_DRIVE_ID = '1-2V970NOplej2tN-JxLNLjoRuZekfgRV'
SEGMENTATION_DRIVE_ID = '1anBppDVDbJLYjpR3-JduxS9e1riFzu5M'

def download_model(file_id, output_path):
    if not os.path.exists(output_path):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output_path, quiet=False)

# Modellarni yuklash
download_model(CLASSIFICATION_DRIVE_ID, CLASSIFICATION_MODEL_PATH)
download_model(SEGMENTATION_DRIVE_ID, SEGMENTATION_MODEL_PATH)

classification_model = load_model(CLASSIFICATION_MODEL_PATH)
segmentation_model = load_model(SEGMENTATION_MODEL_PATH)

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

def clean_mask(pred_mask, min_size=300, hole_size=300):
    """
    Modeldan chiqqan ehtimoliy maskani tozalaydi:
    - kichik ob'ektlar va teshiklarni olib tashlaydi
    - natijani binary maska ko‘rinishida qaytaradi
    """
    cleaned = (pred_mask > 0.5).astype(bool)
    cleaned = remove_small_objects(cleaned, min_size=min_size)
    cleaned = remove_small_holes(cleaned, area_threshold=hole_size)
    return cleaned.astype(np.uint8)

def mask_to_base64(mask_array):
    """
    Binary yoki uint8 maskani base64 PNG stringga aylantiradi
    """
    mask_img = Image.fromarray(mask_array)
    buffered = io.BytesIO()
    mask_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def extract_lesion_black_bg(original_img: Image.Image, binary_mask: np.ndarray) -> str:
    """
    original_img: PIL.Image rasm (RGB)
    binary_mask: np.ndarray 0 va 1 bilan, original rasm o'lchamida

    Lezyon joyini qora fon ustida ajratadi.
    Natijani base64 PNG formatida string sifatida qaytaradi.
    """

    img_np = np.array(original_img)  # (H, W, 3)

    # Maskani 3 kanallik qilib ko'paytiramiz
    mask_3ch = np.stack([binary_mask]*3, axis=-1)

    # Qora fon yaratamiz
    black_bg = np.zeros_like(img_np)

    # Lezyon joyini ajratamiz, qolgan joylar qora
    result_np = np.where(mask_3ch == 1, img_np, black_bg).astype(np.uint8)

    # Natijani PIL rasmga aylantiramiz (asl o'lchamda)
    result_img = Image.fromarray(result_np)

    # Base64 ga aylantirish
    buffered = io.BytesIO()
    result_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return img_str


app = FastAPI()
@app.post("/api/predict/")
async def predict_and_segment(image: UploadFile = File(...)):
    try:
        # Rasmni o'qish va RGB formatga o'tkazish
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')

        # Asl o'lchamni saqlab qolish
        original_size = img.size  # (width, height)

        # --- Tasniflash uchun tayyorlash ---
        img_class = img.resize((224, 224))
        arr_class = np.expand_dims(np.array(img_class) / 255.0, axis=0)
        predictions = classification_model.predict(arr_class)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # --- Segmentatsiya uchun tayyorlash ---
        img_seg = img.resize((256, 256))
        arr_seg = np.expand_dims(np.array(img_seg) / 255.0, axis=0)
        pred_mask = segmentation_model.predict(arr_seg)[0]

        # Model chiqishini tekshirish va tozalash
        if pred_mask.ndim == 4:
            pred_mask = pred_mask[0]
        if pred_mask.ndim == 3 and pred_mask.shape[-1] == 1:
            pred_mask = np.squeeze(pred_mask, axis=-1)
        if pred_mask.ndim == 3 and pred_mask.shape[-1] == 2:
            pred_mask = pred_mask[..., 1]

        # Binary maska yaratish
        cleaned_mask = clean_mask(pred_mask)

        # Maskani asl o'lchamga qayta o'lchash
        mask_img = Image.fromarray((cleaned_mask * 255).astype(np.uint8))
        mask_img = mask_img.resize(original_size, resample=Image.NEAREST)
        resized_mask = np.array(mask_img) // 255  # 0/1 qilib qayta o'zgartiramiz

        # Base64 formatda maskani olish
        mask_base64 = mask_to_base64((resized_mask * 255).astype(np.uint8))

        # Lezyonni qora fon bilan ajratish
        extracted_lesion_base64 = extract_lesion_black_bg(img, resized_mask.astype(np.uint8))

        return JSONResponse({
            "class": class_names[predicted_class],
            "description": class_descriptions[predicted_class],
            "segmentation_mask": mask_base64,
            "extracted_lesion": extracted_lesion_base64,
            "probabilities": predictions[0].tolist(),
            "original_size": {"width": original_size[0], "height": original_size[1]}
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
