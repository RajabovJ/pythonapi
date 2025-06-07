from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io
import os
import base64
import gdown
from tensorflow.keras.models import load_model
from skimage.morphology import remove_small_objects, remove_small_holes

# Fayl yo‘llari
CLASSIFICATION_MODEL_PATH = 'skin_cancer_model.h5'
SEGMENTATION_MODEL_PATH = 'unet_melanom_batch64.keras'

# Google Drive file IDlari
CLASSIFICATION_DRIVE_ID = '1-2V970NOplej2tN-JxLNLjoRuZekfgRV'
SEGMENTATION_DRIVE_ID = '1anBppDVDbJLYjpR3-JduxS9e1riFzu5M'

# Model yuklash funksiyasi
def download_model(file_id, output_path):
    if not os.path.exists(output_path):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output_path, quiet=False)

# Modellarni yuklaymiz
download_model(CLASSIFICATION_DRIVE_ID, CLASSIFICATION_MODEL_PATH)
download_model(SEGMENTATION_DRIVE_ID, SEGMENTATION_MODEL_PATH)

classification_model = load_model(CLASSIFICATION_MODEL_PATH)
segmentation_model = load_model(SEGMENTATION_MODEL_PATH)

# Klass nomlari va tavsiflari
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
    cleaned = (pred_mask > 0.5).astype(bool)
    cleaned = remove_small_objects(cleaned, min_size=min_size)
    cleaned = remove_small_holes(cleaned, area_threshold=hole_size)
    return cleaned.astype(np.uint8)

def mask_to_base64(mask_array):
    mask_img = Image.fromarray(mask_array)
    buffered = io.BytesIO()
    mask_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def extract_lesion_black_bg(original_img: Image.Image, binary_mask: np.ndarray) -> str:
    img_np = np.array(original_img)
    mask_3ch = np.stack([binary_mask]*3, axis=-1)
    black_bg = np.zeros_like(img_np)
    result_np = np.where(mask_3ch == 1, img_np, black_bg).astype(np.uint8)
    result_img = Image.fromarray(result_np)
    result_img = result_img.resize((300, 300), resample=Image.Resampling.LANCZOS)
    buffered = io.BytesIO()
    result_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# FastAPI ilova
app = FastAPI()

@app.post("/api/predict/")
async def predict_and_segment(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')

        # Klassifikatsiya
        img_class = img.resize((224, 224))
        arr_class = np.expand_dims(np.array(img_class) / 255.0, axis=0)
        predictions = classification_model.predict(arr_class)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Segmentatsiya
        img_seg = img.resize((256, 256))
        arr_seg = np.expand_dims(np.array(img_seg) / 255.0, axis=0)
        pred_mask = segmentation_model.predict(arr_seg)[0]

        if pred_mask.ndim == 4:
            pred_mask = pred_mask[0]
        if pred_mask.ndim == 3 and pred_mask.shape[-1] == 1:
            pred_mask = np.squeeze(pred_mask, axis=-1)
        if pred_mask.ndim == 3 and pred_mask.shape[-1] == 2:
            pred_mask = pred_mask[..., 1]

        cleaned_mask = clean_mask(pred_mask)

        mask_img = Image.fromarray((cleaned_mask * 255).astype(np.uint8))
        mask_img = mask_img.resize(img.size, resample=Image.NEAREST)
        resized_mask = np.array(mask_img) / 255.0

        mask_base64 = mask_to_base64(np.array(mask_img))
        extracted_lesion_base64 = extract_lesion_black_bg(img, resized_mask)

        return JSONResponse({
            "class": class_names[predicted_class],
            "description": class_descriptions[predicted_class],
            "segmentation_mask": mask_base64,
            "extracted_lesion": extracted_lesion_base64,
            "probabilities": predictions[0].tolist()
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# import os
# import io
# import base64
# import numpy as np
# from PIL import Image
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import JSONResponse
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# import gdown

# from skimage.morphology import remove_small_objects, remove_small_holes

# app = FastAPI()

# # Fayl yo'llari
# CLASSIFICATION_MODEL_PATH = 'skin_cancer_model.h5'
# SEGMENTATION_MODEL_PATH = 'unet_segmentation.h5'

# # Google Drive file ID
# CLASSIFICATION_DRIVE_ID = '1-2V970NOplej2tN-JxLNLjoRuZekfgRV'
# SEGMENTATION_DRIVE_ID = '1AbCdEfGhIJklMnOpQRsTuVwxYz012345'  # O'zingizni segmentatsiya model ID bilan almashtiring

# # Yuklab olish agar mavjud bo'lmasa
# def download_model(file_id, output_path):
#     if not os.path.exists(output_path):
#         url = f'https://drive.google.com/uc?id={file_id}'
#         gdown.download(url, output_path, quiet=False)

# download_model(CLASSIFICATION_DRIVE_ID, CLASSIFICATION_MODEL_PATH)
# download_model(SEGMENTATION_DRIVE_ID, SEGMENTATION_MODEL_PATH)

# # Modellarni yuklash
# classification_model = load_model(CLASSIFICATION_MODEL_PATH)
# segmentation_model = load_model(SEGMENTATION_MODEL_PATH)

# # Klasslar
# class_names = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']
# class_descriptions = [
#     "Quyosh nurlari ta'sirida yillar davomida paydo bo‘ladigan dog‘.",
#     "Bazal hujayrali karsinoma, teri saratoni turi.",
#     "Saratonsiz teri o‘sishi.",
#     "Fibroz teri shishi.",
#     "Xol yoki melanotsitlarning xavfsiz ko‘payishi.",
#     "Qon tomirlari hosil bo‘lgan xavfsiz o‘sishlar.",
#     "Melanotsitlardan kelib chiqadigan jiddiy teri saratoni."
# ]

# # Segmentatsiya maskni tozalash
# def clean_mask(pred_mask, min_size=300, hole_size=300):
#     cleaned = (pred_mask > 0.5).astype(bool)
#     cleaned = remove_small_objects(cleaned, min_size=min_size)
#     cleaned = remove_small_holes(cleaned, area_threshold=hole_size)
#     return cleaned.astype(np.uint8)

# # Maskni PNG base64 formatga o‘tkazish
# def mask_to_base64(mask_array):
#     mask_img = Image.fromarray((mask_array * 255).astype(np.uint8))
#     buffered = io.BytesIO()
#     mask_img.save(buffered, format="PNG")
#     return base64.b64encode(buffered.getvalue()).decode("utf-8")

# # API endpoint
# @app.post("/api/predict/")
# async def predict_and_segment(image: UploadFile = File(...)):
#     try:
#         contents = await image.read()
#         img = Image.open(io.BytesIO(contents)).convert('RGB')

#         # --- Tasniflash uchun ---
#         img_class = img.resize((224, 224))
#         arr_class = np.expand_dims(np.array(img_class) / 255.0, axis=0)
#         predictions = classification_model.predict(arr_class)
#         predicted_class = np.argmax(predictions, axis=1)[0]

#         # --- Segmentatsiya uchun ---
#         img_seg = img.resize((240, 240))
#         arr_seg = img_to_array(img_seg) / 255.0
#         arr_seg = np.expand_dims(arr_seg, axis=0)
#         pred_mask = segmentation_model.predict(arr_seg)[0]
#         pred_mask = np.argmax(pred_mask, axis=-1)
#         cleaned_mask = clean_mask(pred_mask)

#         # mask'ni base64 PNG shaklga o'tkazamiz
#         mask_base64 = mask_to_base64(cleaned_mask)

#         return JSONResponse({
#             "class": class_names[predicted_class],
#             "description": class_descriptions[predicted_class],
#             "segmentation_mask": mask_base64
#         })
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
