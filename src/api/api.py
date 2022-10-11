"""
Basic api for making predictions on brain images.
Output: segmentation mask.
"""
import os
from io import BytesIO
import numpy as np
from PIL import Image
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response

from keras.models import load_model
from keras.utils import img_to_array

SCRIPT_PATH = os.path.dirname(__file__)
MODEL_PATH = os.path.join(SCRIPT_PATH ,"..", "..", "models", "unet_brain_segmentation.h5")

app = FastAPI(title='Brain MRI Segmentation image prediction')

model = load_model(MODEL_PATH)

@app.post(
    "/predict/image",
    responses = {
        200: {
            "content": { "image/png": {} }
        }
    },
    response_class=Response
)
async def predict_api(file: UploadFile = File(...)):
    """
    Make mask prediction on given image file.

    Args:
        file: image file in .png .jpg .jpeg .tif format.

    Return:
        prediction mask image (one channel).
    """
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png", "tif")
    if not extension:
        return "Image must be jpg, png or tif format!"

    image = Image.open(BytesIO(await file.read())).resize((224, 224))
    image = np.expand_dims(img_to_array(image), 0)
    image = image.astype('float32') / 255.0
    image = model.predict(image)
    image = image.reshape((224, 224)) * 255
    image = image.astype(np.uint8)

    prediction = Image.fromarray(image)

    with BytesIO() as buf:
        prediction.save(buf, format='PNG')
        im_bytes = buf.getvalue()

    return Response(content=im_bytes, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, debug=True)
