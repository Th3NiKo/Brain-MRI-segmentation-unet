"""
Make a prediction on whole dataset and save outcome to files.
"""
import os
from keras.models import load_model
from keras.preprocessing.image import save_img
from data_preparation import DataLoader

SCRIPT_PATH = os.path.dirname(__file__)

data_loader = DataLoader()
images, masks = data_loader.load_data(img_size=(224, 224))

model = load_model(os.path.join(SCRIPT_PATH, "..", "models", "unet_brain_segmentation.h5"))
predicted_masks = model.predict(images, batch_size=8)

for i, (image, mask) in enumerate(zip(images, predicted_masks)):
    save_img(f'output/{i}.png', image)
    save_img(f'output/{i}_mask.png', mask)
