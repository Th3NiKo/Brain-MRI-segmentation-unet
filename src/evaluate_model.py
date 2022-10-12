"""
Print default model evaluation metrics (Binary iou).
"""
import os
from keras.models import load_model
from data_preparation import DataLoader, split_data_train_test

SCRIPT_PATH = os.path.dirname(__file__)

data_loader = DataLoader()
images, masks = data_loader.load_data(img_size=(224, 224))
images_train, images_test, masks_train, masks_test = split_data_train_test(images, masks)

model = load_model(os.path.join(SCRIPT_PATH, "..", "models", "unet_brain_segmentation.h5"))
evaluate_train = model.evaluate(images_train, masks_train)
evaluate_test = model.evaluate(images_test, masks_test)

print(evaluate_train, evaluate_test)
