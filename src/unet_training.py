"""
Training pipeline for unet model with model saving.
"""

import os
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.metrics import BinaryIoU

from data_preparation import DataLoader, split_data_train_test
from unet_model_recipe import unet_model

SCRIPT_PATH = os.path.dirname(__file__)
EPOCHS = 10
BATCH_SIZE = 16

# Preprocess data
data_loader = DataLoader()
images, masks = data_loader.load_data(img_size=(224, 224))
images_train, images_test, masks_train, masks_test = split_data_train_test(images, masks)

# Create and compile model
model = unet_model()
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=[BinaryIoU(name="iou")]
)

# Train and save
model_checkpoint = ModelCheckpoint(
    os.path.join(SCRIPT_PATH, "..", "models", "unet_brain_segmentation.h5"),
    monitor='val_iou',
    verbose=1,
    save_best_only=True
)

history = model.fit(
    images_train,
    masks_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    validation_split=0.2,
    shuffle=True,
    callbacks=[model_checkpoint]
)
