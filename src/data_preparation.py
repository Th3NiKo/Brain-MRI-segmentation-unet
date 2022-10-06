"""
Module for loading and preprocessing data.

Kaggle dataset: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
"""

import os
import glob
from typing import List, Tuple

import numpy as np
from keras.utils import load_img, img_to_array
from sklearn.model_selection import train_test_split

SCRIPT_PATH = os.path.dirname(__file__)
DATASET_PATH = os.path.join(SCRIPT_PATH, "..", "data", "kaggle_3m")

class DataLoader:
    """
    Load Brain Mri Segmentation data, preprocess and return target input pairs.

    Dataset should be in same convention as original kaggle one, it is:
    Patient_folder/
        file_1_mask.tif
        file_1.tif
        ...
    """
    dataset_path: str
    image_files: List[str]
    mask_files: List[str]

    def __init__(self, dataset_path: str = DATASET_PATH):
        self._dataset_path = dataset_path
        self.__prepare_filenames()

    def __prepare_filenames(self):
        """
        Go through files in dataset path and split them into mask_files and image_files.
        Only files with masks included.
        """
        file_names = list(sorted(glob.glob(self._dataset_path + "/**/*mask.tif")))
        self.image_files, self.mask_files = [], []
        for path in file_names:
            self.mask_files.append(path)
            self.image_files.append(path[: -9] + ".tif")

    def load_data(
            self,
            normalize: bool = True,
            img_size: Tuple[int, int] = (255, 255),
            specific_patient: str = None
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data and return as numpy arrays for training (input and target).

        Args:
            normalize: if True function will normalize image into 0-1 range.
            img_size: Tuple representing shape of image. Img_height as first, img_width as second.
            specific_patient: string representing patient identifier ex: TCGA_DU_A5TS_19970726. 
                Allow to load data only for specific patient.
        Returns:
            Tuple containing numpy arrays of shapes:
                (number_of_images, img_height, img_width, 3)
                (number_of_images, img_height, img_width, 1)
        """
        images_files = self.image_files
        masks_files = self.mask_files
        if specific_patient is not None:
            images_files = [
                image_path for image_path in images_files if specific_patient in image_path
            ]
            masks_files = [
                mask_path for mask_path in masks_files if specific_patient in mask_path
            ]

        inputs = np.zeros(
            (len(images_files), *img_size, 3),
            dtype=np.float32
        )
        targets = np.zeros(
            (len(masks_files), *img_size, 1),
            dtype=np.uint8
        )

        for index, (image_path, mask_path) in enumerate(zip(images_files, masks_files)):
            image = load_img(image_path, color_mode="rgb", target_size = img_size)
            image = img_to_array(image)

            mask = load_img(mask_path, color_mode="grayscale", target_size = img_size)
            mask = img_to_array(mask)

            if normalize:
                image = image.astype('float32') / 255.0
                mask = mask.astype('uint8') / 255

            inputs[index] = image
            targets[index] = mask

        return inputs, targets


def split_data_train_test(
        inputs: np.ndarray, targets: np.ndarray,
        random_state=2022, test_size=0.2
    ):
    """
    Function for spliting data into train and test set (based on sklearn train_test_split).
    Default input shape (number_of_images, img_height, img_width, 3).
    Keeping random_state frozen for reproducible outcomes.
    """
    input_train, input_test, target_train, target_test = train_test_split(
        inputs,
        targets,
        test_size=test_size,
        random_state=random_state
    )

    return input_train, input_test, target_train, target_test
