# Brain MRI Segmentation (Unet model)

## Table of contents

* [Description](#description)
* [Getting Started](#getting-started)
* [Usage](#usage)
* [Example output](#example-output)

## Description

Repository contains whole training pipeline using own implementation of [unet model](https://arxiv.org/abs/1505.04597) on [Brain MRI segmentation dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation).
Main difference between original paper model and this implementation is droput replacement with batch normalization.
Purpose: create segmentation model for anomalous brain parts detection -> helping doctors with expertise.

**Files structure:**

1. src (all scripts)
    * data_preparation.py - loading whole data/specific patient and spliting into test train
    * evaluate_model.py - evaluate model with default model metric (binary iou)
    * predict.py - make prediction on data and save images in output folder
    * unet_model_recipe.py - whole unet model architecture
    * unet_training.py - model training pipeline
2. notebooks (notebooks and analysis)
    * model_predictions_analysis.ipynb - check model predictions, specific patient output
3. models (pretrained models)
    * unet_brain_segmentation.h5 - pretrained unet model for brain segmentation
4. data (raw data)

## Getting Started

### Quick start

Tested with python 3.10.4

Libraries used:
    - tensorflow
    - scikit-learn
    - numpy
    - matplotlib

You can install all using pip.

```bash
pip install -r requirements.txt
```

Download dataset from Kaggle
https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

Default data structure looks like this:
data
-> kaggle_3m
    --->patient_1
    --->patient_2
    --->...

## Usage

Model **training**:

```bash
python src/unet_training.py
```

Model **evaluation**:

```bash
python src/evaluate_model.py
```

Make **prediction** on whole dataset:

```bash
python src/predict.py
```

## Example output

### Binary IoU metric for unet

Train set: 0.9047
Test set: 0.8806

### One patient data (0.5 > threshold)

[![Example one patient data](https://i.imgur.com/ft3xywn.png "Example one patient data")](https://i.imgur.com/ft3xywn.png "Example one patient data")
