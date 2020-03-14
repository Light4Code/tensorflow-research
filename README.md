# tensorflow-research

## Goal

The goal of this repository is to research TensorFlow for machine vision industrial usage. <br/>
It will provide easy to use methods to train models from scratch and also fine tune them afterwards.

## Use
### Train
The "train.py" script is consuming a configuration file (you will find some predefined in "config" folder).
```bash
python src/train.py config/sample.json
```
### Predict
The "predict.py" script will predict and plot the rsesults.
```bash
python src/predict.py config/sample.json --checkpoint_path PATH_TO_CHECKPOINT --test_files_path PATH_TO_THE_IMAGES_TO_PREDICT
```

### Export model
The "export.py" script will save the models as "SavedModel" format and optional also save the frozen graph.

```bash
python src/export.py config/sample.json --checkpoint_path PATH_TO_CHECKPOINT --output_path DIRECTORY_TO_SAVE_MODEL --save_frozen_graph OPTIONAL_DEFAULT_FALSE
```

## Models
### Anomaly detection
Anomaly detection will use a autoencoder approche, the prediction substratced from the original image should show the anomaly.
- Deep Autoencoder: Simple 1D autoencoder
- Custom Convolutional Autoencoder
### Segmentation
#### Unet
- [Vanilla Unet](https://arxiv.org/pdf/1505.04597.pdf) (original paper)
- [Custom/Satellite Unet](https://github.com/karolzak/keras-unet)

|              Model               |       Group       |          Config           |
| :------------------------------: | :---------------: | :-----------------------: |
|   Convolutional Classification   |  Classification   |   `conv_classification`   |
|               ---                |        ---        |            ---            |
|         Deep Autoencoder         | Anomaly Detection |    `deep_autoencoder`     |
|    Convolutional Autoencoder     | Anomaly Detection |    `conv_autoencoder`     |
| Custom Convolutional Autoencoder | Anomaly Detection | `custom_conv_autoencoder` |
|               ---                |        ---        |            ---            |
|           Vanilla Unet           |   Segmentation    |      `vanilla_unet`       |
|          Satellite Unet          |   Segmentation    |     `satellite_unet`      |
|           Custom Unet            |   Segmentation    |       `custom_unet`       |