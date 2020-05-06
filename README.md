# tensorflow-research

## Goal

The goal of this repository is to research TensorFlow for machine vision industrial usage. <br/>
It will provide easy to use methods to train models from scratch and also fine tune them afterwards.

## Starting point
To quickly get into the usage you can look into the notebooks, they provide the whole training pipline step by step.

## Backbonses
Backbones provide you with a ready to use model architectur.
### Anomaly detection
Anomaly detection will use a autoencoder approche, the prediction substratced from the original image should show the anomaly.

### Segmentation
#### Unet
- [Vanilla Unet](https://arxiv.org/pdf/1505.04597.pdf) (original paper)
- [Custom/Satellite Unet](https://github.com/karolzak/keras-unet)

|           Class            |       Group       |           Backbone            |            Sample notebook            |
|:--------------------------:|:-----------------:|:-----------------------------:|:-------------------------------------:|
|    `ClassificationConv`    |  Classification   |     `classification_conv`     |                                       |
|                            |        ---        |              ---              |                                       |
|     `AutoEncoderConv`      | Anomaly Detection |      `auto_encoder_conv`      | Choco waffle anomaly detection sample |
| `AutoEncoderFullConnected` | Anomaly Detection | `auto_encoder_full_connected` |                                       |
|                            |        ---        |              ---              |                  ---                  |
| `SegmentationVanillaUnet`  |   Segmentation    |  `segmentation_vanilla_unet`  |    Wood plate segmentation sample     |
|            TODO            |   Segmentation    |       `satellite_unet`        |                                       |
|            TODO            |   Segmentation    |         `custom_unet`         |                                       |