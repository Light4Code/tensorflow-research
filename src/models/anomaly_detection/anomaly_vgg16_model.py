from tensorflow.keras.applications import vgg16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


class AnomalyVgg16Model():
  def create(self, input_shape=(256,256,3)):
    vgg_conv = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    # Freeze the layers except the last 4 layers
    for layer in vgg_conv.layers[:-8]:
        layer.trainable = False
    # Check the trainable status of the individual layers
    # for layer in vgg_conv.layers:
    #     print(layer, layer.trainable)
    x = vgg_conv.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(2, activation="softmax")(x)
    model = Model(vgg_conv.input, x)
    print(model.summary())
    return model
