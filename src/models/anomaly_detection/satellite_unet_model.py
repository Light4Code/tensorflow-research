# https://github.com/karolzak/keras-unet
# https://deepsense.ai/deep-learning-for-satellite-imagery-via-image-segmentation/
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (BatchNormalization, Conv2D,
                                     Conv2DTranspose, Input, MaxPooling2D,
                                     UpSampling2D, concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

from models.base_model import BaseModel


def iou(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

def threshold_binarize(x, threshold=0.5):
    ge = tf.greater_equal(x, tf.constant(threshold))
    y = tf.where(ge, x=tf.ones_like(x), y=tf.zeros_like(x))
    return y

def iou_thresholded(y_true, y_pred, threshold=0.5, smooth=1.):
    y_pred = threshold_binarize(y_pred, threshold)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

class SatelliteUnetModel(BaseModel):
  def __init__(self, config):
    super().__init__(config)
  
  def bn_conv_relu(self, input, filters, bachnorm_momentum, **conv2d_args):
      x = BatchNormalization(momentum=bachnorm_momentum)(input)
      x = Conv2D(filters, **conv2d_args)(x)
      return x

  def bn_upconv_relu(self, input, filters, bachnorm_momentum, **conv2d_trans_args):
      x = BatchNormalization(momentum=bachnorm_momentum)(input)
      x = Conv2DTranspose(filters, **conv2d_trans_args)(x)
      return x

  def create_optimizer(self):
      self.optimizer_name = 'sgd'
      self.optimizer = SGD(lr=self.config.learning_rate, momentum=0.99)

  def compile(self, loss='binary_crossentropy'):
    self.model.compile(optimizer=self.optimizer, loss=loss, metrics=[iou, iou_thresholded])

  def create_model(self):
      input_shape = self.config.input_shape
      num_classes=1
      output_activation='sigmoid'
      num_layers=4
      filters = 64
      upconv_filters = 96
      kernel_size = (3,3)
      activation = 'relu'
      strides = (1,1)
      padding = 'same'
      kernel_initializer = 'he_normal'

      inputs = Input(input_shape)   
      
      conv2d_args = {
          'kernel_size':kernel_size,
          'activation':activation, 
          'strides':strides,
          'padding':padding,
          'kernel_initializer':kernel_initializer
          }

      conv2d_trans_args = {
          'kernel_size':kernel_size,
          'activation':activation, 
          'strides':(2,2),
          'padding':padding,
          'output_padding':(1,1)
          }

      bachnorm_momentum = 0.01

      pool_size = (2,2)
      pool_strides = (2,2)
      pool_padding = 'valid'

      maxpool2d_args = {
          'pool_size':pool_size,
          'strides':pool_strides,
          'padding':pool_padding,
          }
      
      x = Conv2D(filters, **conv2d_args)(inputs)
      c1 = self.bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)    
      x = self.bn_conv_relu(c1, filters, bachnorm_momentum, **conv2d_args)
      x = MaxPooling2D(**maxpool2d_args)(x)

      down_layers = []

      for l in range(num_layers):
          x = self.bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
          x = self.bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
          down_layers.append(x)
          x = self.bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
          x = MaxPooling2D(**maxpool2d_args)(x)

      x = self.bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
      x = self.bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
      x = self.bn_upconv_relu(x, filters, bachnorm_momentum, **conv2d_trans_args)

      for conv in reversed(down_layers):        
          x = concatenate([x, conv])  
          x = self.bn_conv_relu(x, upconv_filters, bachnorm_momentum, **conv2d_args)
          x = self.bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
          x = self.bn_upconv_relu(x, filters, bachnorm_momentum, **conv2d_trans_args)

      x = concatenate([x, c1])
      x = self.bn_conv_relu(x, upconv_filters, bachnorm_momentum, **conv2d_args)
      x = self.bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
            
      outputs = Conv2D(num_classes, kernel_size=(1,1), strides=(1,1), activation=output_activation, padding='valid') (x)       
      
      self.model = Model(inputs=[inputs], outputs=[outputs])
