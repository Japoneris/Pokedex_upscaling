"""
The network class is defined here
"""


from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential


class Converter(Model):
    def __init__(self, input_shape=(50, 50, 3), conv_config=[(32, 7), (64, 1)], out_kernel=3):
        """Upscaling network. 
        
        :param input_shape: size of a patch + dimension (RGB: 3, Gray: 1)
        :param conv_config: list of tuples (n_filters, width)
        :param out_kernel: kernel size of the last layer
        """
        super(Converter, self).__init__()

        # Setup model architecture
        # Conv: number of filter, kernel size
        lst_layers = [layers.Input(shape=input_shape)]
        for n_filters, kernel_size in conv_config:
            lst_layers.append(layers.Conv2D(n_filters, kernel_size, activation="relu", padding="valid")) # same: no loss of size, valid: prevent border effect

        lst_layers.append(layers.Conv2D(input_shape[-1], out_kernel, activation="relu", padding="valid")) 

        self.network = Sequential(lst_layers)
        return
        
    def call(self, x):
        """
        :param x: one/several images. Should be the result of a bicubic upscaling
        """
        return self.network(x)
