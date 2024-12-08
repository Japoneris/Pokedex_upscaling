"""
The network class is defined here
"""


from tensorflow.keras import layers, saving
from tensorflow.keras.models import Model, Sequential

def get_conv_fx(conv_type="conv2D"):
    """Select between normal convolution or separable convolution
    
    """
    if conv_type == "conv2D":
        return layers.Conv2D
    elif conv_type == "separable":
        return layers.SeparableConv2D
    else:
        assert(False, "unknown convolution name")
        
class SRCNN(Model):
    def __init__(self, input_shape=(50, 50, 3), conv_config=[(32, 7), (64, 1)], out_kernel=3, conv_type="conv2D", old_layers=None):
        """Upscaling network. 
        
        :param input_shape: size of a patch + dimension (RGB: 3, Gray: 1). The most important is the last dimension.
        :param conv_config: list of tuples (n_filters, kernel_width)
        :param out_kernel: kernel_width of the last layer
        """
        super(SRCNN, self).__init__()

        if old_layers is not None:
            # Load from saved model
            self.nn_layers = old_layers
            return 
            
        # Function to create layers
        fx_layer = get_conv_fx(conv_type)
        
        
        # Setup model architecture
        # conv_config: number of filter, kernel size
        self.nn_layers = [] 
        self.decay = 0
        for n_filters, kernel_size in conv_config:
            self.nn_layers.append(fx_layer(n_filters, kernel_size, activation="relu", padding="valid")) 
            self.decay += kernel_size - 1
        
        # Last layer with fixed output dimension
        self.nn_layers.append(fx_layer(input_shape[-1], out_kernel, activation="relu", padding="valid")) 
        self.decay += out_kernel - 1
        
        return
        
    def call(self, x):
        """
        :param x: one/several images. Should be the result of a bicubic upscaling
        """
        for lay in self.nn_layers:
            x = lay(x)
        
        return x

    def get_config(self):
        base_config = super().get_config()
        
        config = {}
        config["n_layers"] = len(self.nn_layers)
        config["decay"] = self.decay

        
        for idx, lay in enumerate(self.nn_layers):
            config["conv_{}".format(idx)] = saving.serialize_keras_object(lay) 

        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        """
        Needed to export(+load) the model correctly
        """

        n_layers = saving.deserialize_keras_object(config.pop("n_layers"))
        
        lst_conv = []
        for idx in range(n_layers):
            lay_conf = config.pop("conv_{}".format(idx))
            lst_conv.append(saving.deserialize_keras_object(lay_conf))

        md = cls(old_layers=lst_conv) # Init the model with previous layers
        md.decay = saving.deserialize_keras_object(config.pop("decay"))
        return md

class VDSR(Model):
    """Convolutional network + residual block
    """
    def __init__(self, dims=3, n_layers=5, n_filters=32, conv_type="separable", old_layers=None):
        """
        :param convtype: "conv2D" or "separable" (more efficient)
        :param dims: 3 RGB, 1: Grayscale, 4: PNG (RGB+alpha)
        """
        super(VDSR, self).__init__()

        self.dims = dims
        self.decay = 0

        # Initialization with an already trained network.
        if old_layers is not None:
            self.lst_conv = old_layers
            return
        
        # Function to create layers
        fx_layer = get_conv_fx(conv_type=conv_type)
        
        self.lst_conv = []
        for _ in range(n_layers-1):
            self.lst_conv.append(fx_layer(n_filters, 3, activation="relu", padding="same"))

        # No activation for the last layer
        if n_layers != 0:
            self.lst_conv.append(fx_layer(dims, 3, padding="same"))
            
        return

    def call(self, x):
        """
        :param x: one/several images. Should be the result of a bicubic upscaling
        """
        y = x # make a copy
        for lay in self.lst_conv:
            x = lay(x)
            
        return y + x

    def get_config(self):
        base_config = super().get_config()
        
        config = {}
        config["n_layers"] = len(self.lst_conv)
        config["dims"] = self.dims

        
        for idx, lay in enumerate(self.lst_conv):
            config["conv_{}".format(idx)] = saving.serialize_keras_object(lay) 

        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        """
        Needed to export(+load) the model correctly
        """

        n_layers = saving.deserialize_keras_object(config.pop("n_layers"))
        dims = saving.deserialize_keras_object(config.pop("dims"))
        
        lst_conv = []
        for idx in range(n_layers):
            lay_conf = config.pop("conv_{}".format(idx))
            lst_conv.append(saving.deserialize_keras_object(lay_conf))

        md = cls(dims, old_layers=lst_conv) # Init the model with previous layers
        return md


