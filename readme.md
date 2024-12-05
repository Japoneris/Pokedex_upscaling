# Super-Resolution Pokemons with CNN

Neural network to improve image quality.

We followed the intuition of [SRCNN (Super-Resolution CNN) algorithm](https://arxiv.org/pdf/1501.00092) to upscale Pokemon images.

Nevertheless, we adapt convolutional filters number and size to our target.

Pokedex images can be downloaded [here](https://www.pokebip.com/download/pokedex_offline_2.0.3_avec_images.zip).



## How it works?

### Inference Process

1. You have an image of size `a x b`
2. You upscale it as `a*f x b*f`. This one is blurry. `f` can be an integer (`2`) or a floating value (reco.: `1.5`, 1.75`)
3. You feed the blurry image to the network
4. The network outputs a denoised image

### Training Process 

1. Take your images (this is the `Y`)
2. Downscale the images
3. Rescale them to the original size (they are blurry, this is your `X`)
4. Train the network to learn `X -> Y`

*Note*: Here, the network learns to upscale images **without** having access to **high resolution images**. 

 


## Example

### `ID 003: Venusaur`

Initial image (`157 x 127` pixels):

![](./img/examples/3/raw.png)

Upscaled image x2 (`367 x 300`):

![](./img/examples/3/F1.5_L2_fine.png)

Upscaled image x3.5 (`556 x 456`):

![](./img/examples/3/F1.5_L3_clean.png)

Upscaled image x5.6 (`891 x 731`):

![](./img/examples/3/F1.75_L3_fine.png)


## Training time

Between 3 to 10 minutes for 20 epochs with `Intel® Core™ i7-8850H CPU @ 2.60GHz × 12`

Of course, it depends on the number of layers. Nevertheless, this network can be trained on a regular laptop.


# How to run?

## Dependencies

First, install dependencies with:

`pip install -r requirements.txt`

We use `tensorflow` and `opencv`.


## Run

The repository can be run without training any network as an already trained network is provided in the `NN/` folder.

`python3 extract_readme_images.py "./NN/NN_[(64, 7), (64, 5), (64, 3)]_1_color_E20.keras"  ./dataset/sugimori/363.png`


## Training

You can train a network with the following command:

`python3 train.py ./dataset/sugimori/  --nn_filters 64 64 64 --nn_kernels 7 5 3 1 --n_epochs=20 --label="_color" --patch_size=50`


### Network customization

The network is made of `k+1` layers, the last one reconstructing the final image.

To customize the network, you have two main parameters:

- the number of filters (`--nn_filters` param)
- the size of the filters (`--nn_kernerls` param)

You can customize the `k` first layers as you which. However, for the last one, as we want an RGB image, we need exactly `3` filters, so you do not have to specify it.
Therefore, you need to specify `k` number of filters and `k+1` kernels width (Yet, it is recommended to have a small kernel size for the last layer to avoid "patch effect").

*Note*: the sum of the convolutional kernel size should be odd. Otherwise, the transformed image and expected image cannot be aligned.


### Patch size 

When the layers have been specified, you may play on the `patch_size`.
The dataset images do not have the same size. Therefore, instead of training on the full image, the network is trained on patches (images cropped at random).

Because convolutional layers are applied without padding, the image "shrink" one step after the other.
When selecting the patch size, verify that the number of resulting pixels is sufficient.





## Files description

- `network.py`: Neural network class definition
- `train.py`: Script to launch to train the network
- `infer.py`: Script to run to use a trained neural network
- `tools.py`: Function to load / process data
- `extract_readme_images.py`: Use a neural network to generate different image variations from a single image


# Recommendations

- When upscaling, use a scale factor which is can be written as `k/2` or `k/4`
- Even if the base model is trained with an upscaling factor of `f=2`, it can be preferable to upscale with `f=1.5` several times
- Avoid large upscaling (e.g., when a model learned an upscaling of `2`, avoid upscaling with `4`, as learned kernels cannot find enough info)
- Passing twice the image through the model (without upscaling) can improbe the sharpness
- Avoid passing the image too many times within the model, as colors tend to change

## Recommandation

**TODO**

Show differences between F3 L1 VS F1.75 L2

# TODO

- [ ] Small pixel images:
    - Tested: interpolation with `cv.INTER_NEAREST` to keep a pixel-like image.
    - Problem: Images are too small, and the network struggle to identify patterns / informations
- [ ] Background removal / Alpha mask handling.
    - Tested idea: upscale normally. Problem: blurry border
    - To test: Train a NN to upscale the alpha layer.
- [ ] Illustrate recommendations
- [ ] OS agnostic. Change the `/` by `os.path.join`
