"""
Script used to train a neural network.
"""

import argparse
import json
import time
import os

import tensorflow as tf
import numpy as np

from network import Converter
import tools as tl


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a single neural network to upscale pokemon images")
    parser.add_argument("img_path", type=str, help="Folder of images to process.")
    parser.add_argument("--label", type=str, default="", help="Additional network name")
    parser.add_argument("--scale", type=float, default=2., help="Scale factor")
    parser.add_argument("--patch_size", type=int, default=50, help="Size of a patch.")
    parser.add_argument("--k_patch", type=int, default=10, help="Number of patch per image per epoch.")
    parser.add_argument("--n_epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--save_path", type=str, default="NN/", help="Folder where the NN should be saved.")
    parser.add_argument('--nn_filters', nargs="+", type=int, default=[32, 32],
            help="number of filters for the layers")
    parser.add_argument('--nn_kernels', nargs="+", type=int, default=[7, 1, 3],
            help="Kernel size of filters for all layers + last")

    args = parser.parse_args()
    
    n_epochs = args.n_epochs
    p_size = args.patch_size # Patch size 
    k = args.k_patch # Number of patch to extract per image
    
    filter_count = args.nn_filters
    filter_size  = args.nn_kernels
    
    # Check if NN config is valid
    
    assert(len(filter_count) == len(filter_size)-1)
    config = [x for x in zip(filter_count, filter_size[:-1])]
    network_name = 'NN_{}_{}{}'.format(config, filter_size[-1], args.label)
    


    # Define the network. Assume RGB images
    model = Converter((p_size, p_size, 3), conv_config=config, out_kernel=filter_size[-1])
    optimizer = tf.keras.optimizers.Adam(0.001)

    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())
    model.network.summary()

    # Because there is no padding, output is cropped
    px = tl.get_model_decay(model.network)
    print("Pixel cropped: {}".format(px))
    assert(px % 2 == 0, "The convoltional kernel should remove an even number of pixel")
    
    # Create folder if it does not exist
    os.makedirs(args.save_path, exist_ok=True)

    # X: Blurry version
    Dataset = tl.load_images_from_path(args.img_path)
    Y_train, Y_test = tl.split_dataset(Dataset)
    Y_train = tl.data_augmentation_using_color(Y_train) # color augmentation
    X_train = tl.downscale_images(Y_train, f=args.scale, version="v1")
    X_test  = tl.downscale_images(Y_test, f=args.scale, version="v1")

    t0 = time.time()
    for epoch in range(n_epochs):
        print("=== Epoch {} / {} ===".format(epoch + 1, n_epochs))
        if epoch + 1 == 20:
            optimizer.learning_rate.assign(0.0002)
            print("Change learning rate to:", optimizer.learning_rate)

        X_train_sub, Y_train_sub = tl.get_complementary_patches(X_train, Y_train, p_size, k)
        X_test_sub, Y_test_sub   = tl.get_complementary_patches(X_test, Y_test, p_size, k)

        Y_train_sub = tl.crop_patches(Y_train_sub, px)
        Y_test_sub = tl.crop_patches(Y_test_sub, px)

        model.fit(X_train_sub, Y_train_sub, epochs=1, shuffle=True,
                 validation_data=(X_test_sub, Y_test_sub))

        # During the training process, generate samples to see how the process is going on
        tl.check_process(X_test_sub, Y_test_sub, model, n=10, label="epoch_{}".format(epoch+1))
    
    t1 = time.time()
    print("Training done in {:.4} minutes".format((t1-t0)/60))


    # Save the model
    # Do not include optimizer to save space
    filepath = "{}{}_E{}.keras".format(args.save_path, network_name, n_epochs)
    print("Will save the model at:")
    print("\t", filepath)
    model.network.save(filepath, include_optimizer=False)
    print("Done")



    
        

