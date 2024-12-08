"""
Script used to train a neural network.
"""

import argparse
import json
import time
import os

import tensorflow as tf
import numpy as np

from network import SRCNN, VDSR
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
    parser.add_argument('--conv', choices=["conv2D", "separable"], default="separable", help="If set, use separable convolution to save space")

    # Add subparser
    subparsers = parser.add_subparsers(dest='command', help='Network architecture', required=True)

    # One subparser per network architectue
    parser_srcnn = subparsers.add_parser('SRCNN', help="SRCNN.")
    parser_srcnn.set_defaults(cmd='srcnn')
    parser_srcnn.add_argument('--nn_filters', nargs="+", type=int, default=[32, 32],
            help="number of filters for the layers")
    parser_srcnn.add_argument('--nn_kernels', nargs="+", type=int, default=[7, 1, 3],
            help="Kernel size of filters for all layers + last")


    parser_vdcr = subparsers.add_parser('VDSR', help="VDSR neural architecture.")
    parser_vdcr.set_defaults(cmd='vdsr')
    parser_vdcr.add_argument('--n_layers', type=int, default=5, help="Number of convolutional layers (all identical)")
    parser_vdcr.add_argument('--n_filters', type=int, default=32, help="Number of convoltuional filters per layers")
    
    args = parser.parse_args()

    # Common parameters
    n_epochs = args.n_epochs
    p_size = args.patch_size # Patch size 
    k = args.k_patch # Number of patch to extract per image

    # Create folder to save neural network (if it does not exist)
    os.makedirs(args.save_path, exist_ok=True)

    ######################
    # Define the network #
    ######################
    network_name = ""
    model = None
    
    if args.cmd == "srcnn":
        print("Setting up an SCRNN")
        filter_count = args.nn_filters
        filter_size  = args.nn_kernels
        crop_px = sum(filter_size) - len(filter_size)
        # Check if NN config is valid
        
        assert(len(filter_count) == len(filter_size)-1)
        config = [x for x in zip(filter_count, filter_size[:-1])]
        config_name = ["n{}k{}".format(x[0], x[1]) for x in config]
        
        network_name = 'SRCNN_{}_n3k{}_{}'.format("_".join(config_name), filter_size[-1], args.label)
        # Define the network. Assume RGB images (3)
        model = SRCNN((p_size, p_size, 3), conv_config=config, out_kernel=filter_size[-1], conv_type=args.conv)
        
    elif args.cmd == "vdsr":
        print("Setting up a VDSR network")
        
        n_layers = args.n_layers
        n_filters  = args.n_filters
        
        network_name = 'VDSR_n{}_f{}_{}'.format(n_layers, n_filters, args.label)
        
        # Define the network. Assume RGB images (3)
        model = VDSR(3, n_layers, n_filters, conv_type=args.conv)

    print("Network name:")
    print("\t", network_name)

    crop_px = model.decay
    print("Pixel cropped: {}".format(crop_px))
    assert(crop_px % 2 == 0, "The convoltional kernel should remove an even number of pixel")

    optimizer = tf.keras.optimizers.Adam(0.001)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())
    
    # Send the model random data to initialize weights and size
    model(np.random.rand(1, p_size, p_size, 3))
    model.summary()
    
    ######################
    # Prepare dataset    #
    # X: Blurry version  #
    # Y: Normal          #
    ######################
    
    Dataset = tl.load_images_from_path(args.img_path)
    Y_train, Y_test = tl.split_dataset(Dataset)
    Y_train = tl.data_augmentation_using_color(Y_train) # color augmentation
    X_train = tl.downscale_images(Y_train, f=args.scale, version="v1")
    X_test  = tl.downscale_images(Y_test, f=args.scale, version="v1")

    
    t0 = time.time()
    for epoch in range(n_epochs):
        print("=== Epoch {} / {} ===".format(epoch + 1, n_epochs))

        X_train_sub, Y_train_sub = tl.get_complementary_patches(X_train, Y_train, p_size, k)
        X_test_sub, Y_test_sub   = tl.get_complementary_patches(X_test, Y_test, p_size, k)

        if crop_px != 0: # Remove border pixel to match f(X) and Y
            Y_train_sub = tl.crop_patches(Y_train_sub, crop_px)
            Y_test_sub = tl.crop_patches(Y_test_sub, crop_px)

        model.fit(X_train_sub, Y_train_sub, epochs=1, shuffle=True,
                 validation_data=(X_test_sub, Y_test_sub))

        # During the training process, generate samples to see how the process is going on
        tl.check_process(X_test_sub, Y_test_sub, model, n=10, label="epoch_{}".format(epoch+1))
    
    t1 = time.time()
    print("Training done in {:.4} minutes".format((t1-t0)/60))

    ##################
    # Save the model #
    ##################
    filepath = "{}{}_E{}.keras".format(args.save_path, network_name, n_epochs)
    print("Will save the model at:")
    print("\t", filepath)
    
    # Do not include optimizer to save space
    model.save(filepath, include_optimizer=False)
    print("Done")



    
        

