"""
When neural network is trained, use one to upscale one image
"""

import argparse
import json
import os

import cv2
import numpy as np
import tensorflow as tf

import tools as tl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("network", type=str, help="Neural network path")
    parser.add_argument("image", type=str, help="Image to upscale")
    parser.add_argument("--scale", type=float, default=2., help="Scale factor")
    parser.add_argument("--loop", type=int, default=1., help="Number of scale loop")
    parser.add_argument("--save_path", type=str, default="img/upscaled/", help="Save location")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    print("=== Load Neural Network ===")
    model = tf.keras.models.load_model(args.network)
    model.summary()
    px = tl.get_model_decay(model)
    px2 = px//2
    print("Model size decay:", px)

    print("=== Load Image ===")
    imgz = tl.load_single_image(args.image)
    border_color = float(imgz[0].mean())
    
    print("Init size: {}".format(imgz.shape[:2]))
    print("==> Rescale by F={}".format(args.scale))
    
    for loop in range(args.loop):
        imgz = cv2.copyMakeBorder(imgz, px2, px2, px2, px2, cv2.BORDER_CONSTANT, None, [border_color] * 3)
        s0, s1, _ = imgz.shape
        s00, s11 = int(s0 * args.scale), int(s1 * args.scale)
        print("\tLoop {}: \t({}x{}) => \t ({}x{})".format(loop, s0, s1, s00, s11))
        
        imgx = cv2.resize(imgz, (s11, s00))
        imgy = imgx[np.newaxis]
        imgz = model(imgy).numpy()[0].clip(0, 1)
    
    print("Final size: {}".format(imgz.shape[:2]))
            
    
    imgz = tl.img_1_to_255(imgz)
    
    # TODO: If PNG, maybe gather the mask ?
    basename = args.image.rsplit(".", 1)[0].rsplit("/", 1)[-1]
    filename = args.save_path + "{}_f_{}_loop_{}.jpg".format(basename, args.scale, args.loop)
    print("Upscaled image saved at {}".format(filename))
    cv2.imwrite(filename, imgz)
    


