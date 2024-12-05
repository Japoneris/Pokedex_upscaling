"""
Script to automatically generate readme images.

TODO
"""

import argparse
import os


import cv2
import numpy as np
import tensorflow as tf

import tools as tl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("network", type=str, help="Neural network path")
    parser.add_argument("image", type=str, help="Image to upscale")
    parser.add_argument("--save_path", type=str, default="img/examples/", help="Save location")
    args = parser.parse_args()

    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    
    model = tf.keras.models.load_model(args.network)
    model.summary()
    px = tl.get_model_decay(model)
    px2 = px//2

    img = cv2.imread(args.image)
    img = img.astype(float)/255
    border_color = float(img[:3].mean())
    
    # Save the raw image to compare easily
    poke_ID = args.image.rsplit("/", 1)[-1][:-4]
    save_folder = os.path.join(save_path, poke_ID)
    os.makedirs(save_folder , exist_ok=True)
    cv2.imwrite(os.path.join(save_folder, "raw.png"), tl.img_1_to_255(img))
    

    loop = 6 # Max number
    for scale in [1.25, 1.5, 1.75, 2]:
        print("=== Scale factor: {} ===".format(scale))
        imgz = img.copy()
        for lid in range(loop):
            print("\tloop {}".format(lid))
            imgz = cv2.copyMakeBorder(imgz, px2, px2, px2, px2, cv2.BORDER_CONSTANT, None, [border_color] * 3)
    
            s0, s1, _ = imgz.shape
            s00, s11 = int(s0 * scale), int(s1 * scale)
            imgx = cv2.resize(imgz, (s11, s00))
            print("\t\tInit size:  {} \tx {}".format(s0, s1))
            print("\t\tFinal size: {} \tx {}".format(s00, s11))
            if s00 > 2000:
                print("\tSTOP: image too large to be processed")
                break

            cv2.imwrite(os.path.join(save_folder, "F{}_L{}_raw.png".format(scale, lid+1)), tl.img_1_to_255(imgx))
            imgz = model(imgx[np.newaxis]).numpy()[0].clip(0, 1)
    
            cv2.imwrite(os.path.join(save_folder, "F{}_L{}_clean.png".format(scale, lid+1)), tl.img_1_to_255(imgz))

            # Final pass to sharpen
            imgz1 = cv2.copyMakeBorder(imgz, px2, px2, px2, px2, cv2.BORDER_CONSTANT, None, [border_color] * 3)
            imgz1 = model(imgz1[np.newaxis]).numpy()[0].clip(0, 1)
            cv2.imwrite(os.path.join(save_folder, "F{}_L{}_fine.png".format(scale, lid+1)), tl.img_1_to_255(imgz1))
            if s00 > 1000:
                print("\tSTOP: too large")
                break

        print()
