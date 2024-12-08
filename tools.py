import os

import cv2
import numpy as np

def load_single_image(filename):
    """Load image from path

    If PNG with alpha layer, alpha is replaced by white background.
    :rparma: image scaled in [0, 1] 
    """
    
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED).astype(np.float32)
    if img.shape[2] == 3:
        # JPG / PNG with no alpha
        return img / 255.
    
    else:
        return img[:,:,:3] / 255.
        
        # Select all pixels with alpha != max
        msk = 255. - img[:,:,3]
        
        for i in range(3):
            img[:,:, i] += msk

        
        return img[:,:,:3].clip(0., 255.) / 255.
    

def load_images_from_path(path):
    """
    Load images located on a specific path.

    For png image, alpha layer is converted to white
    """
    files = os.listdir(path)
    files = list(filter(lambda x: x[-4:] in [".png", ".jpg"], files))
    
    lst = []
    for file in files:
        filename = os.path.join(path, file)
        lst.append(load_single_image(filename))
    
    return lst
    


def downscale_images(lst_img, f=2, version="v1"):
    lst = []
    fx = None
    if version == "v1":
        fx = lambda x: downscale_image_v1(x, f)
    
    else:
        fx = lambda x: downscale_image_v2(x, f)

    return list(map(fx, lst_img))


def downscale_image_v1(img, f=2.):
    """Reduce quality using compression:
    Large -> Small -> Large
    """
    # Compute intermediate shape
    s0, s1 = img.shape[:2]
    s00, s11 = int(round(s0 / f)), int(round(s1 / f))

    # Rescale
    img_tmp = cv2.resize(img, (s11, s00))
    return cv2.resize(img_tmp, (s1, s0))

def downscale_image_v2(img, b=2):
    """Reduce quality using bluring:
    Large -> Small -> Large

    :param img: input image
    :param b: blur width (1: no blur)
    :rparam img: deblurred image
    """
    return cv2.blur(img, (b, b))


def downscale_pixel_list(lst, f):
    return list(map(lambda x: downscale_pixel(x, f), lst))
    
def downscale_pixel(img, f):
    s0, s1 = img.shape[:2]
    img1 =  cv2.resize(img, (int(s1/f), int(s0/f)), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(img1, (s1, s0), interpolation=cv2.INTER_NEAREST)


def get_complementary_patches(lst_img_1, lst_img_2, patch_size=50, k=10):
    """
    Avoid blurring operations + prevent border effect
    
    :param lst_img_1, lst_img_2: list of images (paired)
    :param patch_size: size of a square to extract from image
    :param k: number of patch per image
    :rparam: list_1, list_2 of patches
    """
    lst_1, lst_2 = [], []
    for img_1, img_2 in zip(lst_img_1, lst_img_2):
        s0, s1 = img_1.shape[:2]
        for _ in range(k):
            i = np.random.randint(s0 - patch_size)
            j = np.random.randint(s1 - patch_size)
            P1 = img_1[i:i+patch_size, j:j+patch_size]
            
            if P1.std() == 0: # Same color everywhere
                continue
            # Else, accept the patch
            lst_1.append(P1)
            lst_2.append(img_2[i:i+patch_size, j:j+patch_size])


    return np.array(lst_1), np.array(lst_2)


def split_dataset(X, p=0.7):
    """Split data into train and test
    
    :param X: list of items to split
    :param p: percentage for training
    :rparam: (train list, test list)
    """
    l = len(X)
    l_split = int(l* p)
    x = np.arange(l)
    np.random.shuffle(x)
    return [X[i] for i in x[:l_split]], [X[i] for i in x[l_split:]] 

def img_1_to_255(img):
    """Convert float32 images [0, 1] to [0, 255]
    """
    return (img * 255).round().clip(0, 255).astype(np.uint8)
    
    
def check_process(X, Y, model, n=10, label="", save_path="img/train/"):
    """
    Save images during training to see how it goes

    :param X: blurred images
    :param Y: Ground truth
    :param model: CNN to convert images
    :param n: number of images to select
    :param label: image base title
    :param save_path: save location
    """
    os.makedirs(save_path, exist_ok=True)
    
    ID = np.arange(len(X))
    ID = np.random.choice(ID, size=n)

    X_id, Y_id, Z_id = X[ID], Y[ID], model(X[ID]).numpy()
    # Transform images from [0, 1] to [0, 255]
    X_id = list(map(img_1_to_255, X_id))
    Y_id = list(map(img_1_to_255, Y_id))
    Z_id = list(map(img_1_to_255, Z_id))
    
    for i in range(n):
        cv2.imwrite("{}{}_{}_blured.png".format(save_path, label, i), X_id[i])
        cv2.imwrite("{}{}_{}_raw.png".format(save_path, label, i), Y_id[i])
        cv2.imwrite("{}{}_{}_upscaled.png".format(save_path, label, i), Z_id[i])

    return

def crop_image(img, dx2):
    """Crop a single image
    
    :param img: image to cut
    :param dx2: pixels to remove left / right / up / down from image
    :rparam: cropped image
    """
    return img[dx2:-dx2,dx2:-dx2] 
    
def crop_patches(X, dx):
    """
    :param X: array of images
    :param output_shape: output size of the network
    :rparam: Cropped X
    """
    dx2 = dx//2
    return np.array(list(map(lambda img: crop_image(img, dx2), X)))

color_exchanges = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 1, 0],
    [2, 0, 1]
]

def rotate_image(img):
    """Change color order to increase dataset
    """
    return list(map(lambda x: img[:,:, x], color_exchanges))

def data_augmentation_using_color(lst_img):
    """Swap colors to get more images
    """
    lst = []
    for img in lst_img:
        lst.extend(rotate_image(img))
    
    return lst

