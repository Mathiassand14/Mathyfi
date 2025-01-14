import cv2
import numpy as np
import PIL.Image
from PIL import Image
from numpy.typing import NDArray
from typing import Tuple


def img2array(path: str, size: Tuple[int,int] | bool = False, normalized: bool = True) ->\
        (NDArray[np.float16] | NDArray[np.uint8]):
    """
    Convert a png file to a numpy array.
    """
    img: PIL.Image.Image = None
    with Image.open(path) as img:
        #grayscale
        img: PIL.Image.Image = img.convert('L')
        
        if size is not False:
            img: PIL.Image.Image = img.resize(size = size)
            
        img: NDArray[np.uint8] = np.array(img)
        
        if normalized:
            img: NDArray[np.float16] = np.float16(img) / 255.0
        return img


def letterbox_bw(
        img,              # Single-channel NumPy array, shape (H, W), with 0=black and 1=white
        desired_width,
        desired_height,
        interpolation=cv2.INTER_LINEAR,
        pad_value=1.0     # 1.0 = white
):
    """Letterbox a single-channel 0/1 image to (desired_height x desired_width).
       Maintains aspect ratio by scaling, then pads with 'pad_value' (white).
       
       Returns a float32 array with shape (desired_height, desired_width).
       Pixel values may be between 0 and 1 if interpolation != INTER_NEAREST.
    """
    # Original dimensions
    orig_height, orig_width = img.shape[:2]
    
    # Compute scaling factor
    ratio_w = desired_width / orig_width
    ratio_h = desired_height / orig_height
    scale = min(ratio_w, ratio_h)  # maintain aspect ratio
    
    # Compute new scaled dimensions
    new_width = int(round(orig_width * scale))
    new_height = int(round(orig_height * scale))
    
    # Resize the image (convert to float32 first to avoid issues with some cv2 operations)
    # If you want purely 0/1 (no fractional grays), you could use INTER_NEAREST.
    resized_img = cv2.resize(
        img.astype(np.float32),
        (new_width, new_height),
        interpolation=interpolation
    )
    
    # Create a new image filled with 'pad_value' (white).
    # shape = (desired_height, desired_width)
    letterboxed = np.full(
        (desired_height, desired_width),
        pad_value,
        dtype=np.float32
    )
    
    # Compute offsets to center the resized image
    offset_x = (desired_width - new_width) // 2
    offset_y = (desired_height - new_height) // 2
    
    # Place the resized image onto our white canvas
    letterboxed[offset_y:offset_y+new_height, offset_x:offset_x+new_width] = resized_img
    
    return letterboxed





if __name__ == "__main__":
    print(img2array("FileHandeling/inkml2Png/000bb64516a3f3ac.png", (10, 10)).shape)
    print(img2array("FileHandeling/inkml2Png/000bb64516a3f3ac.png", (10, 10)))
    print(img2array("FileHandeling/inkml2Png/000bb64516a3f3ac.png", (10, 10)).dtype)