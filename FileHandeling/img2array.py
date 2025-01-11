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








if __name__ == "__main__":
    print(img2array("FileHandeling/inkml2Png/000bb64516a3f3ac.png", (10, 10)).shape)
    print(img2array("FileHandeling/inkml2Png/000bb64516a3f3ac.png", (10, 10)))
    print(img2array("FileHandeling/inkml2Png/000bb64516a3f3ac.png", (10, 10)).dtype)