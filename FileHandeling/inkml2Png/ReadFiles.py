import os
from typing import Tuple

from sympy import false
from tqdm import tqdm

from FileHandeling.imkml2img.inkml2img import inkml2img

def convert_to_png(filename: str, size: Tuple[int, int] = (200, 50), show: bool = false) -> str:
    """
    Convert a inkml file to a png file. The filename is the name of the pdf file.
    """
    result_filename: str = filename.replace(".inkml", "") + ".png"
    inkml2img(filename, result_filename, size=size, show=show)
    return result_filename

if __name__ == "__main__":
    
    path = "FileHandeling/inkml2Png/000bb64516a3f3ac.inkml"
    convert_to_png(path, (200, 40), True)
    
    #dir = "mathwriting-2024\\train\\"
    #for filename in tqdm(os.listdir(dir), desc="Converting files"):
    #    file = dir + filename
    #    print(file)
    #    if filename.endswith(".inkml"):
    #        r: str = convert_to_png(file)
    #        os.remove(r)

    # delete the file
    