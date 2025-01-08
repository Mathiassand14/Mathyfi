import os
from tqdm import tqdm

from FileHandeling.imkml2img.inkml2img import inkml2img

def convert_to_png(filename: str) -> str:
    """
    Convert a inkml file to a png file. The filename is the name of the pdf file.
    """
    result_filename: str = filename.replace(".inkml", "") + ".png"
    inkml2img(filename, result_filename)
    return result_filename

if __name__ == "__main__":
    
    path = "FileHandeling/inkml2Png/000bb64516a3f3ac.inkml"
    convert_to_png(path)
    
    #dir = "mathwriting-2024\\train\\"
    #for filename in tqdm(os.listdir(dir), desc="Converting files"):
    #    file = dir + filename
    #    print(file)
    #    if filename.endswith(".inkml"):
    #        r: str = convert_to_png(file)
    #        os.remove(r)

    # delete the file
    