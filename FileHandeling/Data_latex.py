from typing import List

from FileHandeling.Lmdb import create_lmdb, read_entire_lmdb, read_from_lmdb
from Options.OptionsUser import OptionsUser as option
import os
from glob import glob
from tqdm import tqdm
from FileHandeling.img2array import img2array
from FileHandeling.imkml2img.inkml2img import inkml2img
from FileHandeling.img2array import letterbox_bw
import xml.etree.ElementTree as ET
import numpy as np
from skimage.transform import resize


def make_latex_lmdb(type: str, folders: List[str]):
    
    files = []
    for folder in folders:
        files.extend(glob(os.path.join(option.PathToMathWriting, folder, "*.inkml")))
    print(files)
    data = {}
    for file in tqdm(files):
        new_file = file.replace(".inkml", ".png")
        inkml2img(file,new_file,size = (400,50))
        image = img2array(new_file, size = (400,50))
        os.remove(new_file)
        
        with open(file) as f:
            label = f.read().split("\n")[5].removeprefix('<annotation type="normalizedLabel">').removesuffix(
                '</annotation>')
            #print(file.split("\\")[-1].replace(".inkml", ""), label)
            data[file.split("\\")[-1].replace(".inkml", "")] = (label, image)
    
    create_lmdb(os.path.join(option.PathToLatexLmdb, type), data)
if __name__ == "__main__":
    make_latex_lmdb("train.lmdb", ["train"])
    #make_latex_lmdb("test.lmdb", ["test"])
    #make_latex_lmdb("validation.lmdb", ["valid"])



