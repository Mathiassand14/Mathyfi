import os
from glob import glob
from operator import contains
from typing import DefaultDict, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from FileHandeling.img2array import img2array
from Options.OptionsUser import OptionsUser as option
from tqdm import tqdm


class defalt_dict(dict):
    def __init__(self) -> None:
        self.dict: Dict[str, int] = {}
        self.counter: int = 0
    
    def __getitem__(self, key: any):
        if key not in self.dict:
            self.dict[key] = self.counter
            self.counter += 1
        return self.dict[key]
    
    def __setitem__(self, key: any, value: any):
        pass


if __name__ == "__main__":
    dr: str = option.PathToHMSD
    folders: List[Tuple[str, str]] = [(os.path.join(dr, d), d) for d in os.listdir(dr)]
    with open(option.PathToLetterCSV, "w") as images:
        images.write("label,shape,array\n")
    
    with open(option.PathToLetterCSV, "a") as imag:
        [
            (
                img := img2array(filename,size = (45, 45)),
                imag.write(
                    str(
                        filename.split("_")[1].split("\\")[-1]
                        if len(filename.split("_")[1].split("\\")[-1]) == 1
                        else filename.split("\\")[-2]
                    ) +
                    "," + str(img.shape) + "," +
                    np.array2string \
                            (
                            img.flatten(),
                            threshold = img.size,
                            separator = ",",
                            max_line_width = np.inf
                        ) +
                    "\n")
            )
            #for i, label in tqdm(enumerate(folders))
            for filename in tqdm(glob(os.path.join(dr, "*", '*.jpg'), recursive = True))
        ]
