import os
from random import random
from typing import List
from tqdm import tqdm

from Options.OptionsUser import OptionsUser as option

if __name__ == "__main__":
   
    
    
    with open(option.PathToLetterCSV, "r") as images:
        text: str = images.read()
    
    lines: List[str] = text.split("\n")[1:]
    training: List[str] = []
    validation: List[str] = []
    test: List[str] = []
    
    for line in tqdm(lines):
        rand = random()
        if rand < 0.80:
            training.append(line)
        elif rand < 0.95:
            validation.append(line)
        else:
            test.append(line)
    
    with open(option.PathToLetterTrain, "w") as images:
        images.write("letter,shape,array\n")
        images.write("\n".join(training))
        
    with open(option.PathToLetterValidation, "w") as images:
        images.write("letter,shape,array\n")
        images.write("\n".join(validation))
        
    with open(option.PathToLetterTest, "w") as images:
        images.write("letter,shape,array\n")
        images.write("\n".join(test))