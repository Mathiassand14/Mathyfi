import os.path
from glob import glob
from typing import Tuple

import torch
import torchmetrics
from torch.nn.functional import dropout
from tqdm import tqdm

from FileHandeling.img2array import img2array
from FileHandeling.imkml2img.inkml2img import inkml2img
from neural import neural
from Options.OptionsUser import OptionsUser as option
from LegMedChat.LetterRecognition import LetterRecognition

class ImageBoundrybox(neural):
    def __init__\
            (
                self,
                batch_size: int = 1024,
                num_epochs: int = 1000,
                learning_rate: float = 0.001,
                dropout_rate: float = 0.5,
                num_classes: int = 101,
            ) -> None:
        # Hyperparameters and dataset paths
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        
        self.train_path = os.path.join(option.PathToMathWriting, "train")
        self.test_path = os.path.join(option.PathToMathWriting, "test")
        self.validation_path = os.path.join(option.PathToMathWriting, "validation")
        
        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # letter nn
        self.LetterRecognition = LetterRecognition()
        
        # Metrics setup
        self.accuracy_metric = self.LetterRecognition.accuracy_metric
        self.train_accuracy_metric = self.LetterRecognition.train_accuracy_metric
        
        
        
        
        # Model, Loss, and Optimizer
        self.net = self.build_model().to(self.device)
        self.loss_function = self.LetterRecognition.loss_function
        self.optimizer_bb = torch.optim.Adam(self.net.parameters(), lr = self.learning_rate)
        self.optimizer_l = self.LetterRecognition.optimizer
        
        self.data_loader(self.train_path)
        
    
    def build_model(self) -> torch.nn.Sequential:
        """Defines and returns the model"""
        return torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding = 1), # 400*50x32
            torch.nn.Dropout(self.dropout_rate),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2), # 200*25X32
            torch.nn.Conv2d(32, 64, 3, padding = 1), # 200*25X64
            torch.nn.Dropout(self.dropout_rate),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2), # 100*12X64
            torch.nn.Conv2d(64, 128, 3, padding = 1), # 100*12X128
            torch.nn.Dropout(self.dropout_rate),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2), # 50*6X128
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 50 * 6, 256),
            torch.nn.Dropout(self.dropout_rate),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 4),
            
        )
    
    @staticmethod
    def data_loader(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load data from path"""
        files = glob(os.path.join(path, "*.inkml"))
        print(files)
        images = []
        labels = []
        for file in tqdm(files):
            new_file = file.replace(".inkml", ".png")
            inkml2img(file,new_file,size = (400,50))
            image = img2array(new_file, size = (400,50))
            os.remove(new_file)
            images.append(image)
            with open(file) as f:
                label = f.read().split("\n")[5]
                labels.append(label)
        return torch.stack(images).float(), torch.tensor(labels)
    
    
    def forward(self, x):
        x = self.net(x)
        
        return x
    
if __name__ == "__main__":
    model = ImageBoundrybox.data_loader(os.path.join(option.PathToMathWriting, "train"))
    
else:
    #%%
    from FileHandeling.Lmdb import create_lmdb
    from Options.OptionsUser import OptionsUser as option
    import os
    from glob import glob
    from tqdm import tqdm
    from FileHandeling.img2array import img2array
    from FileHandeling.imkml2img.inkml2img import inkml2img
    
    if not os.path.exists(option.PathToMathWritingLmdbTrain):
        files = glob(os.path.join(option.PathToMathWriting, "train", "*.inkml"))
        print(files)
        data = {}
        for file in tqdm(files):
            new_file = file.replace(".inkml", ".png")
            inkml2img(file,new_file,size = (400,50))
            image = img2array(new_file, size = (400,50))
            os.remove(new_file)
            
            with open(file) as f:
                label = f.read().split("\n")[5].removeprefix('<annotation type="normalizedLabel">').removesuffix('</annotation>')
                #print(file.split("\\")[-1].replace(".inkml", ""), label)
                data[file.split("\\")[-1].replace(".inkml", "")] = (label, image)
        
        create_lmdb(option.PathToMathWritingLmdbTrain, data)
