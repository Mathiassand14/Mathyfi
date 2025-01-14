import os.path
import random
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
import torchvision.ops as ops

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
            torch.nn.Conv2d(1, 32, 3, padding = 1), # 400*400x32
            torch.nn.Dropout(0.1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2), # 200*25X32
            torch.nn.Conv2d(32, 64, 3, padding = 1), # 200*200X64
            torch.nn.Dropout(0.2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2), # 100*12X64
            torch.nn.Conv2d(64, 128, 3, padding = 1), # 100*100X128
            torch.nn.Dropout(0.3),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2), # 50*50X128
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 50 * 50, 256),
            torch.nn.Dropout(0.5),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 4),
            
        )
    
    @staticmethod
    def data_loader(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
    
    
    def forward(self, input, output_bb: bool = False) -> torch.Tensor | Tuple[torch.Tensor, torch.float, torch.float, torch.float, torch.float]:
        x, y, w, h = self.net(input).items()
        
        img = ops.roi_align(input, [x, y, w, h], output_size = (45, 45))
        
        output = self.LetterRecognition.net(img)
        
        if output_bb:
            return output, x, y, w, h
        return output
    
    def train(self) -> None:
        """Trains the model"""
        
        
        self.LetterRecognition.load(os.path.join(option.PathToLetterModel, "model.pth"))
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in tqdm(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer_l.zero_grad()
                self.optimizer_bb.zero_grad()
                
                outputs = self.forward(images)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer_l.step()
                self.optimizer_bb.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                self.train_accuracy_metric.update(predicted, labels)
            
            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(self.train_loader)}, Accuracy: {100 * correct / total}")
            self.train_accuracy_metric.compute()

