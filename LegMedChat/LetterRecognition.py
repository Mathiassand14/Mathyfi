import os
from typing import List, Tuple

import pandas as pd
import torch
import torchmetrics
from matplotlib import pyplot as plt
from tqdm import tqdm
from FileHandeling.Data_HMSD import defalt_dict
from Options.OptionsUser import OptionsUser as option
from neural import neural

class LetterRecognition(neural):
    def __init__\
        (
            self,
            batch_size: int = 1024,
            num_epochs: int = 1000,
            learning_rate: float = 0.0001,
            dropout_rate: float = 0.5,
            num_classes: int = 101,
        ) -> None:
        # Hyperparameters and dataset paths
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.train_path = option.PathToLetterTrain
        self.test_path = option.PathToLetterTest
        self.validation_path = option.PathToLetterValidation
        
        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Metrics setup
        self.accuracy_metric = torchmetrics.classification.Accuracy(
            task = "multiclass", num_classes = num_classes
        ).to(self.device)
        self.train_accuracy_metric = torchmetrics.classification.Accuracy(
            task = "multiclass", num_classes = num_classes
        ).to(self.device)
        
        # Model, Loss, and Optimizer
        self.net = self.build_model().to(self.device)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = self.learning_rate)
    
    @staticmethod
    def build_model() -> torch.nn.Sequential:
        return torch.nn.Sequential(
            # --- Block 1 ---
            torch.nn.Conv2d(1, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            # Potentially add a light dropout here if you want, e.g.:
            torch.nn.Dropout(p=0.1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            
            # --- Block 2 ---
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            # Light dropout for second block
            torch.nn.Dropout(p=0.2),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            
            # --- Block 3 ---
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            # Light dropout for third block
            torch.nn.Dropout(p=0.3),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            
            # --- Flatten + FC ---
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 5 * 5, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            # Heavier dropout on fully-connected layer
            torch.nn.Dropout(p=0.5),
            
            torch.nn.Linear(256, 101),
        )
    
    
    @staticmethod
    def parse_csv_labels(cls_path: str) -> dict:
        """Parses class labels from the csv file."""
        class_labels = defalt_dict()
        
        if os.path.exists(cls_path):
            with open(cls_path, "r") as class_label:
                lines = class_label.read().split("\n")[1:]
                for line in lines:
                    if line.strip() == "":
                        continue
                    if line[0] == ",":
                        label = ","
                        int_ = line[2:]
                    else:
                        label, int_ = line.split(",")
                    class_labels[label] = int(int_)
        return class_labels
    
    @staticmethod
    def data_loader(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loads images and their labels from a csv file."""
        images = []
        labels = []
        cls_path = os.path.join(os.getcwd(), "FileHandeling", "cls_labels.csv")
        class_labels = LetterRecognition.parse_csv_labels(cls_path)
        
        with open(path, "r") as file:
            lines = file.read().split("\n")[1:]
        
        for line in tqdm(lines, desc = "Loading data"):
            if line.strip() == "":
                continue
            # Parse each line
            if line[0] != ",":
                label, shape0, shape1, *array = line.split(",")
            else:
                label = ","
                shape0, shape1, *array = line[2:].split(",")
            
            # Cleaning and converting data
            shape0, shape1 = int(shape0.strip("(")), int(shape1.strip(")"))
            array = [
                float("".join(c for c in value if c.isdigit() or c == "."))
                for value in array
            ]
            
            # Create image tensor and append
            image = torch.tensor(array).reshape(shape0, shape1)
            images.append(image)
            labels.append(class_labels[label])
        
        return torch.stack(images).float(), torch.tensor(labels)
    
    def train(self):
        """Train the CNN model with the training dataset."""
        # Load training and validation data
        train_images, train_labels = self.data_loader(self.train_path)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_images, train_labels),
            batch_size = self.batch_size,
            shuffle = True,
        )
        
        val_images, val_labels = self.data_loader(self.validation_path)
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(val_images, val_labels),
            batch_size = self.batch_size,
            shuffle = True,
        )
        
        # Prepare accuracy log
        with open("accuracy.csv", "w") as acc:
            acc.write("epoch,accuracy,train_accuracy\n")
        
        # Training loop
        for epoch in tqdm(range(self.num_epochs), desc = "Training"):
            self.net.train()
            epoch_loss, train_accuracy = 0, 0
            self.accuracy_metric.reset()
            
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.unsqueeze(1).to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward pass, loss, backward pass
                outputs = self.net(x_batch)
                loss = self.loss_function(outputs, y_batch)
                epoch_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                self.train_accuracy_metric.update(outputs, y_batch)
            
            # Validation phase
            self.net.eval()
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val = x_val.unsqueeze(1).to(self.device)
                    y_val = y_val.to(self.device)
                    outputs = self.net(x_val)
                    self.accuracy_metric.update(outputs, y_val)
            
            # Write epoch metrics
            val_accuracy = self.accuracy_metric.compute().item()
            train_accuracy = self.train_accuracy_metric.compute().item()
            with open("accuracy.csv", "a") as acc:
                acc.write(f"{epoch},{val_accuracy},{train_accuracy}\n")
            
            print(
                f"Epoch {epoch + 1}/{self.num_epochs}, "
                f"Train Accuracy: {train_accuracy:.4f}, "
                f"Val Accuracy: {val_accuracy:.4f}"
            )
            
            torch.save(self.net.state_dict(), os.path.join(option.PathToLetterNet,f"net{epoch}.pt"))
            
            # Reset metrics after each epoch
            self.accuracy_metric.reset()
            self.train_accuracy_metric.reset()
            
            df = pd.read_csv("accuracy.csv")
            df.plot(x = "epoch", y = ["accuracy","train_accuracy"])
            plt.show()
    
    def test(self):
        """Test the CNN model on the test dataset."""
        test_images, test_labels = self.data_loader(self.test_path)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(test_images, test_labels),
            batch_size = self.batch_size,
            shuffle = False,
        )
        
        # Reset metric and evaluate the model
        self.net.eval()
        self.accuracy_metric.reset()
        
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.unsqueeze(1).to(self.device)
                y_batch = y_batch.to(self.device)
                outputs = self.net(x_batch)
                self.accuracy_metric.update(outputs, y_batch)
        
        test_accuracy = self.accuracy_metric.compute().item()
        print(f"Test Accuracy: {test_accuracy:.4f}")
    
    def load(self, path: str):
        """Load pre-trained model weights."""
        if os.path.exists(path):
            self.net.load_state_dict(torch.load(path))
            self.net.eval()
            print(f"Model successfully loaded from: {path}")
        else:
            raise FileNotFoundError(f"No model file found at: {path}")
    
    def eval(self, input_data: torch.Tensor) -> int:
        """
        Evaluate the input tensor using the trained model.

        Args:
            input_data (torch.Tensor): A single input tensor (e.g., an image) to evaluate,
                                       with shape (Height, Width) or (Channels, Height, Width).

        Returns:
            int: Predicted label (class) for the input.
        """
        
        # Ensure the model is in evaluation mode
        self.net.eval()
        
        # If input_data lacks a batch dimension, add it
        if input_data.ndim == 2:  # Expects a single-channel image
            input_data = input_data.unsqueeze(0).unsqueeze(0)  # (Height, Width) -> (1, 1, Height, Width)
        elif input_data.ndim == 3:  # If shape is (Channels, Height, Width)
            input_data = input_data.unsqueeze(0)  # (Channels, Height, Width) -> (1, Channels, Height, Width)
        
        # Transfer input to the appropriate device
        input_data = input_data.to(self.device)
        
        # Forward pass through the model
        with torch.no_grad():
            outputs = self.net(input_data)
        
        # Get the predicted class (highest probability)
        predicted_label = torch.argmax(outputs, dim = 1).item()
        
        return predicted_label

if __name__ == "__main__":
    # Initialize the model
    model = LetterRecognition()
    
    # Train the model
    model.train()
    
   