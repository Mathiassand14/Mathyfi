import os
from typing import List, Tuple

import Levenshtein
import pandas as pd
import torch
import torchmetrics
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from tqdm import tqdm

import FileHandeling.Lmdb

from Options.OptionsUser import OptionsUser as option


class CRNN(torch.nn.Module):
    def __init__\
        (
            self,
            batch_size: int = 512 + 256,
            num_epochs: int = 1000,
            learning_rate: float = 0.0001,
            
        ) -> None:
        super().__init__()
        
        self.symbols = r" !#%&()*+-.,/0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_abcdefghijklmnopqrstuvwxyz{|}<>\""
        
        self.symbols_dict = {}
        for i, symbol in enumerate(self.symbols):
            self.symbols_dict[symbol] = i
        
        # Hyperparameters and dataset paths
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.num_classes = self.num_classes = len(self.symbols)
        self.train_path = os.path.join(option.PathToLatexLmdb, "validation.lmdb")#"train.lmdb")
        self.test_path = os.path.join(option.PathToLatexLmdb, "test.lmdb")
        self.validation_path = os.path.join(option.PathToLatexLmdb, "test.lmdb")#"validation.lmdb")
        
        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Metrics setup
        self.accuracy_metric = torchmetrics.classification.Accuracy(
            task = "multiclass", num_classes = self.num_classes
        ).to(self.device)
        self.train_accuracy_metric = torchmetrics.classification.Accuracy(
            task = "multiclass", num_classes = self.num_classes
        ).to(self.device)
        
        
        
        
        self.conv = torch.nn.Sequential(
            # --- Block 1 ---
            torch.nn.Conv2d(1, 32, 3, padding=1), # 500*50X1
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            # Potentially add a light dropout here if you want, e.g.:
            torch.nn.Dropout(p=0.1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            
            # --- Block 2 ---
            torch.nn.Conv2d(32, 64, 3, padding=1), #250*25X32
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            
            # Light dropout for second block
            torch.nn.Dropout(p=0.2),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            
            # --- Block 3 ---
            torch.nn.Conv2d(64, 128, 3, padding=1), # 125*12X64
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            # Light dropout for third block
            torch.nn.Dropout(p=0.3),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
    
        self.Lstm = torch.nn.LSTM(128, 256, 1, batch_first=True, bidirectional=True) # 64*6X128
    
        self.fc = torch.nn.Sequential(
           
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            # Heavier dropout on fully-connected layer
            torch.nn.Dropout(p=0.5),
            
            torch.nn.Linear(256, self.num_classes + 1),
        )
        
        
        
        # Model, Loss, and Optimizer
        self.criterion = torch.nn.CTCLoss(blank = self.num_classes, zero_infinity = True, reduction = "mean")
        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        
        self.to(self.device)
    
    
    
    
    
    def forward(self, x):
        #print(f"Input to CNN: {x.shape}")
        x = x.unsqueeze(1)
        x = self.conv(x) # Shape: (batch_size, 128, 62, 6)
        b, c, h, w = x.size()
        
        #print(f"Input to CNN: {x.shape}")
        x = x.permute(0, 2, 3, 1)  # Shape: (batch_size, 62, 6, 128)
        #print(f"Input to CNN: {x.shape}")
        x = x.reshape(b, h * w, c)  # Shape: (batch_size, 372, 128)
        
        #print(f"Input to CNN: {x.shape}")
        x, _ = self.Lstm(x)  # Shape: (batch_size, 372, 512)
        b, seq_len, hidden_dim = x.shape
        
        x = x.reshape(b * seq_len, hidden_dim)
        
        # Fully connected: (b*372, num_classes + 1)
        x = self.fc(x)
        
        # Reshape back to 3D: (b, 372, num_classes + 1)
        x = x.reshape(b, seq_len, self.num_classes + 1)
        
        # Permute to (T, b, num_classes+1) so CTC sees (time, batch, classes)
        x = x.permute(1, 0, 2)
        
        
        x = torch.nn.functional.log_softmax(x, dim=2)
        return x
    
    def compute_levenshtein_distance(self, decoded_preds, decoded_targets):
        total_distance = 0
        total_words = len(decoded_targets)
        
        for pred, target in zip(decoded_preds, decoded_targets):
            pred_str = "".join(map(str, pred))  # Convert to string for comparison
            target_str = "".join(map(str, target))
            total_distance += Levenshtein.distance(pred_str, target_str)
        
        return total_distance / total_words if total_words > 0 else 0
    
    @staticmethod
    def ctc_decode(preds, symbols, blank_token=0):
        """
        Decodes CTC output by removing consecutive duplicate characters and blank tokens.
        
        Args:
            preds (torch.Tensor): The model's output logits of shape (T, batch_size, num_classes).
            symbols (str): The dictionary mapping indices to characters.
            blank_token (int): The index of the blank token in CTC.
    
        Returns:
            List[str]: A list of decoded strings.
        """
        preds = preds.argmax(2)  # Convert logits to predicted character indices
        decoded = []
        
        for seq in preds.permute(1, 0):  # Iterate over each batch sample
            new_seq = []
            prev_char = None
            for char_idx in seq:
                char = char_idx.item()
                if char != blank_token:  # Remove duplicates and blank tokens
                    new_seq.append(symbols[char])
                prev_char = char
            decoded.append("".join(new_seq))  # Convert list of characters to string
        
        return decoded
    
    
    def get_index(self, string: str) -> List[int]:
        """Get the index of a character in the alphabet."""
        
        
        
        indexes = []
        for i in string:
            
            indexes.append(self.symbols_dict[i])
        
        return indexes
    
    class CustomDataset(Dataset):
        def __init__(self, image_tensors, label_tensors):
            self.image_tensors = image_tensors
            self.label_tensors = label_tensors  # Labels should be padded
    
        def __len__(self):
            return len(self.image_tensors)
        
        def __getitem__(self, idx):
            return self.image_tensors[idx], self.label_tensors[idx]
    
    
    def data_loader(self, path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loads images and their labels from a csv file."""
        images, labels = [], []
        data = FileHandeling.Lmdb.read_entire_lmdb(path)
        for key, value in tqdm(data.items()):
            images.append(torch.tensor(value[1]))
            labels.append(torch.tensor(self.get_index(value[0])))
        pad_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value = 0)
        return torch.stack(images).float(), pad_labels
    
    def ttraining(self):
        """Train the CNN model with the training dataset."""
        # Load training and validation data
        train_images, train_labels = self.data_loader(self.train_path)
        train_dataset = self.CustomDataset(train_images, train_labels)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        
        val_images, val_labels = self.data_loader(self.validation_path)
        val_dataset = self.CustomDataset(val_images, val_labels)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,  # It was incorrectly loading train_dataset earlier
            batch_size=self.batch_size,
            shuffle=True,
        )
        
        # Prepare accuracy log
        with open("accuracy_2.csv", "w") as acc:
            acc.write("epoch,val_accuracy,train_accuracy,loss\n")
        
        # Training loop
        for epoch in tqdm(range(self.num_epochs), desc="Training"):
            
            epoch_loss, train_accuracy = 0, 0
            #self.accuracy_metric.reset()
            
            # Validation phase
            self.eval()
            val_accuracy_total = 0
            with torch.no_grad():
                for x_val, y_val in tqdm(val_loader):
                    x_val = x_val.to(self.device)
                    y_val = y_val.to(self.device)
                    outputs = self.forward(x_val)  # (T, N, C)
                    
                    # 1. Define input_lengths (assuming all sequences have the same length T)
                    T = outputs.size(0)  # Time steps
                    N = outputs.size(1)  # Batch size
                    input_lengths = torch.full(
                        size=(N,), fill_value=T, dtype=torch.long, device=self.device
                    )
                    
                    # 2. Define target_lengths
                    target_lengths = (y_val != self.num_classes).sum(dim=1).to(torch.long)
                    
                    # 3. Compute CTC loss
                    loss = self.criterion(outputs, y_val, input_lengths, target_lengths)
                    
                    # Optionally, compute accuracy
                    decoded_preds = self.ctc_decode(outputs, self.symbols, blank_token=self.num_classes)
                    decoded_targets = []
                    for label in y_val:
                        chars = [self.symbols[i.item()] for i in label if i.item() != self.num_classes]
                        decoded_targets.append(chars)
                    val_accuracy_total += self.compute_levenshtein_distance(decoded_preds, decoded_targets)
            
            val_accuracy = val_accuracy_total / len(val_loader)
            total_loss = 0
            # Training phase
            self.train()
            train_accuracy_total = 0
            for x_batch, y_batch in tqdm(train_loader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward pass, loss, backward pass
                outputs = self.forward(x_batch)  # (T, N, C)
                
                # 1. Define input_lengths
                T = outputs.size(0)
                N = outputs.size(1)
                input_lengths = torch.full(
                    size=(N,), fill_value=T, dtype=torch.long, device=self.device
                )
                
                # 2. Define target_lengths
                target_lengths = (y_batch != self.num_classes).sum(dim=1).to(torch.long)
                
                # 3. Compute CTC loss
                loss = self.criterion(outputs, y_batch, input_lengths, target_lengths)
                epoch_loss += loss.item()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
                # Optionally, compute accuracy
                decoded_preds = self.ctc_decode(outputs, self.symbols, blank_token=self.num_classes)
                decoded_targets = []
                for label in y_batch:
                    chars = [self.symbols[i.item()] for i in label if i.item() != self.num_classes]
                    decoded_targets.append(chars)
                train_accuracy_total += self.compute_levenshtein_distance(decoded_preds, decoded_targets)
            
            avg_loss = total_loss / len(train_loader)
            train_accuracy = train_accuracy_total / len(train_loader)
            
            # Write epoch metrics
            with open("accuracy_2.csv", "a") as acc:
                acc.write(f"{epoch},{val_accuracy},{train_accuracy},{avg_loss}\n")
            
            print(
                f"Epoch {epoch + 1}/{self.num_epochs}, "
                f"Train Accuracy: {train_accuracy:.4f}, "
                f"Val Accuracy: {val_accuracy:.4f}, "
                f"Loss: {avg_loss:.4f}"
            )
            
            torch.save(self.state_dict(), os.path.join(option.PathToLetterNet, f"netCRNN{epoch}.pt"))
            
            # Reset metrics after each epoch
            self.accuracy_metric.reset()
            self.train_accuracy_metric.reset()
            
            df = pd.read_csv("accuracy_2.csv")
            df.plot(x="epoch", y="loss")
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
    
    #def eval(self, input_data: torch.Tensor) -> int:
        """
        Evaluate the input tensor using the trained model.

        Args:
            input_data (torch.Tensor): A single input tensor (e.g., an image) to evaluate,
        #                               with shape (Height, Width) or (Channels, Height, Width).
#
        #Returns:
        #    int: Predicted label (class) for the input.
        #"""
        #
        ## Ensure the model is in evaluation mode
        #self.eval(input_data)
        #
        ## If input_data lacks a batch dimension, add it
        #if input_data.ndim == 2:  # Expects a single-channel image
        #    input_data = input_data.unsqueeze(0).unsqueeze(0)  # (Height, Width) -> (1, 1, Height, Width)
        #elif input_data.ndim == 3:  # If shape is (Channels, Height, Width)
        #    input_data = input_data.unsqueeze(0)  # (Channels, Height, Width) -> (1, Channels, Height, Width)
        #
        ## Transfer input to the appropriate device
        #input_data = input_data.to(self.device)
        #
        ## Forward pass through the model
        #with torch.no_grad():
        #    outputs = self.net(input_data)
        #
        ## Get the predicted class (highest probability)
        #predicted_label = torch.argmax(outputs, dim = 1).item()
        #
        #return predicted_label

if __name__ == "__main__":
    # Initialize the model
    model = CRNN()
    
    # Train the model
    model.ttraining()
    
   