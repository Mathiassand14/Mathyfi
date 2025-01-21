import os
from typing import List, Tuple

import Levenshtein
import pandas as pd
import torch
import torchmetrics
from matplotlib import pyplot as plt
from sympy import factor
from torch.utils.data import Dataset
from tqdm import tqdm

import FileHandeling.Lmdb

from Options.OptionsUser import OptionsUser as option

class LmdbDataset(Dataset):
    def __init__(self, path: str, num_classes: int):
        """Initialize the dataset and load LMDB keys."""
        self.symbols = r" !#%&()*+-.,/0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_abcdefghijklmnopqrstuvwxyz{|}<>\""
        
        self.symbols_dict = {}
        for i, symbol in enumerate(self.symbols):
            self.symbols_dict[symbol] = i
        
        self.data = FileHandeling.Lmdb.read_entire_lmdb(path)
        self.keys = list(self.data.keys())
        self.num_classes = num_classes
    
    def __len__(self):
        """Return the total number of items in the dataset."""
        return len(self.keys)
    
    def get_index(self, string: str) -> List[int]:
        """Get the index of a character in the alphabet."""
        
        
        
        indexes = []
        for i in string:
            
            indexes.append(self.symbols_dict[i])
        
        return indexes
    
    def __getitem__(self, index):
        """Get an item at the specified index."""
        key = self.keys[index]
        value = self.data[key]
        latex = value[0].replace(r'<annotation type="normalizedLabel">', "")
        image = torch.tensor(value[1]).float()
        label = torch.tensor(self.get_index(latex))
        return image, label




class CRNN(torch.nn.Module):
    def __init__\
        (
            self,
            batch_size: int = 130,
            num_epochs: int = 1000,
            learning_rate: float = 0.0001,
            gpu: bool = None
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
        self.train_path = os.path.join(option.PathToLatexLmdb, "trainx2.lmdb") #"validation.lmdb")#
        self.test_path = os.path.join(option.PathToLatexLmdb, "testx2.lmdb")
        self.validation_path = os.path.join(option.PathToLatexLmdb, "validationx2.lmdb") # "test.lmdb")#
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if gpu is None else "cuda" if gpu else "cpu"
        
        
        # Metrics setup
        self.accuracy_metric = torchmetrics.classification.Accuracy(
            task = "multiclass", num_classes = self.num_classes
        ).to(self.device)
        self.train_accuracy_metric = torchmetrics.classification.Accuracy(
            task = "multiclass", num_classes = self.num_classes
        ).to(self.device)
        
        
        
        
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, padding=1), # 500*50X1
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            # Potentially add a light dropout here if you want, e.g.:
            torch.nn.Dropout(p=0.1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            
            
            # --- Block 1 ---
            torch.nn.Conv2d(16, 32, 3, padding=1), # 500*50X1
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
            torch.nn.Dropout(p=0.1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            
            # --- Block 3 ---
            torch.nn.Conv2d(64, 128, 3, padding=1), # 125*12X64
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            # Light dropout for third block
            torch.nn.Dropout(p=0.2),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
    
        self.Lstm = torch.nn.LSTM(128, 384, 2, batch_first=True, bidirectional=True, dropout = 0.3) # 64*6X128
    
        self.fc = torch.nn.Sequential(
            
            torch.nn.Linear(768, self.num_classes + 1),
            #torch.nn.BatchNorm1d(512),
            #torch.nn.ReLU(),
            ## Heavier dropout on fully-connected layer
            #torch.nn.Dropout(p=0.4),
            #
            #torch.nn.Linear(512, self.num_classes + 1),
        )
        
        
        
        # Model, Loss, and Optimizer
        self.criterion = torch.nn.CTCLoss(blank = self.num_classes, zero_infinity = True, reduction = "mean")
        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate, weight_decay = 1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode = "min", factor = 0.1, patience = 4)
        
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
            # Map indices to characters and exclude blank tokens
            pred_str = "".join([self.symbols[idx] for idx in pred if idx != self.num_classes])
            target_str = "".join([self.symbols[idx] for idx in target if idx != self.num_classes])
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
            prev_char = blank_token  # Initialize with blank to handle the first character correctly
            for char_idx in seq:
                char = char_idx.item()
                if char != prev_char:
                    if char != blank_token:
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
    
    

    
    def collate_fn(self, batch):
        """Custom collate function to handle padding of labels."""
        images, labels = zip(*batch)
        pad_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first = True, padding_value = self.num_classes)
        return torch.stack(images), pad_labels
    
    def data_loader(self, path: str):
        """Prepare the data loader for the given path."""
        dataset = LmdbDataset(path, self.num_classes)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size = self.batch_size,
            shuffle = True,
            collate_fn = self.collate_fn,  # Custom collate_fn for batching padded labels
        )
        return data_loader
    
    def compute_cer(self, decoded_preds: List[str], decoded_targets: List[str]) -> float:
        """
        Compute Character Error Rate (CER).
        
        Args:
            decoded_preds (List[str]): List of predicted sequences.
            decoded_targets (List[str]): List of ground truth sequences.
    
        Returns:
            float: CER value
        """
        total_chars = 0
        total_edits = 0
        
        for pred, target in zip(decoded_preds, decoded_targets):
            total_chars += len(target)
            
            # Compute edit distance using Levenshtein library
            total_edits += Levenshtein.distance(pred, target)
        
        return total_edits / total_chars if total_chars > 0 else 0
        
    
    
    
    
    def ttraining(self,epok = 0):
        """Train the CNN model with the training dataset."""
        # Load training and validation data
        train_loader = self.data_loader(self.train_path)
        val_loader = self.data_loader(self.validation_path)
        
        # Training loop
        for epoch in tqdm(range(self.num_epochs), desc="Training"):
            
            total_loss = 0
            total_train_cer = 0
            total_val_cer = 0
            
            # Training phase
            self.train()
            for x_batch, y_batch in tqdm(train_loader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward pass
                outputs = self.forward(x_batch)  # (T, N, C)
                
                # Define input_lengths
                T, N = outputs.size(0), outputs.size(1)
                input_lengths = torch.full((N,), fill_value=T, dtype=torch.long, device=self.device)
                
                # Define target_lengths
                target_lengths = (y_batch != self.num_classes).sum(dim=1).to(torch.long)
                
                # Compute CTC loss
                loss = self.criterion(outputs, y_batch, input_lengths, target_lengths)
                
                total_loss += loss.item()
                
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
                self.optimizer.step()
                
                # Decode predictions
                decoded_preds = self.ctc_decode(outputs, self.symbols, blank_token=self.num_classes)
                decoded_targets = []
                for label in y_batch:
                    chars = [self.symbols[i.item()] for i in label if i.item() != self.num_classes]
                    decoded_targets.append("".join(chars))
                cer = self.compute_cer(decoded_preds, decoded_targets)
                print(f"\nloss: {[loss.item()]}\nacuracy: {cer}\nprediction: {decoded_preds[0]}\ntarget: "
                      f"{decoded_targets[0]}")
                # Compute CER
                total_train_cer += cer
            
            avg_train_loss = total_loss / len(train_loader)
            avg_train_cer = total_train_cer / len(train_loader)
            self.scheduler.step(avg_train_loss)
            
            # Validation phase
            total_val_loss = 0
            self.eval()
            with torch.no_grad():
                for x_val, y_val in tqdm(val_loader):
                    x_val = x_val.to(self.device)
                    y_val = y_val.to(self.device)
                    
                    outputs = self.forward(x_val)  # (T, N, C)
                    
                    # Define input_lengths
                    T, N = outputs.size(0), outputs.size(1)
                    input_lengths = torch.full((N,), fill_value=T, dtype=torch.long, device=self.device)
                    
                    # Define target_lengths
                    target_lengths = (y_val != self.num_classes).sum(dim=1).to(torch.long)
                    
                    # Compute CTC loss
                    loss = self.criterion(outputs, y_val, input_lengths, target_lengths)
                    
                    # Decode predictions
                    decoded_preds = self.ctc_decode(outputs, self.symbols, blank_token=self.num_classes)
                    decoded_targets = []
                    for label in y_val:
                        chars = [self.symbols[i.item()] for i in label if i.item() != self.num_classes]
                        decoded_targets.append("".join(chars))
                    
                    # Compute CER
                    total_val_cer += self.compute_cer(decoded_preds, decoded_targets)
                    total_val_loss += loss
            
            val_loss = total_val_loss / len(val_loader)
            
            
            
            avg_val_cer = total_val_cer / len(val_loader)
            
            # Log metrics
            with open("accuracy_x2.csv", "a") as acc:
                acc.write(f"{epoch + epok},{avg_val_cer},{avg_train_cer},{avg_train_loss},{val_loss}\n")
            
            print(
                f"Epoch {epoch + 1 + epok}/{self.num_epochs + epok}, "
                f"Train CER: {avg_train_cer:.4f}, "
                f"Val CER: {avg_val_cer:.4f}, "
                f"Loss: {avg_train_loss:.4f}"
            )
            
            torch.save(self.state_dict(), os.path.join(option.PathToLetterNet, f"netCRNNX2_{epoch + epok}.pt"))
            
            # Reset metrics after each  epoch
            self.accuracy_metric.reset()
            self.train_accuracy_metric.reset()
            
            # Plot loss
            df = pd.read_csv("accuracy_x2.csv")
            df.plot(x="epoch", y=["loss","val_loss","val_accuracy","train_accuracy"])
            plt.show()
    
    def test(self):
        """Test the CNN model on the test dataset."""
        val_loader = self.data_loader(self.test_path)
        
        
        
        total_loss, total_cer, total_corect = 0, 0, 0
        self.eval()
        with torch.no_grad():
            for x_val, y_val in tqdm(val_loader):
                x_val = x_val.to(self.device)
                y_val = y_val.to(self.device)
                
                outputs = self.forward(x_val)  # (T, N, C)
                
                # Define input_lengths
                T, N = outputs.size(0), outputs.size(1)
                input_lengths = torch.full((N,), fill_value=T, dtype=torch.long, device=self.device)
                
                # Define target_lengths
                target_lengths = (y_val != self.num_classes).sum(dim=1).to(torch.long)
                
                # Compute CTC loss
                loss = self.criterion(outputs, y_val, input_lengths, target_lengths)
                
                # Decode predictions
                decoded_preds = self.ctc_decode(outputs, self.symbols, blank_token=self.num_classes)
                decoded_targets = []
                for label in y_val:
                    chars = [self.symbols[i.item()] for i in label if i.item() != self.num_classes]
                    decoded_targets.append("".join(chars))
                
                total_corect += sum([1 for i, j in zip(decoded_preds, decoded_targets) if i == j])
                
                # Compute CER
                
                total_cer += self.compute_cer(decoded_preds, decoded_targets)
                total_loss += loss
        
        val_loss = total_loss / len(val_loader)
        
        
        
        avg_val_cer = total_cer / len(val_loader)
        
        corect_percent = total_corect / len(val_loader)
        
        print(f"Test CER: {avg_val_cer:.4f}, Loss: {val_loss:.4f}, Correct: {corect_percent:.4f}")
    
    
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
    model.load_state_dict(torch.load(os.path.join(option.PathToLetterNet, f"netCRNNX2_51.pt")))
    total_params = sum(p.numel() for p in model.parameters())

    # Total number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    
    model.test()
    #model.device = "cpu"
    # Train the model
    #model.ttraining(19)
    
    