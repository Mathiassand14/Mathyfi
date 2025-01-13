import os.path
from typing import List

import pandas as pd
import torch
import torchmetrics
from matplotlib import pyplot as plt
from torch.nn.functional import dropout
from tqdm import tqdm

from Options.OptionsUser import OptionsUser as option
from FileHandeling.Data_HMSD import defalt_dict


if __name__ == "__main__":
    

    # region Settings
    train_path: str = option.PathToLetterTrain
    test_path: str = option.PathToLetterTest
    validation_path: str = option.PathToLetterValidation
    batch_size: int = 1024
    num_epochs: int = 1000
    learning_rate: float = 0.001
    dropout_rate: float = 0.5
    num_classes: int = 101
    
    # endregion
    
    # region Load data
    def data_loader(path: str):
        images = []
        labels = []
        class_labels = defalt_dict()
        
        cls_path: str = os.path.join(os.getcwd(),"FileHandeling","cls_labels.csv")
        if os.path.exists(cls_path):
            with open(cls_path, "r") as class_label:
                lines = class_label.read().split("\n")[1:]
                for line in lines:
                    if line == "":
                        continue
                    if line[0] == ",":
                        label = ","
                        int_ = line[2:]
                    else:
                        label, int_ = line.split(",")
                    class_labels[label] = int(int_)
        
        with open(path, "r") as imag:
            text: str = imag.read()
        
        lines: List[str] = text.split("\n")[1:]
        
        
        for line in tqdm(lines, desc="Loading data"):
            if line == "":
                continue
            if line[0] != ",":
                
                label, shape0, shape1, *array = line.split(",")
            else:
                label = ","
                shape0, shape1, *array = line[2:].split(",")
            #print(float(",".join(array).strip("[]").replace(" ","").split(",")))
            shape0 = "".join([i for i in shape0 if i.isdigit()])
            shape1 = "".join([i for i in shape1 if i.isdigit()])
            shape0 = shape0 if shape0 != "" else "45"
            shape1 = shape1 if shape1 != "" else "45"
            array = ["".join([j for j in i  if j.isdigit() or j == "."]) for i in array]
            image = \
                torch.tensor(
                    [
                        float(i)
                        for i in array
                    ]
                )
            if len(image) != 2025:
                
                print(image.shape)
                print(label)
                print(shape0, shape1)
                print(str(image.numpy()))
                print(array)
                
            image = \
                image.reshape(
                    int(shape0),
                    int(shape1)
                )
            
            images.append(image)
            
            labels.append(class_labels[label])
            
        print("Data loaded")
        
        with open(cls_path, "w") as class_label:
            class_label.write("label,int\n")
            for key in class_labels.dict:
                class_label.write(key + "," + str(class_labels.dict[key]) + "\n")
        
        return torch.stack(images).float(), torch.tensor(labels)
    # endregion
    
    # region load train data
    train_images_tensor, train_labels_tensor = data_loader(train_path)
    # endregion
    
    # region device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # endregion
    
    # region Dataloader
    train_data = torch.utils.data.TensorDataset(train_images_tensor, train_labels_tensor)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # endregion
    
    # region Model
    net = torch.nn.Sequential(
        torch.nn.Conv2d(1,32,3, padding=1),     # 45x45
        torch.nn.BatchNorm2d(32),   # 32x45x45
        torch.nn.ReLU(),    # 32x45x45
        torch.nn.MaxPool2d(kernel_size = 2, stride = 2),    # 32x22x22
        torch.nn.Conv2d(32,64,3, padding=1),    # 64x22x22X
        torch.nn.BatchNorm2d(64),   # 64x22x22
        torch.nn.ReLU(),    # 64x22x22
        torch.nn.MaxPool2d(kernel_size = 2, stride = 2),    # 64x11x11
        torch.nn.Conv2d(64,128,3, padding=1),   # 128x11x11
        torch.nn.BatchNorm2d(128),  # 128x11x11
        torch.nn.ReLU(),    # 128x11x11
        torch.nn.MaxPool2d(kernel_size = 2, stride = 2),    # 128x5x5
        torch.nn.Flatten(), # 128*5*5
        torch.nn.Linear(128*5*5, 256),  # 256
        torch.nn.BatchNorm1d(256),  # # 256
        torch.nn.ReLU(),    # 256
        torch.nn.Linear(256, num_classes),  # num_classes
        torch.nn.Dropout(dropout_rate)  #
    ).to(device)
    # endregion
    print(f"num parameters: {sum(p.numel() for p in net.parameters())}")
    # region Loss and optimizer
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    
    # endregion
    
    # region Metrics
    accuracy_metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    train_accuracy_metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    # endregion
    
    # region load validation data
    val_images_tensor, val_labels_tensor = data_loader(validation_path)
    
    val_data = torch.utils.data.TensorDataset(val_images_tensor, val_labels_tensor)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
    # endregion
    with open("accuracy.csv", "w") as acc:
        acc.write("epoch,accuracy,train_accuracy\n")
    
    # region Train
    step = 0
    for epoch in tqdm(range(num_epochs),position = 0, leave = True, desc = "Training"):
        accuracy_metric.reset()
        for x, y in train_loader:
            step += 1
            
            
            
            # Put data on GPU
            x = x.unsqueeze(1).to(device)
            y = y.to(device)
            
            # Compute loss and take gradient step
            net.train()
            out = net(x)
            loss = loss_function(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_accuracy_metric.update(out, y)
        
        for x, y in val_loader:
            x = x.unsqueeze(1).to(device)
            y = y.to(device)
            
            net.eval()
            out = net(x)
            loss = loss_function(out, y)
            
            accuracy_metric.update(out, y)
        
        # Update accuracy metric
        torch.save(net.state_dict(), os.path.join(option.PathToLetterNet,f"net{epoch}.pt"))
        print(f"Epoch: {epoch}, Accuracy: {accuracy_metric.compute(), train_accuracy_metric.compute()}")
        with open("accuracy.csv", "a") as acc:
            acc.write(f"{epoch},{accuracy_metric.compute()},{train_accuracy_metric.compute()}\n")
        accuracy_metric.reset()
        train_accuracy_metric.reset()
        df = pd.read_csv("accuracy.csv")
        df.plot(x = "epoch", y = ["accuracy","train_accuracy"])
        plt.show()
