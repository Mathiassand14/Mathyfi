import torch

model = torch.nn.Sequential(
    torch.nn.Conv2d(1,32,3, padding=1),
    torch.nn.BatchNorm2d(32),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
    torch.nn.Conv2d(32,64,3, padding=1),
    torch.nn.BatchNorm2d(64),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
    torch.nn.Conv2d(64,128,3, padding=1),
    torch.nn.BatchNorm2d(128),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
    torch.nn.Flatten(),
    torch.nn.Linear(128*5*5, 256),
    torch.nn.BatchNorm1d(256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(256,83)
    
)