import torch
from torchvision import models
from torch import nn

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.swin_v2_b(weights='DEFAULT')
    model.head = nn.Linear(in_features=model.head.in_features, out_features=7)  # Предполагается, что у вас семь классов
    model.load_state_dict(torch.load('checkpoints/best.pth'))
    model.to(device)
    model.eval()
    return device, model