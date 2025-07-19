import segmentation_models_pytorch as smp
from tqdm import tqdm
import torch
import matplotlib as plt
import logging
import os


#logging.basicConfig(
 #   level=logging.INFO,
  #  format="%(asctime)s - %(message)s",
   # datefmt="%d:%m:%Y %H:%M:%S",
#)

# ----------------------------
# Set the device to GPU if available
# ----------------------------
device = "cpu" #if torch.cuda.is_available() else "cpu" #нужно комментировать
#logging.info(f"Using device: {device}")
#if device == "cpu":
   # os.system("export OMP_NUM_THREADS=64")
    #torch.set_num_threads(os.cpu_count())

class Model(torch.nn.Module):
    def __init__(self, arch, encoder_name, in_channels=3, out_classes=1, **kwargs):
        super().__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )

    def forward(self, image):
        # Normalize image
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask





"""
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%d:%m:%Y %H:%M:%S",
)

# ----------------------------
# Set the device to GPU if available
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")
if device == "cpu":
    os.system("export OMP_NUM_THREADS=64")
    torch.set_num_threads(os.cpu_count())

class Model(torch.nn.Module):
    def __init__(self, arch, encoder_name, in_channels=3, out_classes=1, **kwargs):
        super().__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )

    def forward(self, image):
        # Normalize image
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask.unsqueeze(1)




logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%d:%m:%Y %H:%M:%S",
)

# ----------------------------
# Set the device to GPU if available
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")
if device == "cpu":
    os.system("export OMP_NUM_THREADS=64")
    torch.set_num_threads(os.cpu_count())

class Model(torch.nn.Module):
    def __init__(self, arch, encoder_name, device, in_channels=3, out_classes=1, **kwargs):
        super().__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device) #как понять какой разер тензора?
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )

    def forward(self, image):
        # Normalize image
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask
"""
