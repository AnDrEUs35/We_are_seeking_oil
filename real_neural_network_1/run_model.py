from pathlib import Path
from model import Model
import torch
from main import test_model
import matplotlib as plt
from dataset import DataSet
from torch import DataL
import segmentation_models_pytorch as smp


model = Model("Unet", "resnet34", in_channels=3, out_classes=1)
model.load_state_dict(torch.load(Path(__file__) / "model.bin"))

batch_size = 8  # Batch size for training

test_dataset = Dataset(
        x_test_dir,
        y_test_dir,
        input_image_reshape=input_image_reshape,
        foreground_class=foreground_class,
)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
torch.backends.cudnn.benchmark = True

test_loss = test_model(model, output_dir, test_dataloader, loss_fn, device)


