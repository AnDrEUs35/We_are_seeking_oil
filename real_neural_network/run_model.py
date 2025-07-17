import torch
import segmentation_models_pytorch as smp
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import Dataset
from model import Model
from main import test_model  
import os



main_dir = Path(__file__).parent
x_test_dir = os.path.join(main_dir, "im_test")
y_test_dir = os.path.join(main_dir, "mask_test")
output_dir = os.path.join(main_dir, "output_images")
os.makedirs(output_dir, exist_ok=True)

input_image_reshape = (128, 128)  # или (320, 320) если нужно выше качество
foreground_class = 255
batch_size = 8


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


model = Model("Unet", "resnet34", in_channels=3, out_classes=1)
model.load_state_dict(torch.load(main_dir / "model.bin", map_location=device))
model.to(device)


test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    input_image_reshape=input_image_reshape,
    foreground_class=foreground_class,
)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Функция потерь (должна быть та же, что при обучении)

loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

test_loss, iou = test_model(model, output_dir, test_dataloader, loss_fn, device)

print(f"Test Loss: {test_loss:.4f}, IoU Score: {iou:.4f}")
print(f"Predicted masks saved in: {output_dir}")

