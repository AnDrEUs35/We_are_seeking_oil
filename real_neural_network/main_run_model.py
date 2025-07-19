import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from tqdm import tqdm
from pathlib import Path
from dataset import Dataset
from model import Model

import segmentation_models_pytorch as smp



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%d:%m:%Y %H:%M:%S",
)

# ----------------------------
# Set the device to GPU if available
# ----------------------------


device = "cpu"

main_dir = Path(__file__).parent
mask_test_dir = os.path.join(main_dir, "mask_test")
test_dir = os.path.join(main_dir, "im_test")


# Create a directory to store the output masks
output_dir = os.path.join(main_dir, "output_images")
os.makedirs(output_dir, exist_ok=True)

# ----------------------------
# Define the hyperparameters
# ----------------------------


eta_min = 1e-5  # Minimum learning rate for the scheduler
batch_size = 8  # Batch size for training
input_image_reshape = (128, 128)  # Desired shape for the input images and masks
foreground_class = 255  # 1 for binary segmentation


def visualize(output_dir, image_filename, **images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)
    plt.savefig(os.path.join(output_dir, image_filename))
    plt.show()
    plt.close()


def test_model(model, output_dir, test_dataloader, loss_fn, device):
    model.eval()
    test_loss = 0.0
    tp, fp, fn, tn = 0, 0, 0, 0

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            images, masks = batch
            images, masks = images.to(device), masks.to(device)

            # For BCELoss, apply sigmoid manually before loss
            outputs = model(images)
            prob_outputs = torch.sigmoid(outputs)
            loss = loss_fn(prob_outputs, masks.float())

            for i, output in enumerate(prob_outputs):
                input_img = images[i].cpu().numpy().transpose(1, 2, 0)
                output_img = output.squeeze().cpu().numpy()
                
                visualize(
                    output_dir,
                    f"output_{i}.png",
                    input_image=input_img,
                    output_mask=output_img,
                    binary_mask=output_img > 0.5,
                )

            test_loss += loss.item()

            pred_mask = (prob_outputs.squeeze(1) > 0.5).long()

            batch_tp, batch_fp, batch_fn, batch_tn = smp.metrics.get_stats(
                pred_mask, masks.long(), mode="binary"
            )

            tp += batch_tp.sum().item()
            fp += batch_fp.sum().item()
            fn += batch_fn.sum().item()
            tn += batch_tn.sum().item()

    test_loss_mean = test_loss / len(test_dataloader)

    iou_score = smp.metrics.iou_score(
        torch.tensor([tp]),
        torch.tensor([fp]),
        torch.tensor([fn]),
        torch.tensor([tn]),
        reduction="micro",
    )
    accuracy = smp.metrics.accuracy(
        torch.tensor([tp]),
        torch.tensor([fp]),
        torch.tensor([fn]),
        torch.tensor([tn]),
        reduction="macro",
    )

    f1_score = smp.metrics.f1_score(
        torch.tensor([tp]),
        torch.tensor([fp]),
        torch.tensor([fn]),
        torch.tensor([tn]),
        reduction="micro",
    )

    return test_loss_mean, accuracy.item(), f1_score.item(), iou_score.item()

x_test_dir = test_dir
y_test_dir = mask_test_dir

test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    input_image_reshape=input_image_reshape,
    foreground_class=foreground_class,
)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
model = Model("Unet", "resnet34", in_channels=3, out_classes=1)

# Define the loss function
loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
torch.backends.cudnn.benchmark = True
# # Evaluate the model
model = Model("Unet", "resnet34", in_channels=3, out_classes=1)
model.load_state_dict(torch.load("model1.bin"))
test_loss = test_model(model, output_dir, test_dataloader, loss_fn, device)
logging.info(f"Test Loss: {test_loss[0]:.4f}, IoU Score: {test_loss[3]:.4f}, Accuracy: {test_loss[1]:.4f}, F1 score: {test_loss[2]:.4f}")
logging.info(f"The output masks are saved in {output_dir}.")
