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


device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")
if device == "cpu":
    os.system("export OMP_NUM_THREADS=64")
    torch.set_num_threads(os.cpu_count())

# Change this to your desired directory
main_dir = Path(__file__).parent
val_dir = os.path.join(main_dir, "im_val")
mask_val_dir = os.path.join(main_dir, "mask_val")
mask_train_dir = os.path.join(main_dir, "mask_train")
train_dir = os.path.join(main_dir, "im_train")
mask_test_dir = os.path.join(main_dir, "mask_test")
test_dir = os.path.join(main_dir, "im_test")



# Create a directory to store the output masks
output_dir = os.path.join(main_dir, "output_images")
os.makedirs(output_dir, exist_ok=True)

# ----------------------------
# Define the hyperparameters
# ----------------------------
epochs_max = 70  # Number of epochs to train the model
adam_lr = 2e-4  # Learning rate for the Adam optimizer
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


# Use multiple CPUs in parallel
def train_and_evaluate_one_epoch(
    model, train_dataloader, valid_dataloader, optimizer, scheduler, loss_fn, device
):
    # Set the model to training mode
    model.train()
    train_loss = 0
    for batch in tqdm(train_dataloader, desc="Training"):
        images, masks = batch
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()
    avg_train_loss = train_loss / len(train_dataloader)

    # Set the model to evaluation mode
    model.eval()
    val_loss = 0
    with torch.inference_mode():
        for batch in tqdm(valid_dataloader, desc="Evaluating"):
            images, masks = batch
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(valid_dataloader)
    return avg_train_loss, avg_val_loss


def train_model(
    model,
    train_dataloader,
    valid_dataloader,
    optimizer,
    scheduler,
    loss_fn,
    device,
    epochs,
):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        avg_train_loss, avg_val_loss = train_and_evaluate_one_epoch(
            model,
            train_dataloader,
            valid_dataloader,
            optimizer,
            scheduler,
            loss_fn,
            device,
        )
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        logging.info(
            f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.2f}, Validation Loss: {avg_val_loss:.2f}"
        )

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    return history


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
    logging.info(f"Test Loss: {test_loss_mean:.4f}")

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


# ----------------------------
# Define the data directories and create the datasets
# ----------------------------
x_train_dir = train_dir
y_train_dir = mask_train_dir

x_val_dir = val_dir
y_val_dir = mask_val_dir

x_test_dir = test_dir
y_test_dir = mask_test_dir

train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    input_image_reshape=input_image_reshape,
    foreground_class=foreground_class,
)
valid_dataset = Dataset(
    x_val_dir,
    y_val_dir,
    input_image_reshape=input_image_reshape,
    foreground_class=foreground_class,
)
test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    input_image_reshape=input_image_reshape,
    foreground_class=foreground_class,
)

image, mask = train_dataset[0]
logging.info(f"Unique values in mask: {np.unique(mask)}")
logging.info(f"Image shape: {image.shape}")
logging.info(f"Mask shape: {mask.shape}")

# ----------------------------
# Create the dataloaders using the datasets
# ----------------------------
logging.info(f"Train size: {len(train_dataset)}")
logging.info(f"Valid size: {len(valid_dataset)}")
logging.info(f"Test size: {len(test_dataset)}")

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ----------------------------
# Lets look at some samples
# ----------------------------
# Visualize and save train sample
"""
sample = train_dataset[0]
visualize(
    output_dir,
    "train_sample.png",
    train_image=sample[0].numpy().transpose(1, 2, 0),
    train_mask=sample[1].squeeze(),
)

# Visualize and save validation sample
sample = valid_dataset[0]
visualize(
    output_dir,
    "validation_sample.png",
    validation_image=sample[0].numpy().transpose(1, 2, 0),
    validation_mask=sample[1].squeeze(),
)

# Visualize and save test sample
sample = test_dataset[0]
visualize(
    output_dir,
    "test_sample.png",
    test_image=sample[0].numpy().transpose(1, 2, 0),
    test_mask=sample[1].squeeze(),
)
"""
# ----------------------------
# Create and train the model
# ----------------------------
max_iter = epochs_max * len(train_dataloader)  # Total number of iterations

model = Model("Unet", "resnet34", in_channels=3, out_classes=1)

# Training loop
model = model.to(device)

# Define the Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)

# Define the learning rate scheduler
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=eta_min)

# Define the loss function
loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
torch.backends.cudnn.benchmark = True

# Train the model
history = train_model(
    model,
    train_dataloader,
    valid_dataloader,
    optimizer,
    scheduler,
    loss_fn,
    device,
    epochs_max,
)

# Visualize the training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(history["train_losses"], label="Train Loss")
plt.plot(history["val_losses"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Losses")
plt.legend()
plt.savefig(os.path.join(output_dir, "train_val_losses.png"))
plt.close()


# Evaluate the model
test_loss = test_model(model, output_dir, test_dataloader, loss_fn, device)

logging.info(f"Test Loss: {test_loss[0]}, IoU Score: {test_loss[3]}, Accuracy: {test_loss[1]}, F1 score: {test_loss[2]}")
logging.info(f"The output masks are saved in {output_dir}.")

torch.save(model.state_dict(), Path(__file__).parent / "model.bin")


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

main_dir = os.getcwd()
val_dir = os.path.join(main_dir, "Oil")
mask_dir = os.path.join(main_dir, "Mask_oil")
test_dir = os.path.join(main_dir, "test_dir")

# Create a directory to store the output masks
output_dir = os.path.join(main_dir, "output_images")
os.makedirs(output_dir, exist_ok=True)

# ----------------------------
# Define the hyperparameters
# ----------------------------
epochs_max = 200  # Number of epochs to train the model
adam_lr = 2e-4  # Learning rate for the Adam optimizer
eta_min = 1e-5  # Minimum learning rate for the scheduler
batch_size = 8  # Batch size for training
input_image_reshape = (2048, 2048)  # Desired shape for the input images and masks
foreground_class = 1  # 1 for binary segmentation



def visualize(output_dir, image_filename, **images):
    PLot images in one row.
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)
    plt.show()
    plt.savefig(os.path.join(output_dir, image_filename))
    plt.close()


# Use multiple CPUs in parallel
def train_and_evaluate_one_epoch(
    model, train_dataloader, valid_dataloader, optimizer, scheduler, loss_fn, device
):
    # Set the model to training mode
    model.train()
    train_loss = 0
    for batch in tqdm(train_dataloader, desc="Training"):
        images, masks = batch
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()
    avg_train_loss = train_loss / len(train_dataloader)

    # Set the model to evaluation mode
    model.eval()
    val_loss = 0
    with torch.inference_mode():
        for batch in tqdm(valid_dataloader, desc="Evaluating"):
            images, masks = batch
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(valid_dataloader)
    return avg_train_loss, avg_val_loss


def train_model(
    model,
    train_dataloader,
    valid_dataloader,
    optimizer,
    scheduler,
    loss_fn,
    device,
    epochs,
):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        avg_train_loss, avg_val_loss = train_and_evaluate_one_epoch(
            model,
            train_dataloader,
            valid_dataloader,
            optimizer,
            scheduler,
            loss_fn,
            device,
        )
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        logging.info(
            f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.2f}, Validation Loss: {avg_val_loss:.2f}"
        )

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    return history


def test_model(model, output_dir, test_dataloader, loss_fn, device):
    # Set the model to evaluation mode
    model.eval()
    test_loss = 0
    tp, fp, fn, tn = 0, 0, 0, 0
    with torch.inference_mode():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            images, masks = batch
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)

            for i, output in enumerate(outputs):
                input = images[i].cpu().numpy().transpose(1, 2, 0)
                output = output.squeeze().cpu().numpy()

                visualize(
                    output_dir,
                    f"output_{i}.tif",
                    input_image=input,
                    output_mask=output,
                    binary_mask=output > 0.5,
                )

            test_loss += loss.item()

            prob_mask = outputs.sigmoid().squeeze(1)
            pred_mask = (prob_mask > 0.5).long()
            batch_tp, batch_fp, batch_fn, batch_tn = smp.metrics.get_stats(
                pred_mask, masks, mode="binary"
            )
            tp += batch_tp.sum().item()
            fp += batch_fp.sum().item()
            fn += batch_fn.sum().item()
            tn += batch_tn.sum().item()

        test_loss_mean = test_loss / len(test_dataloader)
        logging.info(f"Test Loss: {test_loss_mean:.2f}")

    iou_score = smp.metrics.iou_score(
        torch.tensor([tp]),
        torch.tensor([fp]),
        torch.tensor([fn]),
        torch.tensor([tn]),
        reduction="micro",
    )

    return test_loss_mean, iou_score.item()
# ----------------------------
# Define the data directories and create the datasets
# ----------------------------
x_train_dir = val_dir
y_train_dir = mask_dir

x_val_dir = val_dir
y_val_dir = mask_dir

x_test_dir = test_dir
y_test_dir = test_dir

train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    input_image_reshape=input_image_reshape,
    foreground_class=foreground_class,
)
valid_dataset = Dataset(
    x_val_dir,
    y_val_dir,
    input_image_reshape=input_image_reshape,
    foreground_class=foreground_class,
)
test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    input_image_reshape=input_image_reshape,
    foreground_class=foreground_class,
)

image, mask = train_dataset[0]
logging.info(f"Unique values in mask: {np.unique(mask)}")
logging.info(f"Image shape: {image.shape}")
logging.info(f"Mask shape: {mask.shape}")

# ----------------------------
# Create the dataloaders using the datasets
# ----------------------------
logging.info(f"Train size: {len(train_dataset)}")
logging.info(f"Valid size: {len(valid_dataset)}")
logging.info(f"Test size: {len(test_dataset)}")

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


#создание модели

max_iter = epochs_max * len(train_dataloader)  # Total number of iterations

model = Model("Unet", "resnet34", device = device, in_channels=3, out_classes=1)

# Training loop
model = model.to(device)

# Define the Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)

# Define the learning rate scheduler
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=eta_min)

# Define the loss function
loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

# Train the model
history = train_model(
    model,
    train_dataloader,
    valid_dataloader,
    optimizer,
    scheduler,
    loss_fn,
    device,
    epochs_max,
)

# Visualize the training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(history["train_losses"], label="Train Loss")
plt.plot(history["val_losses"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Losses")
plt.legend()
plt.savefig(os.path.join(output_dir, "train_val_losses.png"))
plt.close()


# Evaluate the model
test_loss = test_model(model, output_dir, test_dataloader, loss_fn, device)

logging.info(f"Test Loss: {test_loss[0]}, IoU Score: {test_loss[1]}")
logging.info(f"The output masks are saved in {output_dir}.")

"""