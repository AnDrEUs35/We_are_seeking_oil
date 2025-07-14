import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split # Добавлен импорт для train_test_split

import segmentation_models_pytorch as smp
import rasterio # Импортируем rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS

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

# ----------------------------
# Конфигурация и пути к данным (адаптировано под вашу структуру)
# ----------------------------
# Вам нужно будет изменить эти пути в соответствии с вашей локальной структурой файлов.
DATA_DIR = 'C:\\Users\\Sirius\\Desktop\\neuronetwork\\GOTOVO' # Например: 'C:/Users/User/Desktop/my_dataset'
IMAGE_SUBDIR = 'Oil' # Поддиректория, где хранятся TIF изображения
MASK_SUBDIR = 'Mask_oil' # Поддиректория, где хранятся пиксельные маски

# Создаем корневые директории для изображений и масок
images_root_dir = os.path.join(DATA_DIR, IMAGE_SUBDIR)
masks_root_dir = os.path.join(DATA_DIR, MASK_SUBDIR)

# Создаем директорию для сохранения выходных масок
output_dir = os.path.join(DATA_DIR, "output_images") # Используем DATA_DIR для output_images
os.makedirs(output_dir, exist_ok=True)

# ----------------------------
# Define the hyperparameters
# ----------------------------
epochs_max = 200  # Number of epochs to train the model
adam_lr = 2e-4  # Learning rate for the Adam optimizer
eta_min = 1e-5  # Minimum learning rate for the scheduler
batch_size = 8  # Batch size for training
input_image_reshape = (256, 256)  # Desired shape for the input images and masks (из вашей конфигурации)
foreground_class = 1  # 1 for binary segmentation (из вашей конфигурации)

# Определяем количество каналов и классов из вашей конфигурации
NUM_CHANNELS = 2 # Изменено на 2 канала
NUM_CLASSES = 1  # Из вашей конфигурации

# ----------------------------
# Define a custom dataset class for the CamVid dataset
# ----------------------------
class Dataset(BaseDataset):
    """
    A custom dataset class for binary segmentation tasks.
    Адаптирован для приема списков путей к файлам.

    Parameters:
    ----------

    - image_filepaths (list): Список путей к входным изображениям.
    - mask_filepaths (list): Список путей к соответствующим маскам.
    - input_image_reshape (tuple, optional): Desired shape for the input
      images and masks. Default is (320, 320).
    - foreground_class (int, optional): The class value in the mask to be
      considered as the foreground. Default is 1.
    - augmentation (callable, optional): A function/transform to apply to the
      images and masks for data augmentation.
    """

    def __init__(
        self,
        image_filepaths, # Изменено на список путей
        mask_filepaths,  # Изменено на список путей
        input_image_reshape=(256, 256),
        foreground_class=1,
        augmentation=None,
    ):
        self.images_filepaths = image_filepaths
        self.masks_filepaths = mask_filepaths

        self.input_image_reshape = input_image_reshape
        self.foreground_class = foreground_class
        self.augmentation = augmentation

    def __getitem__(self, i):
        """
        Retrieves the image and corresponding mask at index `i`.

        Parameters:
        ----------

        - i (int): Index of the image and mask to retrieve.
        Returns:
        - A tuple containing:
            - image (torch.Tensor): The preprocessed image tensor of shape
            (1, input_image_reshape) - e.g., (1, 320, 320) - normalized to [0, 1].
            - mask_remap (torch.Tensor): The preprocessed mask tensor of
            shape input_image_reshape with values 0 or 1.
        """
        # Read the image using rasterio for robust TIFF loading
        with rasterio.open(self.images_filepaths[i]) as src:
            image = src.read()
        
        # rasterio reads in (C, H, W), convert to (H, W, C)
        if image.ndim == 3:
            image = np.transpose(image, (1, 2, 0))
        elif image.ndim == 2:
            image = np.expand_dims(image, axis=-1) # (H, W) -> (H, W, 1)

        # Ensure image has NUM_CHANNELS
        current_channels = image.shape[-1]

        if current_channels == NUM_CHANNELS:
            pass # Already correct number of channels
        elif NUM_CHANNELS == 1:
            if current_channels > 1:
                # Convert to grayscale if more than 1 channel, then ensure 1 channel
                if current_channels == 3: # If it's RGB/BGR, convert to grayscale
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                elif current_channels == 2: # If it's 2-channel, take the first or average
                    image = image[:, :, 0] # Take first channel
                image = np.expand_dims(image, axis=-1) # Ensure (H, W, 1)
            # If current_channels is already 1, nothing to do.
        elif NUM_CHANNELS == 2:
            if current_channels == 1:
                # Duplicate channel if only one is loaded
                image = np.concatenate([image, image], axis=-1)
            elif current_channels == 3:
                # Take first two channels if 3 are loaded (e.g., RGB -> R,G)
                image = image[:, :, :2]
            elif current_channels == 4:
                # Take first two channels if 4 are loaded (e.g., RGBA -> R,G)
                image = image[:, :, :2]
            else:
                raise ValueError(f"Несоответствие каналов в {self.images_filepaths[i]}: Загружено {current_channels}, ожидается {NUM_CHANNELS}. Неизвестное преобразование.")
        elif NUM_CHANNELS == 3:
            if current_channels == 1:
                # Convert grayscale to BGR (duplicate channel)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif current_channels == 2:
                # If 2 channels, duplicate one or add zero channel to make 3
                # For simplicity, duplicate the first channel to make it 3-channel-like
                image = np.concatenate([image, image[:,:,:1]], axis=-1) # Example: (C1, C2) -> (C1, C2, C1)
                logging.warning(f"Преобразование 2-канального изображения в 3-канальное для {self.images_filepaths[i]}: Дублирован канал. Проверьте, что это соответствует вашим данным.")
            elif current_channels == 4:
                # Take RGB from RGBA
                image = image[:, :, :3]
            else:
                raise ValueError(f"Несоответствие каналов в {self.images_filepaths[i]}: Загружено {current_channels}, ожидается {NUM_CHANNELS}. Неизвестное преобразование.")
        else:
            raise ValueError(f"Неподдерживаемое количество каналов NUM_CHANNELS={NUM_CHANNELS}. Проверьте вашу конфигурацию.")

        # Final check to ensure the shape is correct after all transformations
        if image.shape[-1] != NUM_CHANNELS:
            raise RuntimeError(f"После обработки изображение {self.images_filepaths[i]} имеет {image.shape[-1]} каналов, но ожидалось {NUM_CHANNELS}. Проверьте файл и логику обработки каналов.")


        # resize image to input_image_reshape
        image = cv2.resize(image, self.input_image_reshape)

        # Read the mask using rasterio for robust TIFF loading
        with rasterio.open(self.masks_filepaths[i]) as src_mask:
            mask = src_mask.read(1) # Read first band of mask

        # Update the mask: Set foreground_class to 1 and the rest to 0
        mask_remap = np.where(mask == self.foreground_class, 1, 0).astype(np.uint8)

        # resize mask to input_image_reshape
        mask_remap = cv2.resize(mask_remap, self.input_image_reshape, interpolation=cv2.INTER_NEAREST) # INTER_NEAREST для масок

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask_remap)
            image, mask_remap = sample["image"], sample["mask"]

        # Convert to PyTorch tensors
        # HWC -> CHW and normalize to [0, 1]
        image = torch.tensor(image).float().permute(2, 0, 1) / 255.0 # Нормализация здесь

        # Ensure mask is LongTensor
        mask_remap = torch.tensor(mask_remap).long()

        return image, mask_remap


    def __len__(self):
        return len(self.images_filepaths)


# Define a class for the CamVid model
class CamVidModel(torch.nn.Module):
    """
    A PyTorch model for binary segmentation using the Segmentation Models
    PyTorch library.

    Parameters:
    ----------

    - arch (str): The architecture name of the segmentation model
        (e.g., 'Unet', 'FPN').
    - encoder_name (str): The name of the encoder to use
        (e.g., 'resnet34', 'vgg16').
    - in_channels (int, optional): Number of input channels (e.g., 3 for RGB).
    - out_classes (int, optional): Number of output classes (e.g., 1 for binary)
    **kwargs: Additional keyword arguments to pass to the model
    creation function.
    """

    def __init__(self, arch, encoder_name, in_channels=NUM_CHANNELS, out_classes=NUM_CLASSES, **kwargs): # Адаптировано in_channels
        super().__init__()
        # Адаптировано mean и std для 1, 2 или 3 каналов
        if NUM_CHANNELS == 1:
            mean_values = [0.5]
            std_values = [0.5]
        elif NUM_CHANNELS == 2: # Добавлена логика для 2 каналов
            mean_values = [0.5, 0.5]
            std_values = [0.5, 0.5]
        elif NUM_CHANNELS == 3:
            mean_values = [0.485, 0.456, 0.406] # ImageNet mean
            std_values = [0.229, 0.224, 0.225] # ImageNet std
        else:
            # Для других количеств каналов, просто дублируем 0.5
            mean_values = [0.5] * NUM_CHANNELS
            std_values = [0.5] * NUM_CHANNELS

        self.mean = torch.tensor(mean_values).view(1, NUM_CHANNELS, 1, 1).to(device)
        self.std = torch.tensor(std_values).view(1, NUM_CHANNELS, 1, 1).to(device)
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


def visualize(output_dir, image_filename, **images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        # Для одноканальных изображений imshow может требовать cmap='gray'
        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
            plt.imshow(image.squeeze(), cmap='gray')
        else:
            # Для многоканальных изображений (например, 2 или 3)
            # Если 2 канала, matplotlib может интерпретировать как luminance + alpha
            # или как R/G. Для корректного отображения 2-канального изображения
            # может потребоваться преобразование в 3-канальное RGB (дублирование канала)
            # или отображение отдельных каналов.
            # Здесь мы предполагаем, что для визуализации 2-канальное изображение
            # может быть отображено как RGB, дублируя один из каналов или создавая псевдо-RGB.
            # Для простоты, если image.shape[-1] == 2, отобразим первый канал в оттенках серого.
            # Если вы хотите видеть оба канала, вам нужно будет создать отдельный subplot для каждого.
            if image.ndim == 3 and image.shape[-1] == 2:
                plt.imshow(image[:,:,0], cmap='gray') # Отображаем первый канал
                logging.warning("Визуализация 2-канального изображения: отображается только первый канал. Для просмотра всех каналов создайте отдельные subplots.")
            else:
                plt.imshow(image)
    # plt.show() # Убрано, чтобы не блокировать выполнение в headless-среде
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
                # Адаптировано для 1 или 2 каналов: .squeeze() убирает канал, если он 1
                # Если NUM_CHANNELS = 2, input_img_display будет (H, W, 2)
                input_img_display = images[i].cpu().numpy().transpose(1, 2, 0)
                output_mask_display = output.squeeze().cpu().numpy()

                visualize(
                    output_dir,
                    f"output_{i}.png",
                    input_image=input_img_display,
                    output_mask=output_mask_display,
                    binary_mask=output_mask_display > 0.5, # Используем output_mask_display
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
logging.info("Собираем пути к изображениям и маскам...")
all_image_paths = sorted([os.path.join(images_root_dir, f) for f in os.listdir(images_root_dir) if f.lower().endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg'))])
all_mask_paths = sorted([os.path.join(masks_root_dir, f) for f in os.listdir(masks_root_dir) if f.lower().endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg'))])

if len(all_image_paths) != len(all_mask_paths):
    raise ValueError("Количество изображений и масок не совпадает! Убедитесь, что для каждого изображения есть соответствующая маска.")

# Разделение путей к файлам на обучающую, валидационную и тестовую выборки
# Сначала разделяем на train/temp, затем temp на val/test
train_image_paths, temp_image_paths, train_mask_paths, temp_mask_paths = train_test_split(
    all_image_paths, all_mask_paths, test_size=0.3, random_state=42
)
val_image_paths, test_image_paths, val_mask_paths, test_mask_paths = train_test_split(
    temp_image_paths, temp_mask_paths, test_size=0.5, random_state=42 # 0.5 от 0.3 = 0.15 от всего
)

logging.info(f"Найдено изображений: {len(all_image_paths)}")
logging.info(f"Обучающая выборка: {len(train_image_paths)} изображений")
logging.info(f"Валидационная выборка: {len(val_image_paths)} изображений")
logging.info(f"Тестовая выборка: {len(test_image_paths)} изображений")


train_dataset = Dataset(
    train_image_paths,
    train_mask_paths,
    input_image_reshape=input_image_reshape,
    foreground_class=foreground_class,
)
valid_dataset = Dataset(
    val_image_paths,
    val_mask_paths,
    input_image_reshape=input_image_reshape,
    foreground_class=foreground_class,
)
test_dataset = Dataset(
    test_image_paths,
    test_mask_paths,
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
sample = train_dataset[0]
visualize(
    output_dir,
    "train_sample.png",
    train_image=sample[0].cpu().numpy().transpose(1, 2, 0), # Не используем .squeeze() здесь, так как каналов 2
    train_mask=sample[1].squeeze().cpu().numpy(),
)

# Visualize and save validation sample
sample = valid_dataset[0]
visualize(
    output_dir,
    "validation_sample.png",
    validation_image=sample[0].cpu().numpy().transpose(1, 2, 0), # Не используем .squeeze() здесь
    validation_mask=sample[1].squeeze().cpu().numpy(),
)

# Visualize and save test sample
sample = test_dataset[0]
visualize(
    output_dir,
    "test_sample.png",
    test_image=sample[0].cpu().numpy().transpose(1, 2, 0), # Не используем .squeeze() здесь
    test_mask=sample[1].squeeze().cpu().numpy(),
)

# ----------------------------
# Create and train the model
# ----------------------------
max_iter = epochs_max * len(train_dataloader)  # Total number of iterations

# Адаптировано in_channels и out_classes
model = CamVidModel("Unet", "resnet34", in_channels=NUM_CHANNELS, out_classes=NUM_CLASSES)

# Training loop
model = model.to(device)

# Define the Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)

# Define the learning rate scheduler
loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True) # DiceLoss для бинарной сегментации

# Если вы хотите использовать CosineAnnealingLR, убедитесь, что T_max корректен
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_max, eta_min=eta_min) # T_max = epochs_max для CosineAnnealingLR по эпохам

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
