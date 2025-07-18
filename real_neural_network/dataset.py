from torch.utils.data import Dataset as BaseDataset
import torch
import numpy as np
import os
import cv2
import rasterio
from rasterio.plot import reshape_as_raster, reshape_as_image



class Dataset(BaseDataset):
    """
    A custom dataset class for binary segmentation tasks.

    Parameters:
    ----------

    - images_dir (str): Directory containing the input images.
    - masks_dir (str): Directory containing the corresponding masks.
    - input_image_reshape (tuple, optional): Desired shape for the input
      images and masks. Default is (320, 320).
    - foreground_class (int, optional): The class value in the mask to be
      considered as the foreground. Default is 1.
    - augmentation (callable, optional): A function/transform to apply to the
      images and masks for data augmentation.
    """

    def __init__(
        self,
        images_dir,
        masks_dir,
        input_image_reshape,
        foreground_class,
        augmentation=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_filepaths = [
            os.path.join(images_dir, image_id) for image_id in self.ids
        ]
        self.masks_filepaths = [
            os.path.join(masks_dir, image_id) for image_id in self.ids
        ]

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
        # print(self.images_filepaths[i])
        # Read the image
        image = cv2.imread(
            self.images_filepaths[i], cv2.IMREAD_COLOR_RGB
        )  # Read image as RGB
        # print(image is None, image.shape)
        
        # resize image to input_image_reshape
        image = cv2.resize(image, self.input_image_reshape)

        # Read the mask in grayscale mode
        mask = cv2.imread(self.masks_filepaths[i], 0)

        # Update the mask: Set foreground_class to 1 and the rest to 0
        mask_remap = np.where(mask == self.foreground_class, 1, 0).astype(np.uint8)

        # resize mask to input_image_reshape
        mask_remap = cv2.resize(mask_remap, self.input_image_reshape)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask_remap)
            image, mask_remap = sample["image"], sample["mask"]

        # Convert to PyTorch tensors
        # Add channel dimension if missing
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        # HWC -> CHW and normalize to [0, 1]
        image = torch.tensor(image).float().permute(2, 0, 1) / 255.0

        # Ensure mask is LongTensor
        mask_remap = torch.tensor(mask_remap).long()

        return image, mask_remap

    def __len__(self):
        return len(self.ids)


"""
class Dataset(BaseDataset):
    
    A custom dataset class for binary segmentation tasks.

    Parameters:
    ----------

    - images_dir (str): Directory containing the input images.
    - masks_dir (str): Directory containing the corresponding masks.
    - input_image_reshape (tuple, optional): Desired shape for the input
      images and masks. Default is (320, 320).
    - foreground_class (int, optional): The class value in the mask to be
      considered as the foreground. Default is 1.
    - augmentation (callable, optional): A function/transform to apply to the
      images and masks for data augmentation.
    

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images_dir,
        masks_dir,
        input_image_reshape,
        foreground_class,
        augmentation=None,
    ):
        self.image_paths = [
            os.path.join(images_dir, fname) for fname in os.listdir(images_dir)
        ]
        self.mask_paths = [
            os.path.join(masks_dir, fname) for fname in os.listdir(images_dir)
        ]
        self.input_image_reshape = input_image_reshape
        self.foreground_class = foreground_class
        self.augmentation = augmentation

    def __getitem__(self, i):
    # Load and resize image
        image = cv2.imread(self.image_paths[i], cv2.IMREAD_COLOR)
        image = cv2.resize(image, self.input_image_reshape)

        # Load and resize mask (grayscale)
        mask = cv2.imread(self.mask_paths[i], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Mask at {self.mask_paths[i]} could not be read")

        # DEBUG: print unique values in raw mask
        unique_before = np.unique(mask)
        # print(f"[DEBUG] Unique values in raw mask {self.mask_paths[i]}: {unique_before}")

        # Resize mask with NEAREST to preserve labels
        mask = cv2.resize(mask, self.input_image_reshape, interpolation=cv2.INTER_NEAREST)

        # Convert to binary
        mask_bin = np.where(mask == self.foreground_class, 1, 0).astype(np.uint8)

        # DEBUG: print unique values in binary mask
        unique_after = np.unique(mask_bin)
        # print(f"[DEBUG] Unique values in binarized mask: {unique_after}")

        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask_bin)
            image = augmented["image"]
            mask_bin = augmented["mask"]

        image = torch.tensor(image).float().permute(2, 0, 1) / 255.0  # [3,H,W]
        mask_bin = torch.tensor(mask).unsqueeze(0).long()
       # [1,H,W]

        return image, mask_bin

    def __len__(self):
        return len(self.image_paths)


class Dataset(BaseDataset):
    def __init__(
        self,
        images_dir,
        masks_dir,
        input_image_reshape=(320, 320),
        foreground_class=1,
        augmentation=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_filepaths = [
            os.path.join(images_dir, image_id) for image_id in self.ids
        ]
        self.masks_filepaths = [
            os.path.join(masks_dir, image_id) for image_id in self.ids
        ]

        self.input_image_reshape = input_image_reshape
        self.foreground_class = foreground_class
        self.augmentation = augmentation
    #image = reshape_as_image(raster)
    def __getitem__(self, i):
        with rasterio.open(self.images_filepaths[i], strict = False) as file:
            image = file.read()

        image = np.concatenate([image, [image[0]+image[1]]], axis=0)
        image = reshape_as_image(image)
            
        # image = np.expand_dims(image, axis=-1)  # Add channel dimension

        # resize image to input_image_reshape
        image = cv2.resize(image, self.input_image_reshape)

        # Read the mask in grayscale mode
        # mask = cv2.imread(self.masks_filepaths[i], 0)
        with rasterio.open(self.masks_filepaths[i], strict = False) as masochcka:
            mask = reshape_as_image(masochcka.read())

        # Update the mask: Set foreground_class to 1 and the rest to 0
        mask_remap = np.where(mask == self.foreground_class, 1, 0).astype(np.uint8)

        # resize mask to input_image_reshape
        mask_remap = cv2.resize(mask_remap, self.input_image_reshape)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask_remap)
            image, mask_remap = sample["image"], sample["mask"]

        # Convert to PyTorch tensors
        # Add channel dimension if missing
        # if image.ndim == 2:
        #image = np.expand_dims(image, axis=-1)

        # HWC -> CHW and normalize to [0, 1]
        image = torch.tensor(image).float().permute(2, 0, 1) / 255.0

        # Ensure mask is LongTensor
        mask_remap = torch.tensor(mask_remap).long()

        return image, mask_remap

    def __len__(self):
        return len(self.ids)

"""
