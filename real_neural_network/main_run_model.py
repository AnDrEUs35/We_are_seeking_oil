import logging
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset as BaseDataset
from pathlib import Path
from model import Model
from PIL import Image


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%d:%m:%Y %H:%M:%S",
)


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

input_image_reshape = (128, 128)  # Desired shape for the input images and masks

def test_model(model, output_dir, arr, device, reshape):
    model.eval()
    with torch.no_grad():
        for i in range (len(arr)):
            image = cv2.imread(arr[i])
            image = cv2.resize(image, reshape)
            image = torch.tensor(image).float().permute(2, 0, 1) / 255.0  
            output_img = img_to_mask(model, image.to(device))
            output_img = Image.fromarray(output_img)
            output_img = np.array(output_img)
            #output_img.save(os.path.join(output_dir, f"аутпут{i}.png"))
            print(output_img)
          
def img_to_mask(model, image):        
    output = torch.sigmoid(model(image))
    return output.squeeze().cpu().numpy() > 0.5
   
        
ids = os.listdir(test_dir)
images_filepaths = [
os.path.join(test_dir, image_id) for image_id in ids
]
        

torch.backends.cudnn.benchmark = True
# # Evaluate the model
model = Model("Unet", "resnet34", in_channels=3, out_classes=1)
model.load_state_dict(torch.load("model1.bin"))
test_model(model, output_dir, images_filepaths, device, input_image_reshape)

