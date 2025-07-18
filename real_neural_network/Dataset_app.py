import cv2
from tqdm import tqdm
from model import Model
import torch 
import segmentation_models_pytorch as smp

with torch.no_grad():
    device = "cpu"
    model = Model("Unet", "resnet34", in_channels=3, out_classes=1)
    model.load_state_dict(torch.load("model1.bin"))
    model.eval()

    image = cv2.imread("Oil_00002.png", cv2.IMREAD_COLOR_RGB)  # Read image as RGB
    # mage = cv2.resize(image, self.input_image_reshape)
    image = torch.tensor(image).to(device)
    image = image.float().permute(2, 0, 1) / 255.0

    out = model(image)
    out = torch.sigmoid(out)

    out = out.squeeze().cpu().numpy()
    out = out > 0.5
    print(out.shape, out)

    from PIL import Image
    im = Image.fromarray(out)
    im.save("./your_file.png")