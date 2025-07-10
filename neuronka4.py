import segmentation_models_pytorch as smp

modewl = smp.Unet(
    encoder_name = "resnet34",
    encoder_weights = "imagenet"
    in_channels = 3,
    classes
)