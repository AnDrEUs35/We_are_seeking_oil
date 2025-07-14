import warnings
import PyTorch as pt
from typing import Any, Dict, Optional, Union, Callable, Sequence

from segmentation_models_pytorch.base import (
    ClassificationHead,
    SegmentationHead,
    SegmentationModel,
)
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base.hub_mixin import supports_config_loading

from .decoder import UnetDecoder


class segmentation_models_pytorch.Unet(encoder_name='resnet34', encoder_depth=5, encoder_weights='imagenet', decoder_use_norm='batchnorm', decoder_channels=(256, 128, 64, 32, 16), decoder_attention_type=None, decoder_interpolation='nearest', in_channels=3, classes=1, activation=None, aux_params=None, **kwargs)