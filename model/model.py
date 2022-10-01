import torch
import torch.nn as nn

import timm
import segmentation_models_pytorch as smp


class EfficientNetB0(nn.Module):
    def __init__(self, pre_trained, num_classes):
        super(EfficientNetB0, self).__init__()
        self.model = timm.create_model(
            "tf_efficientnet_b0_ns", pretrained=pre_trained, num_classes=num_classes
        )

    def forward(self, x):
        out = self.model(x)
        return out


class UNetEfficientNet(nn.Module):
    def __init__(self, num_classes, encoder_path):
        super(UNetEfficientNet, self).__init__()
        self.model = smp.Unet(
            encoder_name="timm-efficientnet-b0",
            encoder_weights="noisy-student",
            in_channels=3,
            classes=num_classes,
        )
        self.model.encoder.load_state_dict(torch.load(encoder_path))

    def forward(self, x):
        out = self.model(x)
        return out
