import segmentation_models_pytorch as smp


class MyUnet:
    def __init__(self, encoder_name, encoder_weights, activation):
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=1,
            classes=5,
            activation=activation,
        )

    def train(self):
