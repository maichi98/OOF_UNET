from monai.networks.layers.factories import Norm
from monai.networks.nets import UNet


def init_baseline_model():

    # Build Unet model :
    model = UNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=1,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    )

    return model
