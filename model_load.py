from models.U_net.U_net_new_evi import UNet2d_evi


def model_loading(config):
    """Build the public EviRisk-Seg model.

    The simplified release intentionally keeps only the UNet backbone used for
    the core method implementation.
    """

    if config.network != "unet_evi":
        raise ValueError(
            "This public release only includes network='unet_evi'. "
            f"Got: {config.network!r}"
        )
    return UNet2d_evi(3, 4)
