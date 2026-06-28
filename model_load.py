from models.U_net.U_net_new_evi import UNet2d_evi


def _build_ramMamba_evi():
    from compared_nets.ramMamba.RMAMamba_evi import RMAMamba_S_evi

    return RMAMamba_S_evi(
        pretrained=None,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        depths=[2, 2, 9, 2],
        dims=96,
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_rank_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        # v01 uses the Mamba selective scan path and avoids requiring
        # selective_scan_cuda_core, which is not available in many public setups.
        forward_type="v01",
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        drop_path_rate=0.1,
        patch_norm=True,
        norm_layer="ln",
        downsample_version="v2",
        patchembed_version="v2",
        gmlp=False,
        use_checkpoint=False,
    )


def _build_ultralight_evi():
    from compared_nets.UltraLight_VM_UNet.UltraLight_VM_UNet_evi import UltraLight_VM_UNet_evi

    return UltraLight_VM_UNet_evi(
        num_classes=1,
        input_channels=3,
        c_list=[8, 16, 24, 32, 48, 64],
        split_att="fc",
        bridge=True,
    )


def model_loading(config):
    """Build an evidential segmentation model."""

    if config.network == "unet_evi":
        return UNet2d_evi(3, 4)
    if config.network == "ramMamba_evi":
        return _build_ramMamba_evi()
    if config.network == "UltraLight_VM_UNet_evi":
        return _build_ultralight_evi()
    raise ValueError(
        "Unsupported network {!r}. Available networks are: "
        "unet_evi, ramMamba_evi, UltraLight_VM_UNet_evi.".format(config.network)
    )
