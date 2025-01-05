# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# This is the same as SegFormer but with 256 embed_dims
# SegF. with C_e=256 in Tab. 7

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    #pretrained='pretrained/mit_b5.pth',
    #backbone=dict(type='mit_b5', style='pytorch'),
    # pretrained = 'pretrained/beit_base_patch16_224_pt22k_ft22k.pth',
    # backbone=dict(
    #     type='BEiTAdapter',
    #     img_size=512,
    #     patch_size=16,
    #     embed_dim=768,
    #     depth=12,
    #     num_heads=12,
    #     mlp_ratio=4,
    #     qkv_bias=True,
    #     use_abs_pos_emb=False,
    #     use_rel_pos_bias=True,
    #     init_values=1e-6,
    #     drop_path_rate=0.2,
    #     conv_inplane=64,
    #     n_points=4,
    #     deform_num_heads=16,
    #     cffn_ratio=0.25,
    #     deform_ratio=0.5,
    #     with_cp=True,
    #     interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
    # ),
    pretrained = 'pretrained/beit_large_patch16_224_pt22k_ft22k.pth',
    backbone=dict(
        type='BEiTAdapter',
        img_size=512,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        use_abs_pos_emb=False,
        use_rel_pos_bias=True,
        init_values=1e-6,
        drop_path_rate=0.3,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        with_cp=True,  # set with_cp=True to save memory
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
    ),
    decode_head=dict(
        type='DAFormerHead',
        in_channels=[1024, 1024, 1024, 1024],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(
            embed_dims=256,
            embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            fusion_cfg=dict(
                type='conv',
                kernel_size=1,
                act_cfg=dict(type='ReLU'),
                norm_cfg=norm_cfg),
        ),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# model2 = dict(
#     type='EncoderDecoder',
#     pretrained='pretrained/mit_b5.pth',
#     backbone=dict(type='mit_b5', style='pytorch'),
#     decode_head=dict(
#         type='DAFormerHead',
#         in_channels=[1024, 1024, 1024, 1024],
#         in_index=[0, 1, 2, 3],
#         channels=256,
#         dropout_ratio=0.1,
#         num_classes=19,
#         norm_cfg=norm_cfg,
#         align_corners=False,
#         decoder_params=dict(
#             embed_dims=256,
#             embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
#             embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
#             fusion_cfg=dict(
#                 type='conv',
#                 kernel_size=1,
#                 act_cfg=dict(type='ReLU'),
#                 norm_cfg=norm_cfg),
#         ),
#         loss_decode=dict(
#             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
#     # model training and testing settings
#     train_cfg=dict(),
#     test_cfg=dict(mode='whole'))