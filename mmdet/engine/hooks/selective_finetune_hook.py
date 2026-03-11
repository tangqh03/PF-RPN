# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook
from mmengine.model.wrappers import is_model_wrapper
import re
from mmdet.registry import HOOKS
import inspect
import csv

# @HOOKS.register_module()
# class SelectiveFinetuneHook(Hook):
#     def __init__(self):
#         super().__init__()
#     def before_train(self, runner):
#         model = runner.model.module if is_model_wrapper(runner.model) else runner.model
#         for name, param in model.named_parameters():
#             if ('bbox_head.cls_branches' in name):
#                 param.requires_grad = True
#                 print(f"[Trainable] {name}")

#             else:
#                 param.requires_grad = False
#                 print(f"[Frozen]    {name}")


# @HOOKS.register_module()
# class SelectiveFinetuneHook(Hook):
#     def __init__(self):
#         super().__init__()
#     def before_train(self, runner):
#         model = runner.model.module if is_model_wrapper(runner.model) else runner.model
#         for name, param in model.named_parameters():
#             if any(k not in name for k in ['encoder']):
#                 param.requires_grad = True
#                 print(f"[Trainable] {name}")

#             else:
#                 if 'language_model' in name:
#                     param.requires_grad = True
#                     print(f"[Trainable] {name}")
#                 else:
#                     param.requires_grad = False
#                     print(f"[Frozen]    {name}")

# # 这个就是只修改language 
# @HOOKS.register_module()
# class SelectiveFinetuneHook(Hook):
#     def __init__(self):
#         super().__init__()
#     def before_train(self, runner):
#         model = runner.model.module if is_model_wrapper(runner.model) else runner.model
#         for name, param in model.named_parameters():
#             if all(k not in name for k in ['decoder','encoder']):
#                 param.requires_grad = True
#                 print(f"[Trainable] {name}")

#             else:
#                 if 'language_model' in name:
#                     param.requires_grad = True
#                     print(f"[Trainable] {name}")
#                 else:
#                     param.requires_grad = False
#                     print(f"[Frozen]    {name}")


# @HOOKS.register_module()
# class SelectiveFinetuneHook(Hook):
#     def __init__(self):
#         super().__init__()
#     def before_train(self, runner):
#         model = runner.model.module if is_model_wrapper(runner.model) else runner.model
#         for name, param in model.named_parameters():
#             if('bbox_head.cls_branches' in name) or("language_model.language_backbone.body.text_fs_adapter" in name) or('image_fs_adapter' in name):
                
#                 param.requires_grad = True
#                 print(f"[Trainable] {name}")

#             else:
#                 param.requires_grad = False
#                 print(f"[Frozen]    {name}")



# 正则表达式匹配 attention 后的 FFN 层（intermediate 或 output 但不包含 attention）
bert_ffn_pattern = re.compile(
    r"language_model\.language_backbone\.body\.model\.encoder\.layer\.\d+\.(intermediate|output)(?!.*attention)"
)
vit_ffn_pattern = re.compile(
    r"^backbone\.stages\.\d+\.blocks\.\d+\.ffn\.layers\.\d+(\.\d+)?\.(weight|bias)$"
)
# bert_ffn_pattern = re.compile(
#     r"language_model\.language_backbone\.body\.model\.encoder\.layer\.("     # 前缀
#     r"[4-9]|"                  # 一位数大于3
#     r"[1-9]\d+|"               # 两位及以上数字
#     r"1\d+|"                   # 支持10, 11, ...
#     r"\d{2,}"                  # 或者泛化所有两位及以上
#     r")\.(intermediate|output)(?!.*attention)"
# )[Frozen]    decoder.layers.0.ffn.layers.0.0.bias   decoder.layers.0.ffn.layers.1.weight
# vit_ffn_pattern = re.compile(
#     r"^backbone\.stages\.([1-9]\d*)\.blocks\.\d+\.ffn\.layers\.\d+(\.\d+)?\.(weight|bias)$"
# )

            # elif('bbox_head.reg_branches' in name):
            #     param.requires_grad = True
            #     print(f"[Trainable] {name}")
@HOOKS.register_module()
class SelectiveFinetuneHook(Hook):
    def __init__(self):
        super().__init__()
    def before_train(self, runner):
        model = runner.model.module if is_model_wrapper(runner.model) else runner.model
        for name, param in model.named_parameters():
            if ('bbox_head' in name) or ('learnable_text_embedding' in name) or ('meta_net' in name) or ('text_meta_net' in name) or ('query_embedding' in name) or ('router' in name) or ('conf_branches' in name) or ('vis_proj' in name) or ('attentive_pooling_projection' in name) or ('cmm' in name) or ('multi_scale_feature' in name) or ('cg' in name) or ('sg' in name) or ('channel_prompt' in name) or ('spatial_prompt' in name):
                if 'dinov2' in name or 'foundation_model' in name or 'language_model' in name:
                    param.requires_grad = False
                    print(f"[Frozen] {name}")
                else :
                    param.requires_grad = True
                    print(f"[Trainable] {name}")
            elif 'dinov2' in name or 'foundation_model' in name or 'language_model' in name:
                param.requires_grad = False
                print(f"[Frozen] {name}")

            # elif bert_ffn_pattern.search(name):
            #     if 'dinov2' in name or 'foundation_model' in name:
            #         param.requires_grad = False
            #         print(f"[Frozen] {name}")
            #     else :
            #         param.requires_grad = True
            #         print(f"[Trainable] {name}")
            elif vit_ffn_pattern.search(name):
                if 'dinov2' in name or 'foundation_model' in name or 'language_model' in name:
                    param.requires_grad = False
                    print(f"[Frozen] {name}")
                else :
                    param.requires_grad = True
                    print(f"[Trainable] {name}")
            elif ('decoder' in name) and ('ffn' in name):
                if 'dinov2' in name or 'foundation_model' in name or 'language_model' in name:
                    param.requires_grad = False
                    print(f"[Frozen] {name}")
                else :
                    param.requires_grad = True
                    print(f"[Trainable] {name}")
            elif ('encoder' in  name) and ('ffn' in name):
                if 'dinov2' in name or 'foundation_model' in name or 'language_model' in name:
                    param.requires_grad = False
                    print(f"[Frozen] {name}")
                else :
                    param.requires_grad = True
                    print(f"[Trainable] {name}")
            elif ('self.seq_proj' in name) or ('channel_align_layers' in name) or ('featurefc_layers' in name) or ("featureconv" in name) or ('spatial_channel_align_layers' in name) or ('private_align_layers' in name) or('bbox_head' in name):
                if 'dinov2' in name or 'foundation_model' in name or 'language_model' in name:
                    param.requires_grad = False
                    print(f"[Frozen] {name}")
                else :
                    param.requires_grad = True
                    print(f"[Trainable] {name}")
            else:
                param.requires_grad = False
                print(f"[Frozen]    {name}")