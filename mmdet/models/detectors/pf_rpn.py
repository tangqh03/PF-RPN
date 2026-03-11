# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from ..layers import SinePositionalEncoding
from ..layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerDecoder, GroundingDinoTransformerEncoder)
from .dino import DINO


class AttentionPool2d(nn.Module):

    def __init__(self,
                 spacial_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim + 1, embed_dim) / embed_dim**0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, _, height, width = x.shape
        x = x.reshape(batch_size, x.shape[1], height * width).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)

        cls_pos = self.positional_embedding[0:1, :]
        spatial_pos = self.positional_embedding[1:1 + height * width, :]
        spatial_pos = spatial_pos.reshape(height, width, self.embed_dim)
        spatial_pos = spatial_pos.reshape(-1, self.embed_dim)
        positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)

        x = x + positional_embedding[:, None, :]
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False)

        x = x.permute(1, 2, 0)
        global_feat = x[:, :, 0]
        feature_map = x[:, :, 1:].reshape(batch_size, -1, height, width)
        return global_feat, feature_map


class Attention(nn.Module):

    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 qkv_bias: bool = False,
                 qk_scale: float = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        batch_size, query_len, channels = q.shape
        _, key_len, _ = k.shape
        q = self.q_proj(q).reshape(batch_size, query_len, self.num_heads,
                                   channels // self.num_heads)
        k = self.k_proj(k).reshape(batch_size, key_len, self.num_heads,
                                   channels // self.num_heads)
        v = self.v_proj(v).reshape(batch_size, key_len, self.num_heads,
                                   channels // self.num_heads)

        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(
            batch_size, query_len, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = Attention(d_model, nhead, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(d_model * 4, d_model))

    def forward(self, x: Tensor, mem: Tensor) -> Tensor:
        q = k = v = self.norm1(x)
        x = x + self.self_attn(q, k, v)
        q = self.norm2(x)
        x = x + self.cross_attn(q, mem, mem)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x


class ContextDecoder(nn.Module):

    def __init__(self,
                 transformer_width: int = 256,
                 transformer_heads: int = 4,
                 transformer_layers: int = 6,
                 visual_dim: int = 1024,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__()

        self.memory_proj = nn.Sequential(nn.LayerNorm(visual_dim),
                                         nn.Linear(visual_dim,
                                                   transformer_width),
                                         nn.LayerNorm(transformer_width))

        self.text_proj = nn.Sequential(nn.LayerNorm(visual_dim),
                                       nn.Linear(visual_dim,
                                                 transformer_width))

        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(transformer_width, transformer_heads,
                                    dropout)
            for _ in range(transformer_layers)
        ])

        self.out_proj = nn.Sequential(nn.LayerNorm(transformer_width),
                                      nn.Linear(transformer_width, visual_dim))

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, text: Tensor, visual: Tensor) -> Tensor:
        visual = self.memory_proj(visual)
        x = self.text_proj(text)
        for layer in self.decoder:
            x = layer(x, visual)
        return self.out_proj(x)


@MODELS.register_module()
class PFRPN(DINO):
    """PF-RPN: clean implementation used for paper release.

    This class keeps only the paper path:
    1) sparse MoE pseudo-text construction
    2) cascade self-support enhancement
    3) centerness-score-guided query selection
    """

    def __init__(self,
                 *args,
                 num_pseudo_tokens: int = 3,
                 sp_thr: float = 0.3,
                 sp_iter_num: int = 3,
                 topk: int = 2,
                 **kwargs) -> None:
        self.num_pseudo_tokens = num_pseudo_tokens
        self.sp_thr = sp_thr
        self.sp_iter_num = sp_iter_num
        self.topk = topk
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = GroundingDinoTransformerEncoder(**self.encoder)
        self.decoder = GroundingDinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        self.learnable_text_embedding = nn.Parameter(
            torch.randn(1, self.num_pseudo_tokens, self.embed_dims))
        self.vis_proj = AttentionPool2d(
            spacial_dim=30000, embed_dim=self.embed_dims, num_heads=32)
        self.router = nn.Linear(self.embed_dims, 1)
        self.meta_net = ContextDecoder(
            transformer_width=self.embed_dims,
            transformer_heads=4,
            transformer_layers=3,
            visual_dim=self.embed_dims,
            dropout=0.1,
            style='pytorch')

    def init_weights(self) -> None:
        super().init_weights()
        nn.init.trunc_normal_(self.learnable_text_embedding, std=0.02)
        self.meta_net.apply(self.meta_net._init_weights)

    def cascade_self_prompt_enhancement(
            self,
            vis_embedding: Tensor,
            text_embedding: Tensor,
            spatial_shapes: list,
            thr: float = 0.7,
            iter_num: int = 3) -> Tensor:
        for _ in range(iter_num):
            offset = 0
            for height, width in spatial_shapes:
                vis_feat_map = vis_embedding[:, offset:offset + height * width, :]
                offset += height * width
                cos_sim = F.normalize(vis_feat_map, dim=-1) @ F.normalize(
                    text_embedding, dim=-1).permute(0, 2, 1).contiguous()
                mask_f = (cos_sim.clamp(0, 1) > thr).float()
                masked_vis_feat_sum = (vis_feat_map * mask_f).sum(
                    dim=1, keepdim=True)
                masked_count = mask_f.sum(dim=1, keepdim=True).clamp(min=1e-6)
                map_vis_feat = masked_vis_feat_sum / masked_count
                text_embedding = text_embedding + map_vis_feat
        return text_embedding

    def _build_sparse_moe_pseudo_text(
            self, memory: Tensor,
            spatial_shapes: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = memory.shape[0]
        multi_level_global_vis_feats = []
        expert_feats = []

        offset = 0
        for height, width in spatial_shapes.tolist():
            num_points = height * width
            vis_feat_l = memory[:, offset:offset + num_points, :]
            vis_feat_l = vis_feat_l.reshape(batch_size, height, width,
                                            -1).permute(0, 3, 1,
                                                        2).contiguous()
            vis_global_l, vis_local_l = self.vis_proj(vis_feat_l)
            multi_level_global_vis_feats.append(vis_global_l)
            vis_feat = torch.cat([vis_global_l[:, :, None],
                                  vis_local_l.flatten(2)],
                                 dim=2).permute(0, 2, 1).contiguous()
            expert_feats.append(vis_feat)
            offset += num_points

        expert_feat = torch.stack(multi_level_global_vis_feats, dim=1)
        router_weights = self.router(expert_feat)

        text_feat = self.learnable_text_embedding.expand(batch_size, -1,
                                                         -1).to(memory.device)
        k = min(self.topk, router_weights.size(1))
        idx_selected = torch.topk(router_weights, k=k,
                                  dim=1).indices.squeeze(-1)
        self.router._last_idx_selected = idx_selected.detach()

        router_weights_selected = nn.Softmax(dim=1)(
            router_weights.gather(1, idx_selected.unsqueeze(-1))).squeeze(-1)
        meta_out = torch.zeros(
            batch_size,
            idx_selected.size(1),
            text_feat.size(1),
            text_feat.size(2),
            device=text_feat.device,
            dtype=text_feat.dtype)

        for expert_id, expert_feat in enumerate(expert_feats):
            selected = idx_selected == expert_id
            if not selected.any():
                continue
            b_idx, k_idx = selected.nonzero(as_tuple=True)
            meta_out[b_idx, k_idx] = self.meta_net(text_feat[b_idx],
                                                   expert_feat[b_idx])

        pseudo_text_embeds = (router_weights_selected[:, :, None, None] *
                              meta_out).sum(dim=1)
        return pseudo_text_embeds, router_weights

    def _init_text_dict(self, text_dict: Dict, text_embeddings: Tensor) -> None:
        token_shape = text_embeddings.shape[:2]
        text_dict['embedded'] = text_embeddings
        text_dict['text_token_mask'] = torch.ones(
            size=token_shape, dtype=torch.bool, device=text_embeddings.device)
        text_dict['position_ids'] = torch.zeros(
            size=token_shape, dtype=torch.int64, device=text_embeddings.device)
        text_dict['masks'] = torch.ones(
            size=(token_shape[0], token_shape[1], token_shape[1]),
            dtype=torch.bool,
            device=text_embeddings.device)

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)
        spatial_shapes = encoder_inputs_dict['spatial_shapes']
        memory = encoder_inputs_dict['feat']

        pseudo_text_embeds, router_weights = self._build_sparse_moe_pseudo_text(
            memory, spatial_shapes)
        enhanced = self.cascade_self_prompt_enhancement(
            vis_embedding=memory,
            text_embedding=pseudo_text_embeds[:, 1:2, :],
            spatial_shapes=spatial_shapes.tolist(),
            thr=self.sp_thr,
            iter_num=self.sp_iter_num)
        text_embeddings = pseudo_text_embeds.clone()
        text_embeddings[:, 1:2, :] = enhanced
        self._init_text_dict(text_dict, text_embeddings)

        encoder_outputs_dict = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)
        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        head_inputs_dict['router_weights'] = router_weights
        return head_inputs_dict

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor, feat_pos: Tensor,
                        spatial_shapes: Tensor, level_start_index: Tensor,
                        valid_ratios: Tensor, text_dict: Dict) -> Dict:
        text_token_mask = text_dict['text_token_mask']
        memory, memory_text = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            memory_text=text_dict['embedded'],
            text_attention_mask=~text_token_mask,
            position_ids=text_dict['position_ids'],
            text_self_attention_masks=text_dict['masks'])
        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            memory_text=memory_text,
            text_token_mask=text_token_mask)
        return encoder_outputs_dict

    def _select_score_guided_queries(self, conf_score: Tensor,
                                     cls_score: Tensor) -> Tensor:
        batch_size = conf_score.shape[0]
        conf_num = self.num_queries // 2
        cls_num = self.num_queries - conf_num

        conf_topk = torch.topk(conf_score, k=conf_num, dim=1).indices
        cls_topk = torch.topk(cls_score, k=self.num_queries, dim=1).indices

        cls_remaining = []
        for b_idx in range(batch_size):
            conf_idx = conf_topk[b_idx]
            cls_idx = cls_topk[b_idx]
            dup_mask = (cls_idx.unsqueeze(1) == conf_idx.unsqueeze(0)).any(
                dim=1)
            remaining = cls_idx[~dup_mask]
            if remaining.numel() < cls_num:
                if remaining.numel() == 0:
                    remaining = cls_idx[:1].repeat(cls_num)
                else:
                    pad_num = cls_num - remaining.numel()
                    remaining = torch.cat(
                        [remaining, remaining[-1:].repeat(pad_num)], dim=0)
            else:
                remaining = remaining[:cls_num]
            cls_remaining.append(remaining.unsqueeze(0))

        cls_remaining = torch.cat(cls_remaining, dim=0)
        topk_indices = torch.cat([conf_topk, cls_remaining], dim=1)
        return topk_indices

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        batch_size = memory.shape[0]
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)

        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory, memory_text,
                                     text_token_mask)
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals
        cls_score = enc_outputs_class.max(-1)[0].sigmoid()

        enc_outputs_conf = self.bbox_head.conf_branches[self.decoder.num_layers](
            output_memory)
        enc_outputs_conf = enc_outputs_conf.squeeze(-1).sigmoid()

        topk_indices = self._select_score_guided_queries(enc_outputs_conf,
                                                         cls_score)
        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()
        topk_conf_score = torch.gather(enc_outputs_conf.unsqueeze(-1), 1,
                                       topk_indices.unsqueeze(-1))

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, batch_size, 1).transpose(0, 1)
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask)

        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            enc_outputs_conf=topk_conf_score,
            dn_meta=dn_meta) if self.training else dict()
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        return decoder_inputs_dict, head_inputs_dict

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Dict[str, Tensor]:
        text_dict = {}
        visual_features = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(visual_features, text_dict,
                                                    batch_data_samples)

        num_tokens = text_dict['text_token_mask'].shape[1]
        positive_idx = 1 if num_tokens > 1 else 0
        for i, data_samples in enumerate(batch_data_samples):
            text_token_mask = text_dict['text_token_mask'][i]
            num_gt = data_samples.gt_instances.labels.shape[0]
            positive_map = torch.zeros(
                (num_gt, num_tokens),
                dtype=torch.float32,
                device=batch_inputs.device)
            if num_gt > 0:
                positive_map[:, positive_idx] = 1
            data_samples.gt_instances.positive_maps = positive_map
            data_samples.gt_instances.text_token_mask = text_token_mask.unsqueeze(
                0).repeat(num_gt, 1)

        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        visual_feats = self.extract_feat(batch_inputs)

        text_dict = {}
        entities = (['object'],) * batch_inputs.shape[0]
        for data_samples in batch_data_samples:
            data_samples.token_positive_map = [1]

        head_inputs_dict = self.forward_transformer(visual_feats, text_dict,
                                                    batch_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)

        for data_sample, pred_instances, entity in zip(batch_data_samples,
                                                       results_list, entities):
            if len(pred_instances) > 0:
                label_names = []
                for label in pred_instances.labels:
                    label_idx = int(label)
                    if label_idx >= len(entity):
                        warnings.warn(
                            'Unexpected label index; fallback to "unobject".')
                        label_names.append('unobject')
                    else:
                        label_names.append(entity[label_idx])
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances
        return batch_data_samples
