# networks/hmdmv.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from itertools import combinations
import timm

class AllCombMultiImage(nn.Module):
    def __init__(self, arch, num_classes, num_view, pretrained_weights=True):
        super().__init__()
        self.num_view = num_view
        self.num_classes = num_classes
        self.pretrained_weights = pretrained_weights

        drop_rate = .0 if 'tiny' in arch else .1
        self.model = timm.create_model(
            arch,
            pretrained=self.pretrained_weights,
            num_classes=self.num_classes,
            drop_rate=drop_rate
        )
        for block in self.model.blocks:
            block.attn.fused_attn = False

        self.embed_dim = self.model.embed_dim

        for block in self.model.blocks:
            block.attn.proj_drop = nn.Dropout(p=0.0)

        self.img_embed_matrix = nn.Parameter(torch.zeros(1, num_view, self.embed_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.img_embed_matrix)
        nn.init.zeros_(self.model.head.weight)
        nn.init.zeros_(self.model.head.bias)

    def format_multi_image_tokens(self, x, batch_size, tokens_per_image, view_ids):
        if isinstance(view_ids, int):
            view_ids = tuple(range(view_ids))
        else:
            view_ids = tuple(view_ids)

        num_views = len(view_ids)

        x = einops.rearrange(x, '(b n) s c -> b (n s) c', b=batch_size, n=num_views)
        first_img_token_idx = 0
        if self.model.cls_token is not None:
            for i in range(1, num_views):
                excess_cls_index = i * tokens_per_image + 1
                x = torch.cat((x[:, :excess_cls_index], x[:, excess_cls_index + 1:]), dim=1)
            first_img_token_idx = 1

        image_embeddings = F.normalize(self.img_embed_matrix[:, view_ids], dim=-1)  # [1, num_views, D]
        x[:, first_img_token_idx:] += torch.repeat_interleave(image_embeddings, tokens_per_image, dim=1)
        return x

    def run_blocks(self, x):
        for blk in self.model.blocks:
            x = blk(x)
        return x

    # -------------------------
    # NEW: pure inference forward (full-view only)
    # -------------------------
    def forward_infer(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, V, C, H, W], where V == self.num_view
        return: full_view_logits [B, num_classes]
        """
        batch_size, V, C, H, W = x.shape
        assert V == self.num_view, f"Expected V={self.num_view}, got V={V}"

        x = einops.rearrange(x, 'b n c h w -> (b n) c h w')           # [B*V, C, H, W]
        x = self.model.patch_embed(x)                                 # [B*V, T, D] (T=patch tokens)
        tokens_per_image = x.shape[1]
        x = self.model._pos_embed(x)                                  # [B*V, 1+T, D] (cls added inside)

        x = einops.rearrange(x, '(b n) s c -> b n s c', b=batch_size, n=self.num_view)
        comb_tokens = torch.cat([x[:, i].unsqueeze(1) for i in range(self.num_view)], dim=1)
        comb_tokens = einops.rearrange(comb_tokens, 'b n s c -> (b n) s c')
        comb_tokens = self.format_multi_image_tokens(
            comb_tokens, batch_size, tokens_per_image, view_ids=tuple(range(self.num_view))
        )

        comb_tokens = self.run_blocks(comb_tokens)
        comb_tokens = self.model.norm(comb_tokens)
        full_view_logits = self.model.forward_head(comb_tokens)
        return full_view_logits

    def forward(self, x):
        # (training) all combinations - unchanged
        batch_size = len(x)
        output_dict = {}

        x = einops.rearrange(x, 'b n c h w -> (b n) c h w')
        x = self.model.patch_embed(x)
        tokens_per_image = x.shape[1]
        x = self.model._pos_embed(x)

        tokens = x.clone()
        tokens = self.run_blocks(tokens)
        tokens = self.model.norm(tokens)
        output_dict['1_view'] = self.model.forward_head(tokens)

        x = einops.rearrange(x, '(b n) s c -> b n s c', b=batch_size, n=self.num_view)
        for comb_size in range(2, self.num_view + 1):
            comb_name = f'{comb_size}_view'
            comb_logits_list = []

            view_combinations = list(combinations(range(self.num_view), comb_size))
            for comb in view_combinations:
                comb_tokens = torch.cat([x[:, i].unsqueeze(1) for i in comb], dim=1)
                comb_tokens = einops.rearrange(comb_tokens, 'b n s c -> (b n) s c')
                comb_tokens = self.format_multi_image_tokens(comb_tokens, batch_size, tokens_per_image, view_ids=comb)
                comb_tokens = self.run_blocks(comb_tokens)
                comb_tokens = self.model.norm(comb_tokens)
                comb_logits = self.model.forward_head(comb_tokens)
                comb_logits_list.append(comb_logits)

            comb_logits = torch.stack(comb_logits_list, dim=1)            # [B, NC, K]
            output_dict[comb_name] = einops.rearrange(comb_logits, 'b nc k -> (b nc) k')  # [B*NC, K]

        return output_dict

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward_infer(x)

    def inference(self, x):
        self.eval()
        with torch.no_grad():
            batch_size, num_views, C, H, W = x.shape

            x = einops.rearrange(x, 'b n c h w -> (b n) c h w')
            x = self.model.patch_embed(x)
            tokens_per_image = x.shape[1]
            x = self.model._pos_embed(x)
            x = einops.rearrange(x, '(b n) s c -> b n s c', b=batch_size, n=num_views)

            comb_tokens = torch.cat([x[:, i].unsqueeze(1) for i in range(num_views)], dim=1)
            comb_tokens = einops.rearrange(comb_tokens, 'b n s c -> (b n) s c')
            comb_tokens = self.format_multi_image_tokens(comb_tokens, batch_size, tokens_per_image, view_ids=tuple(range(num_views)))

            comb_tokens = self.run_blocks(comb_tokens)
            comb_tokens = self.model.norm(comb_tokens)
            input_view_logits = self.model.forward_head(comb_tokens)

            return input_view_logits
