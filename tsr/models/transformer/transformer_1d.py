# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# --------
#
# Modified 2024 by the Tripo AI and Stability AI Team.
#
# Copyright (c) 2024 Tripo AI & Stability AI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ...utils import BaseModule
from .basic_transformer_block import BasicTransformerBlock


class Transformer1D(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        num_attention_heads: int = 16
        attention_head_dim: int = 88
        in_channels: Optional[int] = None
        out_channels: Optional[int] = None
        num_layers: int = 1
        dropout: float = 0.0
        norm_num_groups: int = 32
        cross_attention_dim: Optional[int] = None
        attention_bias: bool = False
        activation_fn: str = "geglu"
        only_cross_attention: bool = False
        double_self_attention: bool = False
        upcast_attention: bool = False
        norm_type: str = "layer_norm"
        norm_elementwise_affine: bool = True
        gradient_checkpointing: bool = False

    cfg: Config

    def configure(self) -> None:
        self.num_attention_heads = self.cfg.num_attention_heads
        self.attention_head_dim = self.cfg.attention_head_dim
        inner_dim = self.num_attention_heads * self.attention_head_dim

        linear_cls = nn.Linear

        # 2. Define input layers
        self.in_channels = self.cfg.in_channels

        self.norm = torch.nn.GroupNorm(
            num_groups=self.cfg.norm_num_groups,
            num_channels=self.cfg.in_channels,
            eps=1e-6,
            affine=True,
        )
        self.proj_in = linear_cls(self.cfg.in_channels, inner_dim)

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    self.num_attention_heads,
                    self.attention_head_dim,
                    dropout=self.cfg.dropout,
                    cross_attention_dim=self.cfg.cross_attention_dim,
                    activation_fn=self.cfg.activation_fn,
                    attention_bias=self.cfg.attention_bias,
                    only_cross_attention=self.cfg.only_cross_attention,
                    double_self_attention=self.cfg.double_self_attention,
                    upcast_attention=self.cfg.upcast_attention,
                    norm_type=self.cfg.norm_type,
                    norm_elementwise_affine=self.cfg.norm_elementwise_affine,
                )
                for d in range(self.cfg.num_layers)
            ]
        )

        # 4. Define output layers
        self.out_channels = (
            self.cfg.in_channels
            if self.cfg.out_channels is None
            else self.cfg.out_channels
        )

        self.proj_out = linear_cls(inner_dim, self.cfg.in_channels)

        self.gradient_checkpointing = self.cfg.gradient_checkpointing

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        The [`Transformer1DModel`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Input `hidden_states`.
            encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.

        Returns:
            torch.FloatTensor
        """
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(hidden_states.dtype)
            ) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        batch, _, seq_len = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 1).reshape(
            batch, seq_len, inner_dim
        )
        hidden_states = self.proj_in(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                )

        # 3. Output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = (
            hidden_states.reshape(batch, seq_len, inner_dim)
            .permute(0, 2, 1)
            .contiguous()
        )

        output = hidden_states + residual

        return output
