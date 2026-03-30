#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

import offline_reconstructor_no_gt_local30kv2 as reco_base


class OfflineReconstructorLinearUnsmear(reco_base.OfflineReconstructor):
    """
    Linear-unsmear variant:
    - Keep base architecture and all heads unchanged except unsmear_head.
    - Replace unsmear MLP with a single linear projection.
    - Keep split/unmerge heads in original base form (linear heads).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        embed_dim = int(self.action_head.in_features)
        self.unsmear_head = nn.Linear(embed_dim, 4)

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        incompatible = super().load_state_dict(state_dict, strict=False)
        if strict and (incompatible.missing_keys or incompatible.unexpected_keys):
            print(
                "[OfflineReconstructorLinearUnsmear] Loaded non-strict state dict "
                f"(missing={len(incompatible.missing_keys)}, "
                f"unexpected={len(incompatible.unexpected_keys)})."
            )
        return incompatible

