# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""This code is copied from NVIDIA apex:
      https://github.com/NVIDIA/apex
   with some changes. """

import numbers
import torch
from torch.nn.parameter import Parameter
from torch.nn import init
import importlib
from torch.nn import functional as F
import inspect

from megatron.utils import make_viewless_tensor

try:
    from apex.contrib.layer_norm.layer_norm import FastLayerNormFN

    HAVE_PERSIST_LAYER_NORM = True
except:
    HAVE_PERSIST_LAYER_NORM = False

from apex.normalization.fused_layer_norm import (
    FusedLayerNormAffineFunction,
    FusedRMSNormAffineFunction,
)


global fused_layer_norm_cuda
fused_layer_norm_cuda = None


class MixedFusedLayerNorm(torch.nn.Module):
    def __init__(
        self,
        normalized_shape,
        eps=1e-5,
        no_persist_layer_norm=True,
        context_parallel=False,
        apply_layernorm_1p=False,
        mem_efficient_ln=True,
    ):
        super(MixedFusedLayerNorm, self).__init__()

        self.apply_layernorm_1p = apply_layernorm_1p
        self.mem_efficient_ln = mem_efficient_ln

        global fused_layer_norm_cuda
        fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")

        # List of hiddens sizes supported in the persistent layer norm kernel
        # If the hidden size is not supported, fall back to the non-persistent
        # kernel.
        persist_ln_hidden_sizes = [
            1024,
            1536,
            2048,
            2304,
            3072,
            3840,
            4096,
            5120,
            6144,
            8192,
            10240,
            12288,
            12800,
            15360,
            16384,
            18432,
            20480,
            24576,
            25600,
            30720,
            32768,
            40960,
            49152,
            65536,
        ]
        if (
            normalized_shape not in persist_ln_hidden_sizes
            or not HAVE_PERSIST_LAYER_NORM
        ):
            no_persist_layer_norm = True

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.weight = Parameter(torch.Tensor(*normalized_shape))
        self.bias = Parameter(torch.Tensor(*normalized_shape))
        self.reset_parameters()
        self.no_persist_layer_norm = no_persist_layer_norm
        self.context_parallel = context_parallel

        # set sequence parallelism flag on weight and bias parameters
        setattr(self.weight, "context_parallel", self.context_parallel)
        setattr(self.bias, "context_parallel", self.context_parallel)

    def reset_parameters(self):

        if self.apply_layernorm_1p:
            init.zeros_(self.weight)
            init.zeros_(self.bias)
        else:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input):

        weight = self.weight + 1 if self.apply_layernorm_1p else self.weight
        # CPU path is here for unittest sake.
        if not input.is_cuda:
            print(
                "WARNING! The input of FusedLayerNorm should be on the GPU."
                "This warning should only be triggered in the FusedLayerNorm unit tests."
            )
            return F.layer_norm(
                input, self.normalized_shape, weight, self.bias, self.eps
            )

        if self.no_persist_layer_norm:
            # Apex does not have versions yet (https://github.com/NVIDIA/apex/pull/1648), so we need to inspect
            # the function manually on whether the extra arg introduced in https://github.com/NVIDIA/apex/pull/1715 exists yet
            if (
                "memory_efficient"
                in inspect.getfullargspec(FusedLayerNormAffineFunction.forward).args
            ):
                return FusedLayerNormAffineFunction.apply(
                    input,
                    weight,
                    self.bias,
                    self.normalized_shape,
                    self.eps,
                    self.mem_efficient_ln,
                )
            else:
                return FusedLayerNormAffineFunction.apply(
                    input, weight, self.bias, self.normalized_shape, self.eps
                )
        else:
            output = FastLayerNormFN.apply(input, weight, self.bias, self.eps)

            # Apex's fast layer norm function outputs a 'view' tensor (i.e., has
            # a populated '_base' field). This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            output = make_viewless_tensor(
                inp=output, requires_grad=input.requires_grad, keep_graph=True
            )

            return output


class MixedFusedRMSNorm(torch.nn.Module):
    def __init__(
        self,
        normalized_shape,
        eps=1e-5,
        no_persist_layer_norm=True,
        sequence_parallel=False,
        apply_rmsnorm_1p=False,
        mem_efficient_rms=True,
    ):
        super(MixedFusedRMSNorm, self).__init__()

        self.apply_rmsnorm_1p = apply_rmsnorm_1p
        self.mem_efficient_rms = mem_efficient_rms
        self.norm_fn = FusedRMSNormAffineFunction

        global fused_layer_norm_cuda
        fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")

        # List of hiddens sizes supported in the persistent layer norm kernel
        # If the hidden size is not supported, fall back to the non-persistent
        # kernel.
        persist_ln_hidden_sizes = [
            1024,
            1536,
            2048,
            2304,
            3072,
            3840,
            4096,
            5120,
            6144,
            8192,
            10240,
            12288,
            12800,
            15360,
            16384,
            18432,
            20480,
            24576,
            25600,
            30720,
            32768,
            40960,
            49152,
            65536,
        ]
        if (
            normalized_shape not in persist_ln_hidden_sizes
            or not HAVE_PERSIST_LAYER_NORM
        ):
            no_persist_layer_norm = True

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.scale = Parameter(torch.Tensor(*normalized_shape))
        self.reset_parameters()
        self.no_persist_layer_norm = no_persist_layer_norm
        self.sequence_parallel = sequence_parallel

        # set sequence parallelism flag on weight and bias parameters
        setattr(self.scale, "sequence_parallel", self.sequence_parallel)

    def reset_parameters(self):

        if self.apply_rmsnorm_1p:
            init.zeros_(self.scale)
        else:
            init.ones_(self.scale)

    def forward(self, input):

        weight = self.scale + 1 if self.apply_rmsnorm_1p else self.scale
        # CPU path is here for unittest sake.
        if not input.is_cuda:
            print(
                "WARNING! The input of FusedLayerNorm should be on the GPU."
                "This warning should only be triggered in the FusedRMSNorm unit tests."
            )
            # Latest pytorch actually supports F.rms_norm but I don't want to break builds so...
            return F.layer_norm(input, self.normalized_shape, weight, None, self.eps)

        # Apex does not have versions yet (https://github.com/NVIDIA/apex/pull/1648), so we need to inspect
        # the function manually on whether the extra arg introduced in https://github.com/NVIDIA/apex/pull/1715 exists yet
        if "memory_efficient" in inspect.getfullargspec(self.norm_fn.forward).args:
            return self.norm_fn.apply(
                input,
                weight,
                self.normalized_shape,
                self.eps,
                self.mem_efficient_rms,
            )
        else:
            return self.norm_fn.apply(input, weight, self.normalized_shape, self.eps)

            # Apex's fast layer norm function outputs a 'view' tensor (i.e., has
            # a populated '_base' field). This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            output = make_viewless_tensor(
                inp=output, requires_grad=input.requires_grad, keep_graph=True
            )

            return output
