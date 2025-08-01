# Copyright (c) 2024, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch


import math

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from .initialize import get_model_parallel_rank
from .initialize import get_model_parallel_world_size
from .mappings import copy_to_model_parallel_region
from .mappings import gather_from_model_parallel_region
from .mappings import reduce_from_model_parallel_region
from .mappings import scatter_to_model_parallel_region
from .mappings import reduce_scatter_to_sequence_parallel_region
from .mappings import gather_from_sequence_parallel_region
from .random import get_cuda_rng_tracker
from .utils import divide
from .utils import VocabUtility
from functools import partial


def _initialize_affine_weight_gpu(weight, init_method, partition_dim, stride=1):
    """Initialize affine weight for model parallel on GPU."""

    weight.model_parallel = True
    weight.partition_dim = partition_dim
    weight.partition_stride = stride

    with get_cuda_rng_tracker().fork():
        init_method(weight)


def _initialize_affine_weight_cpu(
    neox_args,
    weight,
    output_size,
    input_size,
    per_partition_size,
    partition_dim,
    init_method,
    stride=1,
    return_master_weight=False,
):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    weight.model_parallel = True
    weight.partition_dim = partition_dim
    weight.partition_stride = stride

    # Initialize master weight
    master_weight = torch.empty(
        output_size, input_size, dtype=torch.float, requires_grad=False
    )
    init_method(master_weight)
    master_weight = master_weight.to(dtype=neox_args.params_dtype)

    # Split and copy
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(
        master_weight, per_partition_per_stride_size, dim=partition_dim
    )
    rank = get_model_parallel_rank()
    world_size = get_model_parallel_world_size()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(
        self, neox_args, num_embeddings, embedding_dim, init_method=init.xavier_normal_
    ):
        super(VocabParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set the detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.0
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.model_parallel_size = get_model_parallel_world_size()
        # Divide the weight matrix along the vocabulary dimension.
        (
            self.vocab_start_index,
            self.vocab_end_index,
        ) = VocabUtility.vocab_range_from_global_vocab_size(
            self.num_embeddings, get_model_parallel_rank(), self.model_parallel_size
        )
        self.num_embeddings_per_partition = (
            self.vocab_end_index - self.vocab_start_index
        )
        self.init_method = init_method

        # Allocate weights and initialize.
        if neox_args.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition,
                    self.embedding_dim,
                    dtype=neox_args.params_dtype,
                )
            )
            _initialize_affine_weight_cpu(
                neox_args,
                self.weight,
                self.num_embeddings,
                self.embedding_dim,
                self.num_embeddings_per_partition,
                0,
                init_method,
            )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition,
                    self.embedding_dim,
                    device=torch.cuda.current_device(),
                    dtype=neox_args.params_dtype,
                )
            )
            _initialize_affine_weight_gpu(
                self.weight, init_method, partition_dim=0, stride=1
            )

    def mup_reinitialize_weights(self, neox_args):
        if neox_args.use_cpu_initialization:
            _initialize_affine_weight_cpu(
                neox_args,
                self.weight,
                self.num_embeddings,
                self.embedding_dim,
                self.num_embeddings_per_partition,
                0,
                partial(self.init_method, use_mup=True),
            )
        else:
            _initialize_affine_weight_gpu(
                self.weight,
                partial(self.init_method, use_mup=True),
                partition_dim=0,
                stride=1,
            )

    def forward(self, input_):
        if self.model_parallel_size > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | (
                input_ >= self.vocab_end_index
            )
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
            # Get the embeddings.
        output_parallel = F.embedding(
            masked_input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        # Mask the output embedding.
        if self.model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = reduce_from_model_parallel_region(output_parallel)
        return output


class ParallelRelativePositionBias(torch.nn.Module):
    """T5 Relative Position Bias parallelized in the heads dimension

    Based on https://github.com/lucidrains/x-transformers/blob/6b93c21be0d0a679da6f7b9621d9bb638ab18428/x_transformers/x_transformers.py#L106 (14.12.2021)
    and adapted for megatron's model parallelism

    Arguments:
        scale: scaling factor for the bias
        causal: flag for causal/non-causal language modelling.
        num_buckets: number of rp buckets.
        max_distance: max distance in sequence dim for each bucket.
        heads: number of attention heads (total)
    """

    def __init__(
        self,
        neox_args,
        scale,
        causal=True,
        num_buckets=32,
        max_distance=128,
        heads=8,
        init_method=init.xavier_normal_,
    ):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.heads = heads

        # Set the defaults for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.0
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.model_parallel_size = get_model_parallel_world_size()
        self.model_parallel_rank = get_model_parallel_rank()

        # Divide the weight matrix along the heads dimension.
        self.head_start_index, self.head_end_index = self.get_heads_range(
            self.heads, self.model_parallel_rank, self.model_parallel_size
        )
        self.num_heads_per_partition = self.head_end_index - self.head_start_index
        self.init_method = init_method

        # Allocate weights and initialize.
        if neox_args.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.num_buckets,
                    self.num_heads_per_partition,
                    dtype=neox_args.params_dtype,
                )
            )
            _initialize_affine_weight_cpu(
                neox_args,
                self.weight,
                self.num_buckets,
                self.heads,
                self.num_heads_per_partition,
                partition_dim=1,
                init_method=init_method,
            )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.num_buckets,
                    self.num_heads_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=neox_args.params_dtype,
                )
            )
            _initialize_affine_weight_gpu(
                self.weight, init_method, partition_dim=1, stride=1
            )
        self._q_len_cached = None
        self._k_len_cached = None
        self._rel_pos_bucket_cached = None

    def mup_reinitialize_weights(self, neox_args):
        if self.use_cpu_initialization:
            _initialize_affine_weight_cpu(
                neox_args,
                self.weight,
                self.num_buckets,
                self.heads,
                self.num_heads_per_partition,
                partition_dim=1,
                init_method=partial(self.init_method, use_mup=True),
            )
        else:
            _initialize_affine_weight_gpu(
                self.weight,
                partial(self.init_method, use_mup=True),
                partition_dim=1,
                stride=1,
            )

    @staticmethod
    def get_heads_range(global_n_heads, rank, world_size):
        per_partition_n_heads = divide(global_n_heads, world_size)
        index_f = rank * per_partition_n_heads
        index_l = index_f + per_partition_n_heads
        return index_f, index_l

    def _relative_position_bucket(
        self, relative_position, num_buckets=32, max_distance=128
    ):
        ret = 0
        n = -relative_position
        if not self.causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(n.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)
        self._rel_pos_bucket_cached = ret
        return self._rel_pos_bucket_cached

    def forward(self, q_len, k_len):
        if self._q_len_cached != q_len or self._k_len_cached != k_len:
            # cache bucket if first step seq len stays constant
            self._q_len_cached, self._k_len_cached = q_len, k_len
            q_pos = torch.arange(
                q_len, dtype=torch.long, device=torch.cuda.current_device()
            )
            k_pos = torch.arange(
                k_len, dtype=torch.long, device=torch.cuda.current_device()
            )
            rel_pos = k_pos[None, :] - q_pos[:, None]
            rp_bucket = self._relative_position_bucket(
                rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance
            )
        else:
            rp_bucket = self._rel_pos_bucket_cached
        values = F.embedding(
            rp_bucket,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        bias = values.movedim(2, 0).unsqueeze(0)
        return bias * self.scale


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
    """

    def __init__(
        self,
        neox_args,
        input_size,
        output_size,
        bias=True,
        gather_output=True,
        init_method=init.xavier_normal_,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        MOE=False,
        MoE_mp_size=1,
        mup_rescale_parameters=False,
        seq_dim=0,  # Dimension which is the seq_len dimension. final ParallelLinear overrides this to be 1 ; otherwise, the default is used throughout.
    ):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = MoE_mp_size if MOE else get_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, world_size)
        self.skip_bias_add = skip_bias_add

        self.sequence_parallel = neox_args.sequence_parallel
        self.seq_dim = seq_dim

        self.init_method = init_method
        self.stride = stride
        self.mup_rescale_parameters = mup_rescale_parameters
        self.use_mup = neox_args.use_mup

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if neox_args.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    self.input_size,
                    dtype=neox_args.params_dtype,
                )
            )
            self.master_weight = _initialize_affine_weight_cpu(
                neox_args,
                self.weight,
                self.output_size,
                self.input_size,
                self.output_size_per_partition,
                0,
                init_method,
                stride=stride,
                return_master_weight=keep_master_weight_for_test,
            )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    self.input_size,
                    device=torch.cuda.current_device(),
                    dtype=neox_args.params_dtype,
                )
            )
            _initialize_affine_weight_gpu(
                self.weight, init_method, partition_dim=0, stride=stride
            )

        if bias:
            if neox_args.use_cpu_initialization:
                self.bias = Parameter(
                    torch.empty(
                        self.output_size_per_partition, dtype=neox_args.params_dtype
                    )
                )
            else:
                self.bias = Parameter(
                    torch.empty(
                        self.output_size_per_partition,
                        device=torch.cuda.current_device(),
                        dtype=neox_args.params_dtype,
                    )
                )
            self.bias.model_parallel = True
            self.bias.partition_dim = 0
            self.bias.stride = stride
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

    # Copied from Mup
    def width_mult(self):
        assert hasattr(self.weight, "infshape"), (
            "Please call set_base_shapes(...). If using torch.nn.DataParallel, "
            "switch to distributed training with "
            "torch.nn.parallel.DistributedDataParallel instead"
        )
        return self.weight.infshape.width_mult()

    # Copied from Mup
    def _rescale_parameters(self):
        """Rescale parameters to convert SP initialization to μP initialization.
        Warning: This method is NOT idempotent and should be called only once
        unless you know what you are doing.
        """
        if hasattr(self, "_has_rescaled_params") and self._has_rescaled_params:
            raise RuntimeError(
                "`_rescale_parameters` has been called once before already. "
                "Unless you know what you are doing, usually you should not be calling `_rescale_parameters` more than once.\n"
                "If you called `set_base_shapes` on a model loaded from a checkpoint, "
                "or just want to re-set the base shapes of an existing model, "
                "make sure to set the flag `rescale_params=False`.\n"
                "To bypass this error and *still rescale parameters*, set `self._has_rescaled_params=False` before this call."
            )
        if self.bias is not None:
            self.bias.data *= self.width_mult() ** 0.5
        self.weight.data *= self.width_mult() ** 0.5
        self._has_rescaled_params = True

    def mup_reinitialize_weights(self, neox_args):
        if neox_args.use_cpu_initialization:
            self.master_weight = _initialize_affine_weight_cpu(
                neox_args,
                self.weight,
                self.output_size,
                self.input_size,
                self.output_size_per_partition,
                0,
                partial(self.init_method, use_mup=True),
                stride=self.stride,
                return_master_weight=keep_master_weight_for_test,
            )
        else:
            _initialize_affine_weight_gpu(
                self.weight,
                partial(self.init_method, use_mup=True),
                partition_dim=0,
                stride=self.stride,
            )

    def set_parallel_output(self, value: bool):
        assert isinstance(value, bool)
        self.gather_output = (
            not value
        )  # if gather_output is True, parallel output is False, so we set the opposite

    def forward(self, input_):
        if self.use_mup and self.mup_rescale_parameters:
            input_ /= self.width_mult()

        if self.sequence_parallel:
            input_parallel = input_
        else:
            # Set up backprop all-reduce.
            input_parallel = copy_to_model_parallel_region(input_)
        # Matrix multiply.

        if self.sequence_parallel:
            # do an AG in the fwd pass, RS in bwd pass.
            # gather / scatter portion happens across the sequence dim (self.seq_dim)--
            # almost always is [s, b, h] and so dim 0, but for lm_head ParallelLinear it is seq_dim=1 and [b, s, h]
            input_parallel = gather_from_sequence_parallel_region(
                input_parallel, seq_dim=self.seq_dim
            )

        bias = self.bias if not self.skip_bias_add else None
        output_parallel = F.linear(input_parallel, self.weight, bias)
        if self.gather_output:
            # All-gather across the partitions.
            assert (
                not self.sequence_parallel
            ), "sequence_parallel=True and gather_output=True are incompatible!"
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
    """

    def __init__(
        self,
        neox_args,
        input_size,
        output_size,
        bias=True,
        input_is_parallel=False,
        init_method=init.xavier_normal_,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        MOE=False,
        MoE_mp_size=1,
        parallel_output=False,
        mup_rescale_parameters=False,
    ):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = MoE_mp_size if MOE else get_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)
        self.skip_bias_add = skip_bias_add
        self.parallel_output = parallel_output

        self.sequence_parallel = neox_args.sequence_parallel
        assert not (
            self.sequence_parallel and not self.input_is_parallel
        ), "Cannot have self.input_is_parallel=False and self.sequence_parallel=True."

        self.init_method = init_method
        self.stride = stride
        self.keep_master_weight_for_test = keep_master_weight_for_test
        self.mup_rescale_parameters = mup_rescale_parameters
        self.use_mup = neox_args.use_mup

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if neox_args.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.output_size,
                    self.input_size_per_partition,
                    dtype=neox_args.params_dtype,
                )
            )
            self.master_weight = _initialize_affine_weight_cpu(
                neox_args,
                self.weight,
                self.output_size,
                self.input_size,
                self.input_size_per_partition,
                1,
                init_method,
                stride=stride,
                return_master_weight=keep_master_weight_for_test,
            )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.output_size,
                    self.input_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=neox_args.params_dtype,
                )
            )
            _initialize_affine_weight_gpu(
                self.weight, init_method, partition_dim=1, stride=stride
            )
        if bias:
            if neox_args.use_cpu_initialization:
                self.bias = Parameter(
                    torch.empty(self.output_size, dtype=neox_args.params_dtype)
                )
            else:
                self.bias = Parameter(
                    torch.empty(
                        self.output_size,
                        device=torch.cuda.current_device(),
                        dtype=neox_args.params_dtype,
                    )
                )
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

    # Copied from Mup
    def width_mult(self):
        assert hasattr(self.weight, "infshape"), (
            "Please call set_base_shapes(...). If using torch.nn.DataParallel, "
            "switch to distributed training with "
            "torch.nn.parallel.DistributedDataParallel instead"
        )
        return self.weight.infshape.width_mult()

    # Copied from Mup
    def _rescale_parameters(self):
        """Rescale parameters to convert SP initialization to μP initialization.
        Warning: This method is NOT idempotent and should be called only once
        unless you know what you are doing.
        """
        if hasattr(self, "_has_rescaled_params") and self._has_rescaled_params:
            raise RuntimeError(
                "`_rescale_parameters` has been called once before already. "
                "Unless you know what you are doing, usually you should not be calling `_rescale_parameters` more than once.\n"
                "If you called `set_base_shapes` on a model loaded from a checkpoint, "
                "or just want to re-set the base shapes of an existing model, "
                "make sure to set the flag `rescale_params=False`.\n"
                "To bypass this error and *still rescale parameters*, set `self._has_rescaled_params=False` before this call."
            )
        if self.bias is not None:
            self.bias.data *= self.width_mult() ** 0.5
        self.weight.data *= self.width_mult() ** 0.5
        self._has_rescaled_params = True

    def mup_reinitialize_weights(self, neox_args):
        if neox_args.use_cpu_initialization:
            self.master_weight = _initialize_affine_weight_cpu(
                neox_args,
                self.weight,
                self.output_size,
                self.input_size,
                self.input_size_per_partition,
                1,
                partial(self.init_method, use_mup=True),
                stride=self.stride,
                return_master_weight=self.keep_master_weight_for_test,
            )
        else:
            _initialize_affine_weight_gpu(
                self.weight,
                partial(self.init_method, use_mup=True),
                partition_dim=1,
                stride=self.stride,
            )

    def set_parallel_output(self, parallel_output: bool):
        assert isinstance(parallel_output, bool)
        self.parallel_output = parallel_output

    def forward(self, input_):
        if self.use_mup and self.mup_rescale_parameters:
            input_ /= self.width_mult()
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight)
        # All-reduce across all the partitions.
        if self.sequence_parallel and not self.parallel_output:
            # do an RS in the fwd pass, AG in bwd pass.
            # skip in the gpt-j parallel sublayer case (self.parallel_output=True)
            # (user responsible for calling reduce-scatter)
            output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
        elif not self.parallel_output:
            output_ = reduce_from_model_parallel_region(output_parallel)
        else:
            output_ = output_parallel
        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias



class ColumnParallelLinear_LR(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
    """

    def __init__(
        self,
        neox_args,
        input_size,
        output_size,
        bias=True,
        gather_output=True,
        init_method=init.xavier_normal_,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        MOE=False,
        MoE_mp_size=1,
        mup_rescale_parameters=False,
        seq_dim=0,  # Dimension which is the seq_len dimension. final ParallelLinear overrides this to be 1 ; otherwise, the default is used throughout.
        rmax=64,
        rmin=2,
        tol=0.01,
    ):
        super(ColumnParallelLinear_LR, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size

        self.r = 256  # int(rmax / 2)  # Hardcoded for now!
        self.rmax = rmax
        self.rmin = rmin
        self.tol = tol

        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = MoE_mp_size if MOE else get_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, world_size)
        self.skip_bias_add = skip_bias_add

        self.sequence_parallel = neox_args.sequence_parallel
        self.seq_dim = seq_dim

        self.init_method = init_method
        self.stride = stride
        self.mup_rescale_parameters = mup_rescale_parameters
        self.use_mup = neox_args.use_mup

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.

        if neox_args.use_cpu_initialization:

            self.weight = Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    self.input_size,
                    dtype=neox_args.params_dtype,
                )
            )

            self.master_weight = _initialize_affine_weight_cpu(
                neox_args,
                self.weight,
                self.output_size,
                self.input_size,
                self.output_size_per_partition,
                0,
                init_method,
                stride=stride,
                return_master_weight=keep_master_weight_for_test,
            )

        else:  # this is used

            # self.weight = Parameter(
            #    torch.empty(
            #        self.output_size_per_partition,
            #        self.input_size,
            #        device=torch.cuda.current_device(),
            #        dtype=neox_args.params_dtype,
            #    )
            # )

            self.weight_u = Parameter(
                torch.empty(
                    self.rmax,
                    self.input_size,
                    dtype=neox_args.params_dtype,
                )
            )
            self.weight_s = Parameter(
                torch.empty(
                    self.rmax,
                    self.rmax,
                    dtype=neox_args.params_dtype,
                )
            )
            self.weight_v = Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    self.rmax,
                    dtype=neox_args.params_dtype,
                )
            )

            # _initialize_affine_weight_gpu(
            #    self.weight, init_method, partition_dim=0, stride=stride
            # )
            _initialize_affine_weight_gpu(
                self.weight_u, init_method, partition_dim=0, stride=stride
            )
            _initialize_affine_weight_gpu(
                self.weight_s, init_method, partition_dim=0, stride=stride
            )
            _initialize_affine_weight_gpu(
                self.weight_v, init_method, partition_dim=0, stride=stride
            )

        if bias:
            if neox_args.use_cpu_initialization:
                self.bias = Parameter(
                    torch.empty(
                        self.output_size_per_partition, dtype=neox_args.params_dtype
                    )
                )
            else:
                self.bias = Parameter(
                    torch.empty(
                        self.output_size_per_partition,
                        device=torch.cuda.current_device(),
                        dtype=neox_args.params_dtype,
                    )
                )
            self.bias.model_parallel = True
            self.bias.partition_dim = 0
            self.bias.stride = stride
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

    # Copied from Mup
    def width_mult(self):
        assert hasattr(self.weight, "infshape"), (
            "Please call set_base_shapes(...). If using torch.nn.DataParallel, "
            "switch to distributed training with "
            "torch.nn.parallel.DistributedDataParallel instead"
        )
        return self.weight.infshape.width_mult()

    # Copied from Mup
    def _rescale_parameters(self):
        """Rescale parameters to convert SP initialization to μP initialization.
        Warning: This method is NOT idempotent and should be called only once
        unless you know what you are doing.
        """
        if hasattr(self, "_has_rescaled_params") and self._has_rescaled_params:
            raise RuntimeError(
                "`_rescale_parameters` has been called once before already. "
                "Unless you know what you are doing, usually you should not be calling `_rescale_parameters` more than once.\n"
                "If you called `set_base_shapes` on a model loaded from a checkpoint, "
                "or just want to re-set the base shapes of an existing model, "
                "make sure to set the flag `rescale_params=False`.\n"
                "To bypass this error and *still rescale parameters*, set `self._has_rescaled_params=False` before this call."
            )
        if self.bias is not None:
            self.bias.data *= self.width_mult() ** 0.5
        self.weight.data *= self.width_mult() ** 0.5
        self._has_rescaled_params = True

    def mup_reinitialize_weights(self, neox_args):
        if neox_args.use_cpu_initialization:
            self.master_weight = _initialize_affine_weight_cpu(
                neox_args,
                self.weight,
                self.output_size,
                self.input_size,
                self.output_size_per_partition,
                0,
                partial(self.init_method, use_mup=True),
                stride=self.stride,
                return_master_weight=keep_master_weight_for_test,
            )
        else:
            _initialize_affine_weight_gpu(
                self.weight,
                partial(self.init_method, use_mup=True),
                partition_dim=0,
                stride=self.stride,
            )

    def set_parallel_output(self, value: bool):
        assert isinstance(value, bool)
        self.gather_output = (
            not value
        )  # if gather_output is True, parallel output is False, so we set the opposite

    def forward(self, input_):
        if self.use_mup and self.mup_rescale_parameters:
            input_ /= self.width_mult()

        if self.sequence_parallel:
            input_parallel = input_
        else:
            # Set up backprop all-reduce.
            input_parallel = copy_to_model_parallel_region(input_)
        # Matrix multiply.

        if self.sequence_parallel:
            # do an AG in the fwd pass, RS in bwd pass.
            # gather / scatter portion happens across the sequence dim (self.seq_dim)--
            # almost always is [s, b, h] and so dim 0, but for lm_head ParallelLinear it is seq_dim=1 and [b, s, h]
            input_parallel = gather_from_sequence_parallel_region(
                input_parallel, seq_dim=self.seq_dim
            )
        # print(input_parallel.shape)
        # print(self.weight_u.shape)
        # print(self.bias.shape)
        # exit(1)
        bias = self.bias if not self.skip_bias_add else None
        if bias is not None:
            output_parallel = (
                input_parallel
                @ self.weight_u[: self.r, :].T
                @ self.weight_s[: self.r, : self.r].T
                @ self.weight_v[:, : self.r].T
                + bias
            )
        else:
            output_parallel = (
                input_parallel
                @ self.weight_u[: self.r, :].T
                @ self.weight_s[: self.r, : self.r].T
                @ self.weight_v[:, : self.r].T
            )

        # output_parallel = F.linear(input_parallel, self.weight, bias)
        if self.gather_output:
            # All-gather across the partitions.
            assert (
                not self.sequence_parallel
            ), "sequence_parallel=True and gather_output=True are incompatible!"
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    @torch.no_grad()
    def augment(self) -> None:
        """Performs a steepest descend training update on specified low-rank factors
        Args:
            learning_rate: learning rate for training
            dlrt_step: sepcifies step that is taken. Can be 'K', 'L' or 'S'
            adaptive: specifies if fixed-rank or rank-adaptivity is used
        """

        r1 = min(self.rmax, 2 * self.r)

        # K step
        U1, _ = torch.linalg.qr(
            torch.cat(
                (self.weight_u[: self.r, :].T, self.weight_u.grad[: self.r, :].T), 1
            ),
            "reduced",
        )

        # L step
        V1, _ = torch.linalg.qr(
            torch.cat((self.weight_v[:, : self.r], -self.V.grad[:, : self.r]), 1),
            "reduced",
        )

        # Basis projection
        m_t = self.weight_u[: self.r, :] @ U1
        n_t = V1 @ self.weight_v[:, : self.r]

        # M = U1[:, :r1].T @ self.U[:, : self.r]
        # N = self.V[:, : self.r].T @ V1[:, :r1]

        # Project coefficients
        self.weight_s[:r1, :r1] = n_t @ self.weight_s[: self.r, : self.r] @ m_t
        # yields S1.T = self.weight_s[:r1, :r1]

        # self.S.data[:r1, :r1] = M @ self.S[: self.r, : self.r] @ N

        # update basis
        self.weight_u[:r1, :] = U1[:, :r1].T
        self.weight_v[:, :r1] = V1[:, :r1]
        # self.U.data[:, :r1] = U1[:, :r1]
        # self.V.data[:, :r1] = V1[:, :r1]
        self.r = r1

    @torch.no_grad()
    def truncate(self) -> None:
        """Truncates the weight matrix to a new rank"""

        P, d, Q = torch.linalg.svd(self.weight_s[: self.r, : self.r])
        # P corresponds to V!, d corresponds to S, Q corresponds to U!

        tol = self.tol * torch.linalg.norm(d)
        r1 = self.r
        for j in range(0, self.r):
            tmp = torch.linalg.norm(d[j : self.r])
            if tmp < tol:
                r1 = j
                break

        # Check if new ranks is withing legal bounds
        r1 = min(r1, self.rmax)
        r1 = max(r1, self.rmin)

        # update s
        try:
            self.weight_s[:r1, :r1] = torch.diag(d[:r1])
            # since d is diagonal s.T = s  here
        except:
            print(d.shape)
            print(self.weight_s.shape)
            print(r1)
            print("error")
            exit(1)
        # update u and v
        self.weight_u[:r1, :] = Q[:r1, :] @ self.weight_u[: self.r, :]
        self.weight_v[:, :r1] = self.weight_v[:, : self.r] @ P[:, :r1]

        # self.U.data[:, :r1] = torch.matmul(self.U[:, : self.r], P[:, :r1])
        # self.V.data[:, :r1] = torch.matmul(self.V[:, : self.r], Q.T[:, :r1])
        self.r = int(r1)


class RowParallelLinear_LR(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
    """

    def __init__(
        self,
        neox_args,
        input_size,
        output_size,
        bias=True,
        input_is_parallel=False,
        init_method=init.xavier_normal_,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        MOE=False,
        MoE_mp_size=1,
        parallel_output=False,
        mup_rescale_parameters=False,
        rmax=64,
        rmin=2,
        tol=0.01,
    ):
        super(RowParallelLinear_LR, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size

        self.r = 256  # int(rmax / 2)  # Hardcoded for now!
        self.rmax = rmax
        self.rmin = rmin
        self.tol = tol

        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = MoE_mp_size if MOE else get_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)
        self.skip_bias_add = skip_bias_add
        self.parallel_output = parallel_output

        self.sequence_parallel = neox_args.sequence_parallel
        assert not (
            self.sequence_parallel and not self.input_is_parallel
        ), "Cannot have self.input_is_parallel=False and self.sequence_parallel=True."

        self.init_method = init_method
        self.stride = stride
        self.keep_master_weight_for_test = keep_master_weight_for_test
        self.mup_rescale_parameters = mup_rescale_parameters
        self.use_mup = neox_args.use_mup

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if neox_args.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.output_size,
                    self.input_size_per_partition,
                    dtype=neox_args.params_dtype,
                )
            )
            self.master_weight = _initialize_affine_weight_cpu(
                neox_args,
                self.weight,
                self.output_size,
                self.input_size,
                self.input_size_per_partition,
                1,
                init_method,
                stride=stride,
                return_master_weight=keep_master_weight_for_test,
            )
        else:
            # this is used

            # self.weight = Parameter(
            #    torch.empty(
            #        self.output_size_per_partition,
            #        self.input_size,
            #        device=torch.cuda.current_device(),
            #        dtype=neox_args.params_dtype,
            #    )
            # )

            self.weight_u = Parameter(
                torch.empty(
                    self.r,
                    self.input_size_per_partition,
                    dtype=neox_args.params_dtype,
                )
            )
            self.weight_s = Parameter(
                torch.empty(
                    self.r,
                    self.r,
                    dtype=neox_args.params_dtype,
                )
            )
            self.weight_v = Parameter(
                torch.empty(
                    self.output_size,
                    self.r,
                    dtype=neox_args.params_dtype,
                )
            )
            # print(self.weight_u.shape, self.weight_s.shape, self.weight_v.shape)
            # exit(1)

            # _initialize_affine_weight_gpu(
            #    self.weight, init_method, partition_dim=1, stride=stride
            # )
            _initialize_affine_weight_gpu(
                self.weight_u, init_method, partition_dim=1, stride=stride
            )
            _initialize_affine_weight_gpu(
                self.weight_s, init_method, partition_dim=1, stride=stride
            )
            _initialize_affine_weight_gpu(
                self.weight_v, init_method, partition_dim=1, stride=stride
            )
        if bias:
            if neox_args.use_cpu_initialization:
                self.bias = Parameter(
                    torch.empty(self.output_size, dtype=neox_args.params_dtype)
                )
            else:
                self.bias = Parameter(
                    torch.empty(
                        self.output_size,
                        device=torch.cuda.current_device(),
                        dtype=neox_args.params_dtype,
                    )
                )
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

    # Copied from Mup
    def width_mult(self):
        assert hasattr(self.weight, "infshape"), (
            "Please call set_base_shapes(...). If using torch.nn.DataParallel, "
            "switch to distributed training with "
            "torch.nn.parallel.DistributedDataParallel instead"
        )
        return self.weight.infshape.width_mult()

    # Copied from Mup
    def _rescale_parameters(self):
        """Rescale parameters to convert SP initialization to μP initialization.
        Warning: This method is NOT idempotent and should be called only once
        unless you know what you are doing.
        """
        if hasattr(self, "_has_rescaled_params") and self._has_rescaled_params:
            raise RuntimeError(
                "`_rescale_parameters` has been called once before already. "
                "Unless you know what you are doing, usually you should not be calling `_rescale_parameters` more than once.\n"
                "If you called `set_base_shapes` on a model loaded from a checkpoint, "
                "or just want to re-set the base shapes of an existing model, "
                "make sure to set the flag `rescale_params=False`.\n"
                "To bypass this error and *still rescale parameters*, set `self._has_rescaled_params=False` before this call."
            )
        if self.bias is not None:
            self.bias.data *= self.width_mult() ** 0.5
        self.weight.data *= self.width_mult() ** 0.5
        self._has_rescaled_params = True

    def mup_reinitialize_weights(self, neox_args):
        if neox_args.use_cpu_initialization:
            self.master_weight = _initialize_affine_weight_cpu(
                neox_args,
                self.weight,
                self.output_size,
                self.input_size,
                self.input_size_per_partition,
                1,
                partial(self.init_method, use_mup=True),
                stride=self.stride,
                return_master_weight=self.keep_master_weight_for_test,
            )
        else:
            _initialize_affine_weight_gpu(
                self.weight,
                partial(self.init_method, use_mup=True),
                partition_dim=1,
                stride=self.stride,
            )

    def set_parallel_output(self, parallel_output: bool):
        assert isinstance(parallel_output, bool)
        self.parallel_output = parallel_output

    def forward(self, input_):
        if self.use_mup and self.mup_rescale_parameters:
            input_ /= self.width_mult()
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_)
        # Matrix multiply.
        bias = self.bias if not self.skip_bias_add else None
        if bias is not None:
            output_parallel = (
                input_parallel @ self.weight_u.T @ self.weight_s.T @ self.weight_v.T
                + bias
            )
        else:
            output_parallel = (
                input_parallel @ self.weight_u.T @ self.weight_s.T @ self.weight_v.T
            )
        # output_parallel = F.linear(input_parallel, self.weight)
        # All-reduce across all the partitions.
        if self.sequence_parallel and not self.parallel_output:
            # do an RS in the fwd pass, AG in bwd pass.
            # skip in the gpt-j parallel sublayer case (self.parallel_output=True)
            # (user responsible for calling reduce-scatter)
            output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
        elif not self.parallel_output:
            output_ = reduce_from_model_parallel_region(output_parallel)
        else:
            output_ = output_parallel

        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias

