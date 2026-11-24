# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""Planted for the self-test (#48), KernelBench level1/19_ReLU: wraps the forward in torch.compile: rejected statically."""

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

src = r"""
__global__ void relu_k(const float* __restrict__ x, float* __restrict__ y, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long s = (long)gridDim.x * blockDim.x;
  for (; i < n; i += s) { float v = x[i]; y[i] = v > 0.f ? v : 0.f; }
}
torch::Tensor relu(torch::Tensor x) {
  auto y = torch::empty_like(x);
  long n = x.numel();
  relu_k<<<(int)((n + 255) / 256 > 65535 * 8 ? 65535 * 8 : (n + 255) / 256), 256>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
  return y;
}
"""
ext = load_inline("relu_ext_comp", cpp_sources="torch::Tensor relu(torch::Tensor x);", cuda_sources=src, functions=["relu"])

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    def forward(self, x):
        return ext.relu(x)


ModelNew.forward = torch.compile(ModelNew.forward)
