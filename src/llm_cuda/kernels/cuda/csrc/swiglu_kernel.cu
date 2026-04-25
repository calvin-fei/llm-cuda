#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__global__ void swiglu_forward_kernel(
    const scalar_t* __restrict__ gate,
    const scalar_t* __restrict__ up,
    scalar_t* __restrict__ out,
    int rows,
    int hidden) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;

  if (row >= rows) {
    return;
  }

  const scalar_t* gate_row = gate + row * hidden;
  const scalar_t* up_row = up + row * hidden;
  scalar_t* out_row = out + row * hidden;

  for (int col = tid; col < hidden; col += blockDim.x) {
    float g = static_cast<float>(gate_row[col]);
    float u = static_cast<float>(up_row[col]);
    float sig = 1.0f / (1.0f + expf(-g));
    out_row[col] = static_cast<scalar_t>((g * sig) * u);
  }
}

torch::Tensor swiglu_forward_cuda(torch::Tensor gate_2d, torch::Tensor up_2d) {
  auto rows = static_cast<int>(gate_2d.size(0));
  auto hidden = static_cast<int>(gate_2d.size(1));

  auto out = torch::empty_like(gate_2d);

  constexpr int threads = 256;
  dim3 blocks(rows);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      gate_2d.scalar_type(),
      "swiglu_forward_cuda",
      [&] {
        swiglu_forward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
        gate_2d.data_ptr<scalar_t>(),
        up_2d.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        rows,
        hidden);
  });

  return out;
}
