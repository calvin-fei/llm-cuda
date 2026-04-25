#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__global__ void rms_norm_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ w,
    scalar_t* __restrict__ y,
    int rows,
    int hidden,
    float eps) {
  extern __shared__ float shared[];

  const int row = blockIdx.x;
  const int tid = threadIdx.x;

  if (row >= rows) {
    return;
  }

  const scalar_t* x_row = x + row * hidden;
  scalar_t* y_row = y + row * hidden;

  float local_sum = 0.0f;
  for (int col = tid; col < hidden; col += blockDim.x) {
    float v = static_cast<float>(x_row[col]);
    local_sum += v * v;
  }

  shared[tid] = local_sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared[tid] += shared[tid + s];
    }
    __syncthreads();
  }

  float inv_rms = rsqrtf(shared[0] / static_cast<float>(hidden) + eps);

  for (int col = tid; col < hidden; col += blockDim.x) {
    float xv = static_cast<float>(x_row[col]);
    float wv = static_cast<float>(w[col]);
    y_row[col] = static_cast<scalar_t>(xv * inv_rms * wv);
  }
}

torch::Tensor rms_norm_forward_cuda(torch::Tensor x_2d, torch::Tensor weight, double eps) {
  auto rows = static_cast<int>(x_2d.size(0));
  auto hidden = static_cast<int>(x_2d.size(1));

  auto y = torch::empty_like(x_2d);

  constexpr int threads = 256;
  dim3 blocks(rows);
  size_t smem = threads * sizeof(float);

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x_2d.scalar_type(), "rms_norm_forward_cuda", [&] {
    rms_norm_kernel<scalar_t><<<blocks, threads, smem, at::cuda::getDefaultCUDAStream()>>>(
        x_2d.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        y.data_ptr<scalar_t>(),
        rows,
        hidden,
        static_cast<float>(eps));
  });

  return y;
}
