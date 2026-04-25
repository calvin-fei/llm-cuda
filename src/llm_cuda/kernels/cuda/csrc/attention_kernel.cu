#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

// One thread block per (batch, head, query-row).
// Threads cooperate on dot-product reductions via shared memory so each
// QK dot product is computed once (online Flash-Attention-style softmax).
static constexpr int ATTN_BLOCK_DIM = 128;

template <typename scalar_t>
__global__ void attention_forward_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ out,
    int64_t stride_qb,
    int64_t stride_qh,
    int64_t stride_qm,
    int64_t stride_qk,
    int64_t stride_kb,
    int64_t stride_kh,
    int64_t stride_kn,
    int64_t stride_kk,
    int64_t stride_vb,
    int64_t stride_vh,
    int64_t stride_vn,
    int64_t stride_vk,
    int64_t stride_ob,
    int64_t stride_oh,
    int64_t stride_om,
    int64_t stride_ok,
    int bsz,
    int n_heads,
    int seq_len,
    int head_dim,
    float scale) {
  // Shared memory layout: [q_buf | reduce_buf], each of size ATTN_BLOCK_DIM.
  extern __shared__ float shmem[];
  float* q_buf = shmem;
  float* reduce_buf = shmem + ATTN_BLOCK_DIM;

  int row = blockIdx.x;
  if (row >= bsz * n_heads * seq_len) return;

  int m = row % seq_len;
  int tmp = row / seq_len;
  int h = tmp % n_heads;
  int b = tmp / n_heads;

  int d = threadIdx.x;

  // Load the Q row into shared memory once and reuse for all K positions.
  q_buf[d] = (d < head_dim)
      ? static_cast<float>(__ldg(&q[b * stride_qb + h * stride_qh + m * stride_qm + d * stride_qk]))
      : 0.0f;
  __syncthreads();

  float m_i = -INFINITY;
  float l_i = 0.0f;
  float acc_d = 0.0f;  // per-thread accumulator for output dimension d

  // Single pass over causal keys with online softmax update.
  for (int n = 0; n <= m; ++n) {
    // All threads cooperate to compute dot(q, k[n]) via tree reduction.
    reduce_buf[d] = (d < head_dim)
        ? q_buf[d] * static_cast<float>(__ldg(&k[b * stride_kb + h * stride_kh + n * stride_kn + d * stride_kk]))
        : 0.0f;
    __syncthreads();

    for (int s = ATTN_BLOCK_DIM >> 1; s > 0; s >>= 1) {
      if (d < s) reduce_buf[d] += reduce_buf[d + s];
      __syncthreads();
    }

    float score = reduce_buf[0] * scale;

    // Online softmax rescaling (Flash Attention style).
    float m_new = fmaxf(m_i, score);
    float alpha = expf(m_i - m_new);
    float p = expf(score - m_new);

    if (d < head_dim) {
      float vd = static_cast<float>(__ldg(&v[b * stride_vb + h * stride_vh + n * stride_vn + d * stride_vk]));
      acc_d = acc_d * alpha + p * vd;
    }

    l_i = l_i * alpha + p;
    m_i = m_new;
  }

  if (d < head_dim) {
    out[b * stride_ob + h * stride_oh + m * stride_om + d * stride_ok] =
        static_cast<scalar_t>(acc_d / fmaxf(l_i, 1e-9f));
  }
}

torch::Tensor attention_forward_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  auto bsz = static_cast<int>(q.size(0));
  auto n_heads = static_cast<int>(q.size(1));
  auto seq_len = static_cast<int>(q.size(2));
  auto head_dim = static_cast<int>(q.size(3));

  TORCH_CHECK(
      head_dim <= ATTN_BLOCK_DIM,
      "attention_forward_cuda: head_dim must be <= ",
      ATTN_BLOCK_DIM);

  auto out = torch::empty_like(q);

  int total_rows = bsz * n_heads * seq_len;
  size_t shmem_size = 2 * ATTN_BLOCK_DIM * sizeof(float);

  float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      q.scalar_type(),
      "attention_forward_cuda",
      [&] {
        attention_forward_kernel<scalar_t>
            <<<total_rows, ATTN_BLOCK_DIM, shmem_size, at::cuda::getDefaultCUDAStream()>>>(
                q.data_ptr<scalar_t>(),
                k.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                out.stride(3),
                bsz,
                n_heads,
                seq_len,
                head_dim,
                scale);
      });

  return out;
}
