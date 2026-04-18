#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

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
  int row = blockIdx.x;
  int d = threadIdx.x;

  if (d >= head_dim) {
    return;
  }

  int total_rows = bsz * n_heads * seq_len;
  if (row >= total_rows) {
    return;
  }

  int m = row % seq_len;
  int tmp = row / seq_len;
  int h = tmp % n_heads;
  int b = tmp / n_heads;

  const scalar_t* q_row = q + b * stride_qb + h * stride_qh + m * stride_qm;

  float m_i = -INFINITY;
  for (int n = 0; n <= m; ++n) {
    const scalar_t* k_row = k + b * stride_kb + h * stride_kh + n * stride_kn;
    float dot = 0.0f;
    for (int kk = 0; kk < head_dim; ++kk) {
      float qv = static_cast<float>(q_row[kk * stride_qk]);
      float kv = static_cast<float>(k_row[kk * stride_kk]);
      dot += qv * kv;
    }
    float score = dot * scale;
    if (score > m_i) {
      m_i = score;
    }
  }

  float l_i = 0.0f;
  for (int n = 0; n <= m; ++n) {
    const scalar_t* k_row = k + b * stride_kb + h * stride_kh + n * stride_kn;
    float dot = 0.0f;
    for (int kk = 0; kk < head_dim; ++kk) {
      float qv = static_cast<float>(q_row[kk * stride_qk]);
      float kv = static_cast<float>(k_row[kk * stride_kk]);
      dot += qv * kv;
    }
    float score = dot * scale;
    l_i += expf(score - m_i);
  }

  float acc = 0.0f;
  for (int n = 0; n <= m; ++n) {
    const scalar_t* k_row = k + b * stride_kb + h * stride_kh + n * stride_kn;
    const scalar_t* v_row = v + b * stride_vb + h * stride_vh + n * stride_vn;

    float dot = 0.0f;
    for (int kk = 0; kk < head_dim; ++kk) {
      float qv = static_cast<float>(q_row[kk * stride_qk]);
      float kv = static_cast<float>(k_row[kk * stride_kk]);
      dot += qv * kv;
    }
    float score = dot * scale;
    float p = expf(score - m_i) / fmaxf(l_i, 1e-9f);

    float vv = static_cast<float>(v_row[d * stride_vk]);
    acc += p * vv;
  }

  scalar_t* out_row = out + b * stride_ob + h * stride_oh + m * stride_om;
  out_row[d * stride_ok] = static_cast<scalar_t>(acc);
}

torch::Tensor attention_forward_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  auto bsz = static_cast<int>(q.size(0));
  auto n_heads = static_cast<int>(q.size(1));
  auto seq_len = static_cast<int>(q.size(2));
  auto head_dim = static_cast<int>(q.size(3));

  auto out = torch::empty_like(q);

  int total_rows = bsz * n_heads * seq_len;
  dim3 blocks(total_rows);
  dim3 threads(256);

  float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "attention_forward_cuda", [&] {
    attention_forward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
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
