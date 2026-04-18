#include <torch/extension.h>

#include <vector>

torch::Tensor rms_norm_forward_cuda(torch::Tensor x_2d, torch::Tensor weight, double eps);
torch::Tensor swiglu_forward_cuda(torch::Tensor gate_2d, torch::Tensor up_2d);
torch::Tensor attention_forward_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v);

torch::Tensor rms_norm_forward(torch::Tensor x, torch::Tensor weight, double eps) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");
  TORCH_CHECK(x.scalar_type() == weight.scalar_type(), "x/weight dtype mismatch");
  TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "x must be [rows, hidden] or [batch, seq, hidden]");

  auto hidden = x.size(-1);
  TORCH_CHECK(weight.dim() == 1, "weight must be 1D");
  TORCH_CHECK(weight.size(0) == hidden, "weight size must match hidden dimension");

  auto x_contig = x.contiguous();
  auto x_2d = x_contig.view({-1, hidden});
  auto out_2d = rms_norm_forward_cuda(x_2d, weight.contiguous(), eps);
  return out_2d.view(x_contig.sizes());
}

torch::Tensor swiglu_forward(torch::Tensor gate, torch::Tensor up) {
  TORCH_CHECK(gate.is_cuda(), "gate must be CUDA tensor");
  TORCH_CHECK(up.is_cuda(), "up must be CUDA tensor");
  TORCH_CHECK(gate.scalar_type() == up.scalar_type(), "gate/up dtype mismatch");
  TORCH_CHECK(gate.dim() == 2 || gate.dim() == 3, "gate must be [rows, hidden] or [batch, seq, hidden]");
  TORCH_CHECK(gate.sizes() == up.sizes(), "gate/up shape mismatch");

  auto gate_contig = gate.contiguous();
  auto up_contig = up.contiguous();
  auto hidden = gate_contig.size(-1);

  auto gate_2d = gate_contig.view({-1, hidden});
  auto up_2d = up_contig.view({-1, hidden});
  auto out_2d = swiglu_forward_cuda(gate_2d, up_2d);
  return out_2d.view(gate_contig.sizes());
}

torch::Tensor attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  TORCH_CHECK(q.is_cuda(), "q must be CUDA tensor");
  TORCH_CHECK(k.is_cuda(), "k must be CUDA tensor");
  TORCH_CHECK(v.is_cuda(), "v must be CUDA tensor");
  TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q/k dtype mismatch");
  TORCH_CHECK(q.scalar_type() == v.scalar_type(), "q/v dtype mismatch");
  TORCH_CHECK(q.dim() == 4, "q must be [batch, heads, seq, head_dim]");
  TORCH_CHECK(k.sizes() == q.sizes(), "k shape mismatch");
  TORCH_CHECK(v.sizes() == q.sizes(), "v shape mismatch");

  auto q_contig = q.contiguous();
  auto k_contig = k.contiguous();
  auto v_contig = v.contiguous();
  return attention_forward_cuda(q_contig, k_contig, v_contig);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rms_norm_forward", &rms_norm_forward, "RMSNorm forward (CUDA)");
  m.def("swiglu_forward", &swiglu_forward, "SwiGLU forward (CUDA)");
  m.def("attention_forward", &attention_forward, "Causal attention forward (CUDA)");
}
