#include <onnx.h>

void resolver_default_op_MatMulInteger(onnx_node_t *n) {
  if (n->opset >= 10) {
  }
}
