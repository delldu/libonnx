#include <onnx.h>

void resolver_default_op_QuantizeLinear(onnx_node_t *n) {
  if (n->opset >= 13) {
  } else if (n->opset >= 10) {
  }
}
