#include <onnx.h>

void resolver_default_op_LpNormalization(onnx_node_t *n) {
  EMPTY_OPERATOR();
  
  if (n->opset >= 1) {
  }
}
