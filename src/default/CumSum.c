#include <onnx.h>

void resolver_default_op_CumSum(onnx_node_t *n) {
  if (n->opset >= 14) {
  } else if (n->opset >= 11) {
  }
}
