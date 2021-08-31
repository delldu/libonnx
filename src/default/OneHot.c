#include <onnx.h>

void resolver_default_op_OneHot(onnx_node_t *n) {
  if (n->opset >= 11) {
  } else if (n->opset >= 9) {
  }
}
