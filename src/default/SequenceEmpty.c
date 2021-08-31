#include <onnx.h>

void resolver_default_op_SequenceEmpty(onnx_node_t *n) {
  if (n->opset >= 11) {
  }
}
