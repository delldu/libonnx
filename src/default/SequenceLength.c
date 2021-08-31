#include <onnx.h>

void resolver_default_op_SequenceLength(onnx_node_t *n) {
  if (n->opset >= 11) {
  }
}
