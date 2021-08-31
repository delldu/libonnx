#include <onnx.h>

void resolver_default_op_SequenceAt(onnx_node_t *n) {
  EMPTY_OPERATOR();
  
  if (n->opset >= 11) {
  }
}
