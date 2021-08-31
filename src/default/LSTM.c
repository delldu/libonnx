#include <onnx.h>

void resolver_default_op_LSTM(onnx_node_t *n) {
  EMPTY_OPERATOR();
  
  if (n->opset >= 14) {
  } else if (n->opset >= 7) {
  } else if (n->opset >= 1) {
  }
}
