#include <onnx.h>

void resolver_default_op_Upsample(onnx_node_t *n) {
  EMPTY_OPERATOR();
  
  if (n->opset >= 10) {
    return;
  } else if (n->opset >= 9) {
  } else if (n->opset >= 7) {
  }
}
