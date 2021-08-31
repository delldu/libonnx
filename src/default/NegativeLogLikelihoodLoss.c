#include <onnx.h>

void resolver_default_op_NegativeLogLikelihoodLoss(onnx_node_t *n) {
  EMPTY_OPERATOR();
  
  if (n->opset >= 13) {
  } else if (n->opset >= 12) {
  }
}
