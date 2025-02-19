#include <onnx.h>

static int Acosh_init(onnx_node_t *n) {
  if ((n->ninput == 1) && (n->noutput == 1))
    return 1;
  return 0;
}

static int Acosh_exit(onnx_node_t *n) { return 1; }

static int Acosh_reshape(onnx_node_t *n) {
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];

  return onnx_tensor_reshape_identity(y, x, x->type);
}

static void Acosh_float16(onnx_node_t *n) {
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];
  uint16_t *px = (uint16_t *)x->datas;
  uint16_t *py = (uint16_t *)y->datas;
  float v;
  size_t i, l;

  for (i = 0, l = y->ndata; i < l; i++) {
    v = float16_to_float32(px[i]);
    py[i] = float32_to_float16(acoshf(v));
  }
}

static void Acosh_float32(onnx_node_t *n) {
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];
  float *px = (float *)x->datas;
  float *py = (float *)y->datas;
  size_t i, l;

  for (i = 0, l = y->ndata; i < l; i++)
    py[i] = acoshf(px[i]);
}

static void Acosh_float64(onnx_node_t *n) {
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];
  double *px = (double *)x->datas;
  double *py = (double *)y->datas;
  size_t i, l;

  for (i = 0, l = y->ndata; i < l; i++)
    py[i] = acosh(px[i]);
}

void resolver_default_op_Acosh(onnx_node_t *n) {
  n->init = Acosh_init;
  n->exit = Acosh_exit;
  n->reshape = Acosh_reshape;

  if (n->opset >= 9) {
    switch (n->inputs[0]->type) {
    case ONNX_TENSOR_TYPE_FLOAT16:
      n->operator= Acosh_float16;
      break;
    case ONNX_TENSOR_TYPE_FLOAT32:
      n->operator= Acosh_float32;
      break;
    case ONNX_TENSOR_TYPE_FLOAT64:
      n->operator= Acosh_float64;
      break;
    default:
      break;
    }
  }
}
