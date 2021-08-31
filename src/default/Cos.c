#include <onnx.h>

static int Cos_init(onnx_node_t *n) {
  if ((n->ninput == 1) && (n->noutput == 1))
    return 1;
  return 0;
}

static int Cos_exit(onnx_node_t *n) { return 1; }

static int Cos_reshape(onnx_node_t *n) {
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];

  return onnx_tensor_reshape_identity(y, x, x->type);
}

static void Cos_float16(onnx_node_t *n) {
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];
  uint16_t *px = (uint16_t *)x->datas;
  uint16_t *py = (uint16_t *)y->datas;
  float v;

  for (size_t i = 0, l = y->ndata; i < l; i++) {
    v = float16_to_float32(px[i]);
    py[i] = float32_to_float16(cosf(v));
  }
}

static void Cos_float32(onnx_node_t *n) {
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];
  float *px = (float *)x->datas;
  float *py = (float *)y->datas;

  for (size_t i = 0, l = y->ndata; i < l; i++)
    py[i] = cosf(px[i]);
}

static void Cos_float64(onnx_node_t *n) {
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];
  double *px = (double *)x->datas;
  double *py = (double *)y->datas;

  for (size_t i = 0, l = y->ndata; i < l; i++)
    py[i] = cos(px[i]);
}

void resolver_default_op_Cos(onnx_node_t *n) {
  if (n->opset >= 7) {
    switch (n->inputs[0]->type) {
    case ONNX_TENSOR_TYPE_FLOAT16:
      n->init = Cos_init;
      n->exit = Cos_exit;
      n->reshape = Cos_reshape;
      n->operator= Cos_float16;
      break;
    case ONNX_TENSOR_TYPE_FLOAT32:
      n->init = Cos_init;
      n->exit = Cos_exit;
      n->reshape = Cos_reshape;
      n->operator= Cos_float32;
      break;
    case ONNX_TENSOR_TYPE_FLOAT64:
      n->init = Cos_init;
      n->exit = Cos_exit;
      n->reshape = Cos_reshape;
      n->operator= Cos_float64;
      break;
    default:
      break;
    }
  }
}
