#include <onnx.h>

static int Ceil_init(onnx_node_t *n) {
  if ((n->ninput == 1) && (n->noutput == 1))
    return 1;
  return 0;
}

static int Ceil_exit(onnx_node_t *n) { return 1; }

static int Ceil_reshape(onnx_node_t *n) {
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];

  return onnx_tensor_reshape_identity(y, x, x->type);
}

static void Ceil_bfloat16(onnx_node_t *n) {
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];
  uint16_t *px = (uint16_t *)x->datas;
  uint16_t *py = (uint16_t *)y->datas;
  float v;

  for (size_t i = 0, l = y->ndata; i < l; i++) {
    v = bfloat16_to_float32(px[i]);
    py[i] = float32_to_float16(ceilf(v));
  }
}

static void Ceil_float16(onnx_node_t *n) {
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];
  uint16_t *px = (uint16_t *)x->datas;
  uint16_t *py = (uint16_t *)y->datas;
  float v;

  for (size_t i = 0, l = y->ndata; i < l; i++) {
    v = float16_to_float32(px[i]);
    py[i] = float32_to_float16(ceilf(v));
  }
}

static void Ceil_float32(onnx_node_t *n) {
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];
  float *px = (float *)x->datas;
  float *py = (float *)y->datas;

  for (size_t i = 0, l = y->ndata; i < l; i++)
    py[i] = ceilf(px[i]);
}

static void Ceil_float64(onnx_node_t *n) {
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];
  double *px = (double *)x->datas;
  double *py = (double *)y->datas;

  for (size_t i = 0, l = y->ndata; i < l; i++)
    py[i] = ceil(px[i]);
}

void resolver_default_op_Ceil(onnx_node_t *n) {
  n->init = Ceil_init;
  n->exit = Ceil_exit;
  n->reshape = Ceil_reshape;

  if (n->opset >= 13) {
    switch (n->inputs[0]->type) {
    case ONNX_TENSOR_TYPE_BFLOAT16:
      n->operator= Ceil_bfloat16;
      break;
    case ONNX_TENSOR_TYPE_FLOAT16:
      n->operator= Ceil_float16;
      break;
    case ONNX_TENSOR_TYPE_FLOAT32:
      n->operator= Ceil_float32;
      break;
    case ONNX_TENSOR_TYPE_FLOAT64:
      n->operator= Ceil_float64;
      break;
    default:
      break;
    }
  } else if (n->opset >= 6) {
    switch (n->inputs[0]->type) {
    case ONNX_TENSOR_TYPE_FLOAT16:
      n->operator= Ceil_float16;
      break;
    case ONNX_TENSOR_TYPE_FLOAT32:
      n->operator= Ceil_float32;
      break;
    case ONNX_TENSOR_TYPE_FLOAT64:
      n->operator= Ceil_float64;
      break;
    default:
      break;
    }
  } else if (n->opset >= 1) {
    switch (n->inputs[0]->type) {
    case ONNX_TENSOR_TYPE_FLOAT16:
      n->operator= Ceil_float16;
      break;
    case ONNX_TENSOR_TYPE_FLOAT32:
      n->operator= Ceil_float32;
      break;
    case ONNX_TENSOR_TYPE_FLOAT64:
      n->operator= Ceil_float64;
      break;
    default:
      break;
    }
  }
}
