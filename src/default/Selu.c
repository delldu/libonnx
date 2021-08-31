#include <onnx.h>

typedef struct {
  float alpha;
  float gamma;
} operator_pdata_t;

static int Selu_init(onnx_node_t *n) {
  operator_pdata_t *pdat;

  if ((n->ninput == 1) && (n->noutput == 1)) {
    pdat = malloc(sizeof(operator_pdata_t));
    if (pdat) {
      pdat->alpha = onnx_attribute_read_float(n, "alpha", 1.67326);
      pdat->gamma = onnx_attribute_read_float(n, "gamma", 1.0507);
      n->priv = pdat;
      return 1;
    }
  }
  return 0;
}

static int Selu_exit(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;

  if (pdat)
    free(pdat);
  return 1;
}

static int Selu_reshape(onnx_node_t *n) {
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];

  return onnx_tensor_reshape_identity(y, x, x->type);
}

static void Selu_float16(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];
  uint16_t *px = (uint16_t *)x->datas;
  uint16_t *py = (uint16_t *)y->datas;
  float v;

  for (size_t i = 0, l = y->ndata; i < l; i++) {
    v = float16_to_float32(px[i]);
    if (v > 0)
      py[i] = pdat->gamma * v;
    else
      py[i] = pdat->gamma * (pdat->alpha * expf(v) - pdat->alpha);
  }
}

static void Selu_float32(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];
  float *px = (float *)x->datas;
  float *py = (float *)y->datas;

  for (size_t i = 0, l = y->ndata; i < l; i++) {
    if (px[i] > 0)
      py[i] = pdat->gamma * px[i];
    else
      py[i] = pdat->gamma * (pdat->alpha * expf(px[i]) - pdat->alpha);
  }
}

static void Selu_float64(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];
  double *px = (double *)x->datas;
  double *py = (double *)y->datas;

  for (size_t i = 0, l = y->ndata; i < l; i++) {
    if (px[i] > 0)
      py[i] = pdat->gamma * px[i];
    else
      py[i] = pdat->gamma * (pdat->alpha * exp(px[i]) - pdat->alpha);
  }
}

void resolver_default_op_Selu(onnx_node_t *n) {
  n->init = Selu_init;
  n->exit = Selu_exit;
  n->reshape = Selu_reshape;

  if (n->opset >= 6) {
    switch (n->inputs[0]->type) {
    case ONNX_TENSOR_TYPE_FLOAT16:
      n->operator= Selu_float16;
      break;
    case ONNX_TENSOR_TYPE_FLOAT32:
      n->operator= Selu_float32;
      break;
    case ONNX_TENSOR_TYPE_FLOAT64:
      n->operator= Selu_float64;
      break;
    default:
      break;
    }
  } else if (n->opset >= 1) {
    switch (n->inputs[0]->type) {
    case ONNX_TENSOR_TYPE_FLOAT16:
      n->operator= Selu_float16;
      break;
    case ONNX_TENSOR_TYPE_FLOAT32:
      n->operator= Selu_float32;
      break;
    case ONNX_TENSOR_TYPE_FLOAT64:
      n->operator= Selu_float64;
      break;
    default:
      break;
    }
  }
}
