#include <onnx.h>

typedef struct {
  float alpha;
} operator_pdata_t;

static int ThresholdedRelu_init(onnx_node_t *n) {
  operator_pdata_t *pdat;

  if ((n->ninput == 1) && (n->noutput == 1)) {
    pdat = malloc(sizeof(operator_pdata_t));
    if (pdat) {
      pdat->alpha = onnx_attribute_read_float(n, "alpha", 1.0);
      n->priv = pdat;
      return 1;
    }
  }
  return 0;
}

static int ThresholdedRelu_exit(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;

  if (pdat)
    free(pdat);
  return 1;
}

static int ThresholdedRelu_reshape(onnx_node_t *n) {
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];

  return onnx_tensor_reshape_identity(y, x, x->type);
}

static void ThresholdedRelu_float16(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];
  uint16_t *px = (uint16_t *)x->datas;
  uint16_t *py = (uint16_t *)y->datas;
  float v;

  for (size_t i = 0, l = y->ndata; i < l; i++) {
    v = float16_to_float32(px[i]);
    py[i] = (v > pdat->alpha) ? float32_to_float16(v) : 0;
  }
}

static void ThresholdedRelu_float32(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];
  float *px = (float *)x->datas;
  float *py = (float *)y->datas;

  for (size_t i = 0, l = y->ndata; i < l; i++)
    py[i] = (px[i] > pdat->alpha) ? px[i] : 0;
}

static void ThresholdedRelu_float64(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];
  double *px = (double *)x->datas;
  double *py = (double *)y->datas;

  for (size_t i = 0, l = y->ndata; i < l; i++)
    py[i] = (px[i] > pdat->alpha) ? px[i] : 0;
}

void resolver_default_op_ThresholdedRelu(onnx_node_t *n) {
  n->init = ThresholdedRelu_init;
  n->exit = ThresholdedRelu_exit;
  n->reshape = ThresholdedRelu_reshape;

  if (n->opset >= 10) {
    switch (n->inputs[0]->type) {
    case ONNX_TENSOR_TYPE_FLOAT16:
      n->operator= ThresholdedRelu_float16;
      break;
    case ONNX_TENSOR_TYPE_FLOAT32:
      n->operator= ThresholdedRelu_float32;
      break;
    case ONNX_TENSOR_TYPE_FLOAT64:
      n->operator= ThresholdedRelu_float64;
      break;
    default:
      break;
    }
  }
}
