#include <onnx.h>

typedef struct {
  float alpha;
} operator_pdata_t;

static int Celu_init(onnx_node_t *n) {
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

static int Celu_exit(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;

  if (pdat)
    free(pdat);
  return 1;
}

static int Celu_reshape(onnx_node_t *n) {
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];

  return onnx_tensor_reshape_identity(y, x, x->type);
}

static void Celu_float32(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];
  float *px = (float *)x->datas;
  float *py = (float *)y->datas;

  for (size_t i = 0, l = y->ndata; i < l; i++)
    py[i] =
        max((float)0.0, (float)px[i]) +
        min((float)0.0, (float)pdat->alpha * (expf(px[i] / pdat->alpha) - 1));
}

void resolver_default_op_Celu(onnx_node_t *n) {
  n->init = Celu_init;
  n->exit = Celu_exit;
  n->reshape = Celu_reshape;

  if (n->opset >= 12) {
    switch (n->inputs[0]->type) {
    case ONNX_TENSOR_TYPE_FLOAT32:
      n->operator= Celu_float32;
      break;
    default:
      break;
    }
  }
}
