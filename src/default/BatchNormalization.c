#include <onnx.h>

typedef struct {
  float epsilon;
  float momentum;
} operator_pdata_t;

static int BatchNormalization_init(onnx_node_t *n) {
  operator_pdata_t *pdat;

  if ((n->ninput == 5) && (n->noutput >= 1)) {
    pdat = malloc(sizeof(operator_pdata_t));
    if (pdat) {
      pdat->epsilon = onnx_attribute_read_float(n, "epsilon", 1e-05);
      pdat->momentum = onnx_attribute_read_float(n, "momentum", 0.9);
      n->priv = pdat;
      return 1;
    }
  }
  return 0;
}

static int BatchNormalization_exit(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;

  if (pdat)
    free(pdat);
  return 1;
}

static int BatchNormalization_reshape(onnx_node_t *n) {
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];

  return onnx_tensor_reshape_identity(y, x, x->type);
}

static void BatchNormalization_float16(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *scale = n->inputs[1];
  onnx_tensor_t *b = n->inputs[2];
  onnx_tensor_t *mean = n->inputs[3];
  onnx_tensor_t *var = n->inputs[4];
  onnx_tensor_t *y = n->outputs[0];
  uint16_t *px = (uint16_t *)x->datas;
  uint16_t *pscale = (uint16_t *)scale->datas;
  uint16_t *pb = (uint16_t *)b->datas;
  uint16_t *pmean = (uint16_t *)mean->datas;
  uint16_t *pvar = (uint16_t *)var->datas;
  uint16_t *py = (uint16_t *)y->datas;
  int N = x->dims[0];
  int C = x->dims[1];
  int NC = N * C;
  int channel = 1;
  int i, j, o, jc;

  for (i = 2; i < x->ndim; i++)
    channel *= x->dims[i];
  for (j = 0; j < NC; j++) {
    o = j * channel;
    jc = j % C;
    for (i = 0; i < channel; i++)
      py[o + i] = float32_to_float16(
          float16_to_float32(pscale[jc]) *
              ((float16_to_float32(px[o + i]) - float16_to_float32(pmean[jc])) /
               sqrtf(float16_to_float32(pvar[jc]) + pdat->epsilon)) +
          float16_to_float32(pb[jc]));
  }
}

static void BatchNormalization_float32(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *scale = n->inputs[1];
  onnx_tensor_t *b = n->inputs[2];
  onnx_tensor_t *mean = n->inputs[3];
  onnx_tensor_t *var = n->inputs[4];
  onnx_tensor_t *y = n->outputs[0];
  float *px = (float *)x->datas;
  float *pscale = (float *)scale->datas;
  float *pb = (float *)b->datas;
  float *pmean = (float *)mean->datas;
  float *pvar = (float *)var->datas;
  float *py = (float *)y->datas;
  int N = x->dims[0];
  int C = x->dims[1];
  int NC = N * C;
  int channel = 1;
  int i, j, o, jc;

  for (i = 2; i < x->ndim; i++)
    channel *= x->dims[i];
  for (j = 0; j < NC; j++) {
    o = j * channel;
    jc = j % C;
    for (i = 0; i < channel; i++)
      py[o + i] = pscale[jc] * ((px[o + i] - pmean[jc]) /
                                sqrtf(pvar[jc] + pdat->epsilon)) +
                  pb[jc];
  }
}

static void BatchNormalization_float64(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *scale = n->inputs[1];
  onnx_tensor_t *b = n->inputs[2];
  onnx_tensor_t *mean = n->inputs[3];
  onnx_tensor_t *var = n->inputs[4];
  onnx_tensor_t *y = n->outputs[0];
  double *px = (double *)x->datas;
  double *pscale = (double *)scale->datas;
  double *pb = (double *)b->datas;
  double *pmean = (double *)mean->datas;
  double *pvar = (double *)var->datas;
  double *py = (double *)y->datas;
  int N = x->dims[0];
  int C = x->dims[1];
  int NC = N * C;
  int channel = 1;
  int i, j, o, jc;

  for (i = 2; i < x->ndim; i++)
    channel *= x->dims[i];
  for (j = 0; j < NC; j++) {
    o = j * channel;
    jc = j % C;
    for (i = 0; i < channel; i++)
      py[o + i] = pscale[jc] * ((px[o + i] - pmean[jc]) /
                                sqrt(pvar[jc] + pdat->epsilon)) +
                  pb[jc];
  }
}

void resolver_default_op_BatchNormalization(onnx_node_t *n) {
  n->init = BatchNormalization_init;
  n->exit = BatchNormalization_exit;
  n->reshape = BatchNormalization_reshape;

  if (n->opset >= 14) {
    EMPTY_OPERATOR();
  } else if (n->opset >= 9) {
    switch (n->inputs[0]->type) {
    case ONNX_TENSOR_TYPE_FLOAT16:
      n->operator= BatchNormalization_float16;
      break;
    case ONNX_TENSOR_TYPE_FLOAT32:
      n->operator= BatchNormalization_float32;
      break;
    case ONNX_TENSOR_TYPE_FLOAT64:
      n->operator= BatchNormalization_float64;
      break;
    default:
      break;
    }
  } else if (n->opset >= 7) {
    switch (n->inputs[0]->type) {
    case ONNX_TENSOR_TYPE_FLOAT16:
      n->operator= BatchNormalization_float16;
      break;
    case ONNX_TENSOR_TYPE_FLOAT32:
      n->operator= BatchNormalization_float32;
      break;
    case ONNX_TENSOR_TYPE_FLOAT64:
      n->operator= BatchNormalization_float64;
      break;
    default:
      break;
    }
  } else if (n->opset >= 6) {
    EMPTY_OPERATOR();
  } else if (n->opset >= 1) {
    EMPTY_OPERATOR();
  }
}
