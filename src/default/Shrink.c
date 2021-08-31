#include <onnx.h>

typedef struct {
  float bias;
  float lambd;
} operator_pdata_t;

static int Shrink_init(onnx_node_t *n) {
  operator_pdata_t *pdat;

  if ((n->ninput == 1) && (n->noutput == 1)) {
    pdat = malloc(sizeof(operator_pdata_t));
    if (pdat) {
      pdat->bias = onnx_attribute_read_float(n, "bias", 0.0);
      pdat->lambd = onnx_attribute_read_float(n, "lambd", 0.5);
      n->priv = pdat;
      return 1;
    }
  }
  return 0;
}

static int Shrink_exit(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;

  if (pdat)
    free(pdat);
  return 1;
}

static int Shrink_reshape(onnx_node_t *n) {
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];

  return onnx_tensor_reshape_identity(y, x, x->type);
}

static void Shrink_int8(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];
  int8_t *px = (int8_t *)x->datas;
  int8_t *py = (int8_t *)y->datas;

  for (size_t i = 0, l = y->ndata; i < l; i++) {
    if (px[i] < -pdat->lambd)
      py[i] = px[i] + pdat->bias;
    else if (px[i] > pdat->lambd)
      py[i] = px[i] - pdat->bias;
    else
      py[i] = 0;
  }
}

static void Shrink_int16(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];
  int16_t *px = (int16_t *)x->datas;
  int16_t *py = (int16_t *)y->datas;

  for (size_t i = 0, l = y->ndata; i < l; i++) {
    if (px[i] < -pdat->lambd)
      py[i] = px[i] + pdat->bias;
    else if (px[i] > pdat->lambd)
      py[i] = px[i] - pdat->bias;
    else
      py[i] = 0;
  }
}

static void Shrink_int32(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];
  int32_t *px = (int32_t *)x->datas;
  int32_t *py = (int32_t *)y->datas;

  for (size_t i = 0, l = y->ndata; i < l; i++) {
    if (px[i] < -pdat->lambd)
      py[i] = px[i] + pdat->bias;
    else if (px[i] > pdat->lambd)
      py[i] = px[i] - pdat->bias;
    else
      py[i] = 0;
  }
}

static void Shrink_int64(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];
  int64_t *px = (int64_t *)x->datas;
  int64_t *py = (int64_t *)y->datas;

  for (size_t i = 0, l = y->ndata; i < l; i++) {
    if (px[i] < -pdat->lambd)
      py[i] = px[i] + pdat->bias;
    else if (px[i] > pdat->lambd)
      py[i] = px[i] - pdat->bias;
    else
      py[i] = 0;
  }
}

static void Shrink_uint8(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];
  uint8_t *px = (uint8_t *)x->datas;
  uint8_t *py = (uint8_t *)y->datas;

  for (size_t i = 0, l = y->ndata; i < l; i++) {
    if (px[i] < -pdat->lambd)
      py[i] = px[i] + pdat->bias;
    else if (px[i] > pdat->lambd)
      py[i] = px[i] - pdat->bias;
    else
      py[i] = 0;
  }
}

static void Shrink_uint16(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];
  uint16_t *px = (uint16_t *)x->datas;
  uint16_t *py = (uint16_t *)y->datas;

  for (size_t i = 0, l = y->ndata; i < l; i++) {
    if (px[i] < -pdat->lambd)
      py[i] = px[i] + pdat->bias;
    else if (px[i] > pdat->lambd)
      py[i] = px[i] - pdat->bias;
    else
      py[i] = 0;
  }
}

static void Shrink_uint32(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];
  uint32_t *px = (uint32_t *)x->datas;
  uint32_t *py = (uint32_t *)y->datas;

  for (size_t i = 0, l = y->ndata; i < l; i++) {
    if (px[i] < -pdat->lambd)
      py[i] = px[i] + pdat->bias;
    else if (px[i] > pdat->lambd)
      py[i] = px[i] - pdat->bias;
    else
      py[i] = 0;
  }
}

static void Shrink_uint64(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];
  uint64_t *px = (uint64_t *)x->datas;
  uint64_t *py = (uint64_t *)y->datas;

  for (size_t i = 0, l = y->ndata; i < l; i++) {
    if (px[i] < -pdat->lambd)
      py[i] = px[i] + pdat->bias;
    else if (px[i] > pdat->lambd)
      py[i] = px[i] - pdat->bias;
    else
      py[i] = 0;
  }
}

static void Shrink_float16(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];
  uint16_t *px = (uint16_t *)x->datas;
  uint16_t *py = (uint16_t *)y->datas;
  float v;

  for (size_t i = 0, l = y->ndata; i < l; i++) {
    v = float16_to_float32(px[i]);
    if (v < -pdat->lambd)
      py[i] = float32_to_float16(v + pdat->bias);
    else if (v > pdat->lambd)
      py[i] = float32_to_float16(v - pdat->bias);
    else
      py[i] = float32_to_float16(0);
  }
}

static void Shrink_float32(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];
  float *px = (float *)x->datas;
  float *py = (float *)y->datas;

  for (size_t i = 0, l = y->ndata; i < l; i++) {
    if (px[i] < -pdat->lambd)
      py[i] = px[i] + pdat->bias;
    else if (px[i] > pdat->lambd)
      py[i] = px[i] - pdat->bias;
    else
      py[i] = 0;
  }
}

static void Shrink_float64(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];
  double *px = (double *)x->datas;
  double *py = (double *)y->datas;

  for (size_t i = 0, l = y->ndata; i < l; i++) {
    if (px[i] < -pdat->lambd)
      py[i] = px[i] + pdat->bias;
    else if (px[i] > pdat->lambd)
      py[i] = px[i] - pdat->bias;
    else
      py[i] = 0;
  }
}

void resolver_default_op_Shrink(onnx_node_t *n) {
  if (n->opset >= 9) {
    switch (n->inputs[0]->type) {
    case ONNX_TENSOR_TYPE_INT8:
      n->init = Shrink_init;
      n->exit = Shrink_exit;
      n->reshape = Shrink_reshape;
      n->operator= Shrink_int8;
      break;
    case ONNX_TENSOR_TYPE_INT16:
      n->init = Shrink_init;
      n->exit = Shrink_exit;
      n->reshape = Shrink_reshape;
      n->operator= Shrink_int16;
      break;
    case ONNX_TENSOR_TYPE_INT32:
      n->init = Shrink_init;
      n->exit = Shrink_exit;
      n->reshape = Shrink_reshape;
      n->operator= Shrink_int32;
      break;
    case ONNX_TENSOR_TYPE_INT64:
      n->init = Shrink_init;
      n->exit = Shrink_exit;
      n->reshape = Shrink_reshape;
      n->operator= Shrink_int64;
      break;
    case ONNX_TENSOR_TYPE_UINT8:
      n->init = Shrink_init;
      n->exit = Shrink_exit;
      n->reshape = Shrink_reshape;
      n->operator= Shrink_uint8;
      break;
    case ONNX_TENSOR_TYPE_UINT16:
      n->init = Shrink_init;
      n->exit = Shrink_exit;
      n->reshape = Shrink_reshape;
      n->operator= Shrink_uint16;
      break;
    case ONNX_TENSOR_TYPE_UINT32:
      n->init = Shrink_init;
      n->exit = Shrink_exit;
      n->reshape = Shrink_reshape;
      n->operator= Shrink_uint32;
      break;
    case ONNX_TENSOR_TYPE_UINT64:
      n->init = Shrink_init;
      n->exit = Shrink_exit;
      n->reshape = Shrink_reshape;
      n->operator= Shrink_uint64;
      break;
    case ONNX_TENSOR_TYPE_FLOAT16:
      n->init = Shrink_init;
      n->exit = Shrink_exit;
      n->reshape = Shrink_reshape;
      n->operator= Shrink_float16;
      break;
    case ONNX_TENSOR_TYPE_FLOAT32:
      n->init = Shrink_init;
      n->exit = Shrink_exit;
      n->reshape = Shrink_reshape;
      n->operator= Shrink_float32;
      break;
    case ONNX_TENSOR_TYPE_FLOAT64:
      n->init = Shrink_init;
      n->exit = Shrink_exit;
      n->reshape = Shrink_reshape;
      n->operator= Shrink_float64;
      break;
    default:
      break;
    }
  }
}
