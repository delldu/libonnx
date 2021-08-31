#include <onnx.h>

typedef struct {
  enum onnx_tensor_type_t dtype;
  float mean;
  float scale;
  float seed;
  int *shape;
  int nshape;
} operator_pdata_t;

static int RandomNormal_init(onnx_node_t *n) {
  operator_pdata_t *pdat;
  int64_t *ints;
  int i;

  if (n->noutput == 1) {
    pdat = malloc(sizeof(operator_pdata_t));
    if (pdat) {
      pdat->nshape = onnx_attribute_read_ints(n, "shape", &ints);
      if ((pdat->nshape > 0) &&
          (pdat->shape = malloc(sizeof(int) * pdat->nshape))) {
        pdat->dtype =
            (enum onnx_tensor_type_t)onnx_attribute_read_int(n, "dtype", 1);
        pdat->mean = onnx_attribute_read_float(n, "mean", 0.0);
        pdat->scale = onnx_attribute_read_float(n, "scale", 1.0);
        pdat->seed = onnx_attribute_read_float(n, "seed", 0.0);
        for (i = 0; i < pdat->nshape; i++)
          pdat->shape[i] = ints[i];
        n->priv = pdat;
        return 1;
      } else {
        free(pdat);
        return 0;
      }
    }
  }
  return 0;
}

static int RandomNormal_exit(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;

  if (pdat) {
    if (pdat->shape)
      free(pdat->shape);
    free(pdat);
  }
  return 1;
}

static int RandomNormal_reshape(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;
  onnx_tensor_t *y = n->outputs[0];

  return onnx_tensor_reshape(y, pdat->shape, pdat->nshape, pdat->dtype);
}

static void RandomNormal_operator(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;
  onnx_tensor_t *y = n->outputs[0];

  if (pdat->seed != 0.0)
    srand(pdat->seed);
  switch (pdat->dtype) {
  case ONNX_TENSOR_TYPE_FLOAT16: {
    uint16_t *py = (uint16_t *)y->datas;
    float ty, tx;
    for (size_t i = 0, l = y->ndata; i < l; i++) {
      ty = (float)rand() / (RAND_MAX + 1.0f);
      tx = (float)rand() / (RAND_MAX + 1.0f);
      py[i] = float16_to_float32(pdat->mean +
                                 pdat->scale * sqrtf(-2.0f * logf(tx)) *
                                     cosf(2.0f * acosf(-1.0f) * ty));
    }
  } break;
  case ONNX_TENSOR_TYPE_FLOAT32: {
    float *py = (float *)y->datas;
    float ty, tx;
    for (size_t i = 0, l = y->ndata; i < l; i++) {
      ty = (float)rand() / (RAND_MAX + 1.0f);
      tx = (float)rand() / (RAND_MAX + 1.0f);
      py[i] = pdat->mean + pdat->scale * sqrtf(-2.0f * logf(tx)) *
                               cosf(2.0f * acosf(-1.0f) * ty);
    }
  } break;
  case ONNX_TENSOR_TYPE_FLOAT64: {
    double *py = (double *)y->datas;
    double ty, tx;
    for (size_t i = 0, l = y->ndata; i < l; i++) {
      ty = (double)rand() / (RAND_MAX + 1.0f);
      tx = (double)rand() / (RAND_MAX + 1.0f);
      py[i] = pdat->mean + pdat->scale * sqrt(-2.0f * log(tx)) *
                               cos(2.0f * acos(-1.0f) * ty);
    }
  } break;
  default:
    break;
  }
}

void resolver_default_op_RandomNormal(onnx_node_t *n) {
  if (n->opset >= 1) {
    n->init = RandomNormal_init;
    n->exit = RandomNormal_exit;
    n->reshape = RandomNormal_reshape;
    n->operator= RandomNormal_operator;
  }
}
