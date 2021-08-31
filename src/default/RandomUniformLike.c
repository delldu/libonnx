#include <onnx.h>

typedef struct {
  enum onnx_tensor_type_t dtype;
  float high;
  float low;
  float seed;
} operator_pdata_t;

static int RandomUniformLike_init(onnx_node_t *n) {
  operator_pdata_t *pdat;

  if ((n->ninput == 1) && (n->noutput == 1)) {
    pdat = malloc(sizeof(operator_pdata_t));
    if (pdat) {
      pdat->dtype =
          (enum onnx_tensor_type_t)onnx_attribute_read_int(n, "dtype", 0);
      pdat->high = onnx_attribute_read_float(n, "high", 1.0);
      pdat->low = onnx_attribute_read_float(n, "low", 0.0);
      pdat->seed = onnx_attribute_read_float(n, "seed", 0.0);
      n->priv = pdat;
      return 1;
    }
  }
  return 0;
}

static int RandomUniformLike_exit(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;

  if (pdat)
    free(pdat);
  return 1;
}

static int RandomUniformLike_reshape(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];
  enum onnx_tensor_type_t type;

  if (pdat->dtype != ONNX_TENSOR_TYPE_UNDEFINED)
    type = pdat->dtype;
  else
    type = x->type;
  switch (type) {
  case ONNX_TENSOR_TYPE_FLOAT16:
  case ONNX_TENSOR_TYPE_FLOAT32:
  case ONNX_TENSOR_TYPE_FLOAT64:
    return onnx_tensor_reshape(y, x->dims, x->ndim, type);
  default:
    break;
  }
  return 0;
}

static void RandomUniformLike_operator(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;
  onnx_tensor_t *y = n->outputs[0];

  if (pdat->seed != 0.0)
    srand(pdat->seed);
  switch (pdat->dtype) {
  case ONNX_TENSOR_TYPE_FLOAT16: {
    uint16_t *py = (uint16_t *)y->datas;
    for (size_t i = 0, l = y->ndata; i < l; i++)
      py[i] = float16_to_float32(((float)rand() / (float)RAND_MAX) *
                                     (pdat->high - pdat->low) +
                                 pdat->low);
  } break;
  case ONNX_TENSOR_TYPE_FLOAT32: {
    float *py = (float *)y->datas;
    for (size_t i = 0, l = y->ndata; i < l; i++)
      py[i] = ((float)rand() / (float)RAND_MAX) * (pdat->high - pdat->low) +
              pdat->low;
  } break;
  case ONNX_TENSOR_TYPE_FLOAT64: {
    double *py = (double *)y->datas;
    for (size_t i = 0, l = y->ndata; i < l; i++)
      py[i] = ((double)rand() / (double)RAND_MAX) * (pdat->high - pdat->low) +
              pdat->low;
  } break;
  default:
    break;
  }
}

void resolver_default_op_RandomUniformLike(onnx_node_t *n) {
  if (n->opset >= 1) {
    n->init = RandomUniformLike_init;
    n->exit = RandomUniformLike_exit;
    n->reshape = RandomUniformLike_reshape;
    n->operator= RandomUniformLike_operator;
  }
}
