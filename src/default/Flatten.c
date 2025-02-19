#include <onnx.h>

typedef struct {
  int axis;
} operator_pdata_t;

static int Flatten_init(onnx_node_t *n) {
  operator_pdata_t *pdat;

  if ((n->ninput == 1) && (n->noutput == 1)) {
    pdat = malloc(sizeof(operator_pdata_t));
    if (pdat) {
      pdat->axis = onnx_attribute_read_int(n, "axis", 1);
      n->priv = pdat;
      return 1;
    }
  }
  return 0;
}

static int Flatten_exit(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;

  if (pdat)
    free(pdat);
  return 1;
}

static int Flatten_reshape(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];
  int axis = pdat->axis;
  int dims[x->ndim];
  int ndim;
  int i, j;

  if (axis < 0)
    axis += x->ndim;
  if (axis < 0 || axis >= x->ndim)
    return 0;
  for (i = 0, j = 1, ndim = 0; i < x->ndim; i++) {
    if (i != axis)
      j *= x->dims[i];
    else {
      dims[ndim++] = j;
      j = x->dims[i];
    }
  }
  dims[ndim++] = j;
  return onnx_tensor_reshape(y, dims, ndim, x->type);
}

static void Flatten_operator(onnx_node_t *n) {
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *y = n->outputs[0];
  char **px = (char **)x->datas;
  char **py = (char **)y->datas;

  if (x->type == ONNX_TENSOR_TYPE_STRING) {
    for (size_t i = 0, l = y->ndata; i < l; i++) {
      if (py[i])
        free(py[i]);
      py[i] = strdup(px[i]);
    }
  } else {
    memcpy(y->datas, x->datas, x->ndata * onnx_tensor_type_sizeof(x->type));
  }
}

void resolver_default_op_Flatten(onnx_node_t *n) {
  n->init = Flatten_init;
  n->exit = Flatten_exit;
  n->reshape = Flatten_reshape;

  if (n->opset >= 13) {
    switch (n->inputs[0]->type) {
    case ONNX_TENSOR_TYPE_BOOL:
    case ONNX_TENSOR_TYPE_INT8:
    case ONNX_TENSOR_TYPE_INT16:
    case ONNX_TENSOR_TYPE_INT32:
    case ONNX_TENSOR_TYPE_INT64:
    case ONNX_TENSOR_TYPE_UINT8:
    case ONNX_TENSOR_TYPE_UINT16:
    case ONNX_TENSOR_TYPE_UINT32:
    case ONNX_TENSOR_TYPE_UINT64:
    case ONNX_TENSOR_TYPE_BFLOAT16:
    case ONNX_TENSOR_TYPE_FLOAT16:
    case ONNX_TENSOR_TYPE_FLOAT32:
    case ONNX_TENSOR_TYPE_FLOAT64:
    case ONNX_TENSOR_TYPE_COMPLEX64:
    case ONNX_TENSOR_TYPE_COMPLEX128:
    case ONNX_TENSOR_TYPE_STRING:
      n->operator= Flatten_operator;
      break;
    default:
      break;
    }
  } else if (n->opset >= 11) {
    switch (n->inputs[0]->type) {
    case ONNX_TENSOR_TYPE_BOOL:
    case ONNX_TENSOR_TYPE_INT8:
    case ONNX_TENSOR_TYPE_INT16:
    case ONNX_TENSOR_TYPE_INT32:
    case ONNX_TENSOR_TYPE_INT64:
    case ONNX_TENSOR_TYPE_UINT8:
    case ONNX_TENSOR_TYPE_UINT16:
    case ONNX_TENSOR_TYPE_UINT32:
    case ONNX_TENSOR_TYPE_UINT64:
    case ONNX_TENSOR_TYPE_FLOAT16:
    case ONNX_TENSOR_TYPE_FLOAT32:
    case ONNX_TENSOR_TYPE_FLOAT64:
    case ONNX_TENSOR_TYPE_COMPLEX64:
    case ONNX_TENSOR_TYPE_COMPLEX128:
    case ONNX_TENSOR_TYPE_STRING:
      n->operator= Flatten_operator;
      break;
    default:
      break;
    }
  } else if (n->opset >= 9) {
    switch (n->inputs[0]->type) {
    case ONNX_TENSOR_TYPE_BOOL:
    case ONNX_TENSOR_TYPE_INT8:
    case ONNX_TENSOR_TYPE_INT16:
    case ONNX_TENSOR_TYPE_INT32:
    case ONNX_TENSOR_TYPE_INT64:
    case ONNX_TENSOR_TYPE_UINT8:
    case ONNX_TENSOR_TYPE_UINT16:
    case ONNX_TENSOR_TYPE_UINT32:
    case ONNX_TENSOR_TYPE_UINT64:
    case ONNX_TENSOR_TYPE_FLOAT16:
    case ONNX_TENSOR_TYPE_FLOAT32:
    case ONNX_TENSOR_TYPE_FLOAT64:
    case ONNX_TENSOR_TYPE_COMPLEX64:
    case ONNX_TENSOR_TYPE_COMPLEX128:
    case ONNX_TENSOR_TYPE_STRING:
      n->operator= Flatten_operator;
      break;
    default:
      break;
    }
  } else if (n->opset >= 1) {
    switch (n->inputs[0]->type) {
    case ONNX_TENSOR_TYPE_FLOAT16:
    case ONNX_TENSOR_TYPE_FLOAT32:
    case ONNX_TENSOR_TYPE_FLOAT64:
      n->operator= Flatten_operator;
      break;
    default:
      break;
    }
  }
}
