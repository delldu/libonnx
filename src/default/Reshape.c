#include <onnx.h>

static int Reshape_init(onnx_node_t *n) {
  onnx_tensor_t *x;
  onnx_tensor_t *s;

  if ((n->ninput == 2) && (n->noutput == 1)) {
    x = n->inputs[0];
    s = n->inputs[1];
    if ((x->ndim == 0) || (x->type == ONNX_TENSOR_TYPE_UNDEFINED))
      return 0;
    if ((s->ndim == 0) || (s->type != ONNX_TENSOR_TYPE_INT64))
      return 0;
    return 1;
  }
  return 0;
}

static int Reshape_exit(onnx_node_t *n) { return 1; }

static int Reshape_reshape(onnx_node_t *n) {
  onnx_tensor_t *y = n->outputs[0];
  onnx_tensor_t *x = n->inputs[0];
  onnx_tensor_t *s = n->inputs[1];
  int64_t *ps = s->datas;
  int total_dim = 1;
  int total_shape = 1;
  int ndim = s->ndata;
  int dims[ndim];

  for (int i = 0; i < ndim; i++) {
    if (ps[i] == 0)
      dims[i] = x->dims[i];
    else if (ps[i] > 0)
      dims[i] = ps[i];
    else {
      for (int j = 0; j < x->ndim; j++)
        total_dim *= x->dims[j];
      for (int j = 0; j < ndim; j++) {
        if (ps[j] > 0)
          total_shape *= ps[j];
        else if (ps[j] == 0)
          total_shape *= x->dims[j];
      }
      dims[i] = total_dim / total_shape;
    }
  }
  return onnx_tensor_reshape(y, dims, ndim, x->type);
}

static void Reshape_operator(onnx_node_t *n) {
  onnx_tensor_t *y = n->outputs[0];
  onnx_tensor_t *x = n->inputs[0];
  char **py = (char **)y->datas;
  char **px = (char **)x->datas;

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

void resolver_default_op_Reshape(onnx_node_t *n) {
  if (n->opset >= 14) {
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
      n->init = Reshape_init;
      n->exit = Reshape_exit;
      n->reshape = Reshape_reshape;
      n->operator= Reshape_operator;
      break;
    default:
      break;
    }
  } else if (n->opset >= 13) {
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
      n->init = Reshape_init;
      n->exit = Reshape_exit;
      n->reshape = Reshape_reshape;
      n->operator= Reshape_operator;
      break;
    default:
      break;
    }
  } else if (n->opset >= 5) {
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
      n->init = Reshape_init;
      n->exit = Reshape_exit;
      n->reshape = Reshape_reshape;
      n->operator= Reshape_operator;
      break;
    default:
      break;
    }
  } else if (n->opset >= 1) {
  }
}
