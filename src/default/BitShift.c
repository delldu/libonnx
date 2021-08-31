#include <onnx.h>

typedef struct {
  int isleft;
} operator_pdata_t;

static int BitShift_init(onnx_node_t *n) {
  operator_pdata_t *pdat;

  if ((n->ninput == 2) && (n->noutput == 1)) {
    pdat = malloc(sizeof(operator_pdata_t));
    if (pdat) {
      pdat->isleft = (strcmp(onnx_attribute_read_string(n, "direction", "LEFT"),
                             "LEFT") == 0)
                         ? 1
                         : 0;
      n->priv = pdat;
      return 1;
    }
  }
  return 0;
}

static int BitShift_exit(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;

  if (pdat)
    free(pdat);
  return 1;
}

static int BitShift_reshape(onnx_node_t *n) {
  onnx_tensor_t *y = n->outputs[0];
  onnx_tensor_t *a = n->inputs[0];
  onnx_tensor_t *b = n->inputs[1];

  return onnx_tensor_reshape_multi_broadcast(y, a, b, a->type);
}

static void BitShift_uint8(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;
  onnx_tensor_t *y = n->outputs[0];
  onnx_tensor_t *a = n->inputs[0];
  onnx_tensor_t *b = n->inputs[1];
  uint8_t *py = (uint8_t *)y->datas;
  uint8_t *pa;
  uint8_t *pb;

  if (pdat->isleft) {
    for (size_t i = 0, l = y->ndata; i < l; i++) {
      pa = onnx_tensor_broadcast_map_address(a, y, i);
      pb = onnx_tensor_broadcast_map_address(b, y, i);
      py[i] = *pa << *pb;
    }
  } else {
    for (size_t i = 0, l = y->ndata; i < l; i++) {
      pa = onnx_tensor_broadcast_map_address(a, y, i);
      pb = onnx_tensor_broadcast_map_address(b, y, i);
      py[i] = *pa >> *pb;
    }
  }
}

static void BitShift_uint16(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;
  onnx_tensor_t *y = n->outputs[0];
  onnx_tensor_t *a = n->inputs[0];
  onnx_tensor_t *b = n->inputs[1];
  uint16_t *py = (uint16_t *)y->datas;
  uint16_t *pa;
  uint16_t *pb;

  if (pdat->isleft) {
    for (size_t i = 0, l = y->ndata; i < l; i++) {
      pa = onnx_tensor_broadcast_map_address(a, y, i);
      pb = onnx_tensor_broadcast_map_address(b, y, i);
      py[i] = *pa << *pb;
    }
  } else {
    for (size_t i = 0, l = y->ndata; i < l; i++) {
      pa = onnx_tensor_broadcast_map_address(a, y, i);
      pb = onnx_tensor_broadcast_map_address(b, y, i);
      py[i] = *pa >> *pb;
    }
  }
}

static void BitShift_uint32(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;
  onnx_tensor_t *y = n->outputs[0];
  onnx_tensor_t *a = n->inputs[0];
  onnx_tensor_t *b = n->inputs[1];
  uint32_t *py = (uint32_t *)y->datas;
  uint32_t *pa;
  uint32_t *pb;

  if (pdat->isleft) {
    for (size_t i = 0, l = y->ndata; i < l; i++) {
      pa = onnx_tensor_broadcast_map_address(a, y, i);
      pb = onnx_tensor_broadcast_map_address(b, y, i);
      py[i] = *pa << *pb;
    }
  } else {
    for (size_t i = 0, l = y->ndata; i < l; i++) {
      pa = onnx_tensor_broadcast_map_address(a, y, i);
      pb = onnx_tensor_broadcast_map_address(b, y, i);
      py[i] = *pa >> *pb;
    }
  }
}

static void BitShift_uint64(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;
  onnx_tensor_t *y = n->outputs[0];
  onnx_tensor_t *a = n->inputs[0];
  onnx_tensor_t *b = n->inputs[1];
  uint64_t *py = (uint64_t *)y->datas;
  uint64_t *pa;
  uint64_t *pb;

  if (pdat->isleft) {
    for (size_t i = 0, l = y->ndata; i < l; i++) {
      pa = onnx_tensor_broadcast_map_address(a, y, i);
      pb = onnx_tensor_broadcast_map_address(b, y, i);
      py[i] = *pa << *pb;
    }
  } else {
    for (size_t i = 0, l = y->ndata; i < l; i++) {
      pa = onnx_tensor_broadcast_map_address(a, y, i);
      pb = onnx_tensor_broadcast_map_address(b, y, i);
      py[i] = *pa >> *pb;
    }
  }
}

void resolver_default_op_BitShift(onnx_node_t *n) {
    n->init = BitShift_init;
    n->exit = BitShift_exit;
    n->reshape = BitShift_reshape;

  if (n->opset >= 11) {
    switch (n->inputs[0]->type) {
    case ONNX_TENSOR_TYPE_UINT8:
      n->operator= BitShift_uint8;
      break;
    case ONNX_TENSOR_TYPE_UINT16:
      n->operator= BitShift_uint16;
      break;
    case ONNX_TENSOR_TYPE_UINT32:
      n->operator= BitShift_uint32;
      break;
    case ONNX_TENSOR_TYPE_UINT64:
      n->operator= BitShift_uint64;
      break;
    default:
      break;
    }
  }
}
