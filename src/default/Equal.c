#include <onnx.h>

static int Equal_init(onnx_node_t *n) {
  if ((n->ninput == 2) && (n->noutput == 1))
    return 1;
  return 0;
}

static int Equal_exit(onnx_node_t *n) { return 1; }

static int Equal_reshape(onnx_node_t *n) {
  onnx_tensor_t *y = n->outputs[0];
  onnx_tensor_t *a = n->inputs[0];
  onnx_tensor_t *b = n->inputs[1];

  return onnx_tensor_reshape_multi_broadcast(y, a, b, ONNX_TENSOR_TYPE_BOOL);
}

static void Equal_bool(onnx_node_t *n) {
  onnx_tensor_t *y = n->outputs[0];
  onnx_tensor_t *a = n->inputs[0];
  onnx_tensor_t *b = n->inputs[1];
  uint8_t *py = (uint8_t *)y->datas;
  uint8_t *pa;
  uint8_t *pb;

  for (size_t i = 0, l = y->ndata; i < l; i++) {
    pa = onnx_tensor_broadcast_map_address(a, y, i);
    pb = onnx_tensor_broadcast_map_address(b, y, i);
    py[i] = (*pa == *pb) ? 1 : 0;
  }
}

static void Equal_int8(onnx_node_t *n) {
  onnx_tensor_t *y = n->outputs[0];
  onnx_tensor_t *a = n->inputs[0];
  onnx_tensor_t *b = n->inputs[1];
  uint8_t *py = (uint8_t *)y->datas;
  int8_t *pa;
  int8_t *pb;

  for (size_t i = 0, l = y->ndata; i < l; i++) {
    pa = onnx_tensor_broadcast_map_address(a, y, i);
    pb = onnx_tensor_broadcast_map_address(b, y, i);
    py[i] = (*pa == *pb) ? 1 : 0;
  }
}

static void Equal_int16(onnx_node_t *n) {
  onnx_tensor_t *y = n->outputs[0];
  onnx_tensor_t *a = n->inputs[0];
  onnx_tensor_t *b = n->inputs[1];
  uint8_t *py = (uint8_t *)y->datas;
  int16_t *pa;
  int16_t *pb;

  for (size_t i = 0, l = y->ndata; i < l; i++) {
    pa = onnx_tensor_broadcast_map_address(a, y, i);
    pb = onnx_tensor_broadcast_map_address(b, y, i);
    py[i] = (*pa == *pb) ? 1 : 0;
  }
}

static void Equal_int32(onnx_node_t *n) {
  onnx_tensor_t *y = n->outputs[0];
  onnx_tensor_t *a = n->inputs[0];
  onnx_tensor_t *b = n->inputs[1];
  uint8_t *py = (uint8_t *)y->datas;
  int32_t *pa;
  int32_t *pb;

  for (size_t i = 0, l = y->ndata; i < l; i++) {
    pa = onnx_tensor_broadcast_map_address(a, y, i);
    pb = onnx_tensor_broadcast_map_address(b, y, i);
    py[i] = (*pa == *pb) ? 1 : 0;
  }
}

static void Equal_int64(onnx_node_t *n) {
  onnx_tensor_t *y = n->outputs[0];
  onnx_tensor_t *a = n->inputs[0];
  onnx_tensor_t *b = n->inputs[1];
  uint8_t *py = (uint8_t *)y->datas;
  int64_t *pa;
  int64_t *pb;

  for (size_t i = 0, l = y->ndata; i < l; i++) {
    pa = onnx_tensor_broadcast_map_address(a, y, i);
    pb = onnx_tensor_broadcast_map_address(b, y, i);
    py[i] = (*pa == *pb) ? 1 : 0;
  }
}

static void Equal_uint8(onnx_node_t *n) {
  onnx_tensor_t *y = n->outputs[0];
  onnx_tensor_t *a = n->inputs[0];
  onnx_tensor_t *b = n->inputs[1];
  uint8_t *py = (uint8_t *)y->datas;
  uint8_t *pa;
  uint8_t *pb;

  for (size_t i = 0, l = y->ndata; i < l; i++) {
    pa = onnx_tensor_broadcast_map_address(a, y, i);
    pb = onnx_tensor_broadcast_map_address(b, y, i);
    py[i] = (*pa == *pb) ? 1 : 0;
  }
}

static void Equal_uint16(onnx_node_t *n) {
  onnx_tensor_t *y = n->outputs[0];
  onnx_tensor_t *a = n->inputs[0];
  onnx_tensor_t *b = n->inputs[1];
  uint8_t *py = (uint8_t *)y->datas;
  uint16_t *pa;
  uint16_t *pb;

  for (size_t i = 0, l = y->ndata; i < l; i++) {
    pa = onnx_tensor_broadcast_map_address(a, y, i);
    pb = onnx_tensor_broadcast_map_address(b, y, i);
    py[i] = (*pa == *pb) ? 1 : 0;
  }
}

static void Equal_uint32(onnx_node_t *n) {
  onnx_tensor_t *y = n->outputs[0];
  onnx_tensor_t *a = n->inputs[0];
  onnx_tensor_t *b = n->inputs[1];
  uint8_t *py = (uint8_t *)y->datas;
  uint32_t *pa;
  uint32_t *pb;

  for (size_t i = 0, l = y->ndata; i < l; i++) {
    pa = onnx_tensor_broadcast_map_address(a, y, i);
    pb = onnx_tensor_broadcast_map_address(b, y, i);
    py[i] = (*pa == *pb) ? 1 : 0;
  }
}

static void Equal_uint64(onnx_node_t *n) {
  onnx_tensor_t *y = n->outputs[0];
  onnx_tensor_t *a = n->inputs[0];
  onnx_tensor_t *b = n->inputs[1];
  uint8_t *py = (uint8_t *)y->datas;
  uint64_t *pa;
  uint64_t *pb;

  for (size_t i = 0, l = y->ndata; i < l; i++) {
    pa = onnx_tensor_broadcast_map_address(a, y, i);
    pb = onnx_tensor_broadcast_map_address(b, y, i);
    py[i] = (*pa == *pb) ? 1 : 0;
  }
}

static void Equal_bfloat16(onnx_node_t *n) {
  onnx_tensor_t *y = n->outputs[0];
  onnx_tensor_t *a = n->inputs[0];
  onnx_tensor_t *b = n->inputs[1];
  uint8_t *py = (uint8_t *)y->datas;
  uint16_t *pa;
  uint16_t *pb;

  for (size_t i = 0, l = y->ndata; i < l; i++) {
    pa = onnx_tensor_broadcast_map_address(a, y, i);
    pb = onnx_tensor_broadcast_map_address(b, y, i);
    py[i] = (bfloat16_to_float32(*pa) == bfloat16_to_float32(*pb)) ? 1 : 0;
  }
}

static void Equal_float16(onnx_node_t *n) {
  onnx_tensor_t *y = n->outputs[0];
  onnx_tensor_t *a = n->inputs[0];
  onnx_tensor_t *b = n->inputs[1];
  uint8_t *py = (uint8_t *)y->datas;
  uint16_t *pa;
  uint16_t *pb;

  for (size_t i = 0, l = y->ndata; i < l; i++) {
    pa = onnx_tensor_broadcast_map_address(a, y, i);
    pb = onnx_tensor_broadcast_map_address(b, y, i);
    py[i] = (float16_to_float32(*pa) == float16_to_float32(*pb)) ? 1 : 0;
  }
}

static void Equal_float32(onnx_node_t *n) {
  onnx_tensor_t *y = n->outputs[0];
  onnx_tensor_t *a = n->inputs[0];
  onnx_tensor_t *b = n->inputs[1];
  uint8_t *py = (uint8_t *)y->datas;
  float *pa;
  float *pb;

  for (size_t i = 0, l = y->ndata; i < l; i++) {
    pa = onnx_tensor_broadcast_map_address(a, y, i);
    pb = onnx_tensor_broadcast_map_address(b, y, i);
    py[i] = (*pa == *pb) ? 1 : 0;
  }
}

static void Equal_float64(onnx_node_t *n) {
  onnx_tensor_t *y = n->outputs[0];
  onnx_tensor_t *a = n->inputs[0];
  onnx_tensor_t *b = n->inputs[1];
  uint8_t *py = (uint8_t *)y->datas;
  double *pa;
  double *pb;

  for (size_t i = 0, l = y->ndata; i < l; i++) {
    pa = onnx_tensor_broadcast_map_address(a, y, i);
    pb = onnx_tensor_broadcast_map_address(b, y, i);
    py[i] = (*pa == *pb) ? 1 : 0;
  }
}

void resolver_default_op_Equal(onnx_node_t *n) {
  n->init = Equal_init;
  n->exit = Equal_exit;
  n->reshape = Equal_reshape;

  if (n->opset >= 13) {
    switch (n->inputs[0]->type) {
    case ONNX_TENSOR_TYPE_BOOL:
      n->operator= Equal_bool;
      break;
    case ONNX_TENSOR_TYPE_INT8:
      n->operator= Equal_int8;
      break;
    case ONNX_TENSOR_TYPE_INT16:
      n->operator= Equal_int16;
      break;
    case ONNX_TENSOR_TYPE_INT32:
      n->operator= Equal_int32;
      break;
    case ONNX_TENSOR_TYPE_INT64:
      n->operator= Equal_int64;
      break;
    case ONNX_TENSOR_TYPE_UINT8:
      n->operator= Equal_uint8;
      break;
    case ONNX_TENSOR_TYPE_UINT16:
      n->operator= Equal_uint16;
      break;
    case ONNX_TENSOR_TYPE_UINT32:
      n->operator= Equal_uint32;
      break;
    case ONNX_TENSOR_TYPE_UINT64:
      n->operator= Equal_uint64;
      break;
    case ONNX_TENSOR_TYPE_BFLOAT16:
      n->operator= Equal_bfloat16;
      break;
    case ONNX_TENSOR_TYPE_FLOAT16:
      n->operator= Equal_float16;
      break;
    case ONNX_TENSOR_TYPE_FLOAT32:
      n->operator= Equal_float32;
      break;
    case ONNX_TENSOR_TYPE_FLOAT64:
      n->operator= Equal_float64;
      break;
    default:
      break;
    }
  } else if (n->opset >= 11) {
    switch (n->inputs[0]->type) {
    case ONNX_TENSOR_TYPE_BOOL:
      n->operator= Equal_bool;
      break;
    case ONNX_TENSOR_TYPE_INT8:
      n->operator= Equal_int8;
      break;
    case ONNX_TENSOR_TYPE_INT16:
      n->operator= Equal_int16;
      break;
    case ONNX_TENSOR_TYPE_INT32:
      n->operator= Equal_int32;
      break;
    case ONNX_TENSOR_TYPE_INT64:
      n->operator= Equal_int64;
      break;
    case ONNX_TENSOR_TYPE_UINT8:
      n->operator= Equal_uint8;
      break;
    case ONNX_TENSOR_TYPE_UINT16:
      n->operator= Equal_uint16;
      break;
    case ONNX_TENSOR_TYPE_UINT32:
      n->operator= Equal_uint32;
      break;
    case ONNX_TENSOR_TYPE_UINT64:
      n->operator= Equal_uint64;
      break;
    case ONNX_TENSOR_TYPE_FLOAT16:
      n->operator= Equal_float16;
      break;
    case ONNX_TENSOR_TYPE_FLOAT32:
      n->operator= Equal_float32;
      break;
    case ONNX_TENSOR_TYPE_FLOAT64:
      n->operator= Equal_float64;
      break;
    default:
      break;
    }
  } else if (n->opset >= 7) {
    switch (n->inputs[0]->type) {
    case ONNX_TENSOR_TYPE_BOOL:
      n->operator= Equal_bool;
      break;
    case ONNX_TENSOR_TYPE_INT32:
      n->operator= Equal_int32;
      break;
    case ONNX_TENSOR_TYPE_INT64:
      n->operator= Equal_int64;
      break;
    default:
      break;
    }
  } else if (n->opset >= 1) {
    EMPTY_OPERATOR();
  }
}
