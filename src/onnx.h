#ifndef __ONNX_H__
#define __ONNX_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <onnx.proto3.pb-c.h>
#include <onnxconf.h>

#ifdef CONFIG_ENABLE_CUDA
#include <cuda_runtime.h>

#define CUDA_DRIVER_CALL(x)                                                    \
  {                                                                            \
    CUresult result = x;                                                       \
    if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) {        \
      const char *msg;                                                         \
      cuGetErrorName(result, &msg);                                            \
      LOG(FATAL) << "CUDAError: " #x " failed with error: " << msg;            \
    }                                                                          \
  }

#define CUDA_CALL(func)                                                        \
  {                                                                            \
    cudaError_t e = (func);                                                    \
    ICHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)                  \
        << "CUDA: " << cudaGetErrorString(e);                                  \
  }

#endif

#define LIBONNX_MAJOY (1)
#define LIBONNX_MINIOR (0)
#define LIBONNX_PATCH (0)
#define LIBONNX_VERSION                                                        \
  ((LIBONNX_MAJOY * 10000) + (LIBONNX_MINIOR * 100) + LIBONNX_PATCH)

#define EMPTY_OPERATOR() printf("Empty Operator %s at: %d(%s): \n", __FUNCTION__ , (int)__LINE__, __FILE__)

struct onnx_tensor_s;
struct onnx_node_s;
struct onnx_graph_s;
struct onnx_context_s;
struct onnx_resolver_s;

enum onnx_tensor_type_t {
  ONNX_TENSOR_TYPE_UNDEFINED = 0,
  ONNX_TENSOR_TYPE_BOOL = 9,
  ONNX_TENSOR_TYPE_INT8 = 3,
  ONNX_TENSOR_TYPE_INT16 = 5,
  ONNX_TENSOR_TYPE_INT32 = 6,
  ONNX_TENSOR_TYPE_INT64 = 7,
  ONNX_TENSOR_TYPE_UINT8 = 2,
  ONNX_TENSOR_TYPE_UINT16 = 4,
  ONNX_TENSOR_TYPE_UINT32 = 12,
  ONNX_TENSOR_TYPE_UINT64 = 13,
  ONNX_TENSOR_TYPE_BFLOAT16 = 16,
  ONNX_TENSOR_TYPE_FLOAT16 = 10,
  ONNX_TENSOR_TYPE_FLOAT32 = 1,
  ONNX_TENSOR_TYPE_FLOAT64 = 11,
  ONNX_TENSOR_TYPE_COMPLEX64 = 14,
  ONNX_TENSOR_TYPE_COMPLEX128 = 15,
  ONNX_TENSOR_TYPE_STRING = 8,
};

typedef struct onnx_tensor_s {
  char *name;
  enum onnx_tensor_type_t type;
  int *strides;
  int *dims;
  int ndim;
  void *datas;
  size_t ndata;
} onnx_tensor_t;

typedef struct onnx_node_s {
  struct onnx_context_s *ctx;
  struct onnx_resolver_s *r;
  void *rctx;
  int opset;
  struct onnx_tensor_s **inputs;
  int ninput;
  struct onnx_tensor_s **outputs;
  int noutput;
  Onnx__NodeProto *proto;

  int (*init)(struct onnx_node_s *n);
  int (*exit)(struct onnx_node_s *n);
  int (*reshape)(struct onnx_node_s *n);
  void (*operator)(struct onnx_node_s *n);
  void *priv;
} onnx_node_t;

typedef struct onnx_graph_s {
  struct onnx_node_s *nodes;
  int nlen;
} onnx_graph_t;

typedef struct onnx_context_s {
  Onnx__ModelProto *model;
  struct hmap_s *map;
  struct onnx_resolver_s **r;
  void **rctx;
  int rlen;
  struct onnx_graph_s *g;
} onnx_context_t; // xxxx8888 -- onnx_model_t

typedef struct onnx_resolver_s {
  const char *name;

  void *(*create)(void);
  void (*destroy)(void *rctx);

  void (*op_Abs)(onnx_node_t *n);
  void (*op_Acos)(onnx_node_t *n);
  void (*op_Acosh)(onnx_node_t *n);
  void (*op_Add)(onnx_node_t *n);
  void (*op_And)(onnx_node_t *n);
  void (*op_ArgMax)(onnx_node_t *n);
  void (*op_ArgMin)(onnx_node_t *n);
  void (*op_Asin)(onnx_node_t *n);
  void (*op_Asinh)(onnx_node_t *n);
  void (*op_Atan)(onnx_node_t *n);
  void (*op_Atanh)(onnx_node_t *n);
  void (*op_AveragePool)(onnx_node_t *n);
  void (*op_BatchNormalization)(onnx_node_t *n);
  void (*op_BitShift)(onnx_node_t *n);
  void (*op_Cast)(onnx_node_t *n);
  void (*op_Ceil)(onnx_node_t *n);
  void (*op_Clip)(onnx_node_t *n);
  void (*op_Compress)(onnx_node_t *n);
  void (*op_Concat)(onnx_node_t *n);
  void (*op_ConcatFromSequence)(onnx_node_t *n);
  void (*op_Constant)(onnx_node_t *n);
  void (*op_ConstantOfShape)(onnx_node_t *n);
  void (*op_Conv)(onnx_node_t *n);
  void (*op_ConvInteger)(onnx_node_t *n);
  void (*op_ConvTranspose)(onnx_node_t *n);
  void (*op_Cos)(onnx_node_t *n);
  void (*op_Cosh)(onnx_node_t *n);
  void (*op_CumSum)(onnx_node_t *n);
  void (*op_DepthToSpace)(onnx_node_t *n);
  void (*op_DequantizeLinear)(onnx_node_t *n);
  void (*op_Det)(onnx_node_t *n);
  void (*op_Div)(onnx_node_t *n);
  void (*op_Dropout)(onnx_node_t *n);
  void (*op_Einsum)(onnx_node_t *n);
  void (*op_Elu)(onnx_node_t *n);
  void (*op_Equal)(onnx_node_t *n);
  void (*op_Erf)(onnx_node_t *n);
  void (*op_Exp)(onnx_node_t *n);
  void (*op_Expand)(onnx_node_t *n);
  void (*op_EyeLike)(onnx_node_t *n);
  void (*op_Flatten)(onnx_node_t *n);
  void (*op_Floor)(onnx_node_t *n);
  void (*op_GRU)(onnx_node_t *n);
  void (*op_Gather)(onnx_node_t *n);
  void (*op_GatherElements)(onnx_node_t *n);
  void (*op_GatherND)(onnx_node_t *n);
  void (*op_Gemm)(onnx_node_t *n);
  void (*op_GlobalAveragePool)(onnx_node_t *n);
  void (*op_GlobalLpPool)(onnx_node_t *n);
  void (*op_GlobalMaxPool)(onnx_node_t *n);
  void (*op_Greater)(onnx_node_t *n);
  void (*op_HardSigmoid)(onnx_node_t *n);
  void (*op_Hardmax)(onnx_node_t *n);
  void (*op_Identity)(onnx_node_t *n);
  void (*op_If)(onnx_node_t *n);
  void (*op_InstanceNormalization)(onnx_node_t *n);
  void (*op_IsInf)(onnx_node_t *n);
  void (*op_IsNaN)(onnx_node_t *n);
  void (*op_LRN)(onnx_node_t *n);
  void (*op_LSTM)(onnx_node_t *n);
  void (*op_LeakyRelu)(onnx_node_t *n);
  void (*op_Less)(onnx_node_t *n);
  void (*op_Log)(onnx_node_t *n);
  void (*op_Loop)(onnx_node_t *n);
  void (*op_LpNormalization)(onnx_node_t *n);
  void (*op_LpPool)(onnx_node_t *n);
  void (*op_MatMul)(onnx_node_t *n);
  void (*op_MatMulInteger)(onnx_node_t *n);
  void (*op_Max)(onnx_node_t *n);
  void (*op_MaxPool)(onnx_node_t *n);
  void (*op_MaxRoiPool)(onnx_node_t *n);
  void (*op_MaxUnpool)(onnx_node_t *n);
  void (*op_Mean)(onnx_node_t *n);
  void (*op_Min)(onnx_node_t *n);
  void (*op_Mod)(onnx_node_t *n);
  void (*op_Mul)(onnx_node_t *n);
  void (*op_Multinomial)(onnx_node_t *n);
  void (*op_Neg)(onnx_node_t *n);
  void (*op_NonMaxSuppression)(onnx_node_t *n);
  void (*op_NonZero)(onnx_node_t *n);
  void (*op_Not)(onnx_node_t *n);
  void (*op_OneHot)(onnx_node_t *n);
  void (*op_Or)(onnx_node_t *n);
  void (*op_PRelu)(onnx_node_t *n);
  void (*op_Pad)(onnx_node_t *n);
  void (*op_Pow)(onnx_node_t *n);
  void (*op_QLinearConv)(onnx_node_t *n);
  void (*op_QLinearMatMul)(onnx_node_t *n);
  void (*op_QuantizeLinear)(onnx_node_t *n);
  void (*op_RNN)(onnx_node_t *n);
  void (*op_RandomNormal)(onnx_node_t *n);
  void (*op_RandomNormalLike)(onnx_node_t *n);
  void (*op_RandomUniform)(onnx_node_t *n);
  void (*op_RandomUniformLike)(onnx_node_t *n);
  void (*op_Reciprocal)(onnx_node_t *n);
  void (*op_ReduceL1)(onnx_node_t *n);
  void (*op_ReduceL2)(onnx_node_t *n);
  void (*op_ReduceLogSum)(onnx_node_t *n);
  void (*op_ReduceLogSumExp)(onnx_node_t *n);
  void (*op_ReduceMax)(onnx_node_t *n);
  void (*op_ReduceMean)(onnx_node_t *n);
  void (*op_ReduceMin)(onnx_node_t *n);
  void (*op_ReduceProd)(onnx_node_t *n);
  void (*op_ReduceSum)(onnx_node_t *n);
  void (*op_ReduceSumSquare)(onnx_node_t *n);
  void (*op_Relu)(onnx_node_t *n);
  void (*op_Reshape)(onnx_node_t *n);
  void (*op_Resize)(onnx_node_t *n);
  void (*op_ReverseSequence)(onnx_node_t *n);
  void (*op_RoiAlign)(onnx_node_t *n);
  void (*op_Round)(onnx_node_t *n);
  void (*op_Scan)(onnx_node_t *n);
  void (*op_Scatter)(onnx_node_t *n);
  void (*op_ScatterElements)(onnx_node_t *n);
  void (*op_ScatterND)(onnx_node_t *n);
  void (*op_Selu)(onnx_node_t *n);
  void (*op_SequenceAt)(onnx_node_t *n);
  void (*op_SequenceConstruct)(onnx_node_t *n);
  void (*op_SequenceEmpty)(onnx_node_t *n);
  void (*op_SequenceErase)(onnx_node_t *n);
  void (*op_SequenceInsert)(onnx_node_t *n);
  void (*op_SequenceLength)(onnx_node_t *n);
  void (*op_Shape)(onnx_node_t *n);
  void (*op_Shrink)(onnx_node_t *n);
  void (*op_Sigmoid)(onnx_node_t *n);
  void (*op_Sign)(onnx_node_t *n);
  void (*op_Sin)(onnx_node_t *n);
  void (*op_Sinh)(onnx_node_t *n);
  void (*op_Size)(onnx_node_t *n);
  void (*op_Slice)(onnx_node_t *n);
  void (*op_Softplus)(onnx_node_t *n);
  void (*op_Softsign)(onnx_node_t *n);
  void (*op_SpaceToDepth)(onnx_node_t *n);
  void (*op_Split)(onnx_node_t *n);
  void (*op_SplitToSequence)(onnx_node_t *n);
  void (*op_Sqrt)(onnx_node_t *n);
  void (*op_Squeeze)(onnx_node_t *n);
  void (*op_StringNormalizer)(onnx_node_t *n);
  void (*op_Sub)(onnx_node_t *n);
  void (*op_Sum)(onnx_node_t *n);
  void (*op_Tan)(onnx_node_t *n);
  void (*op_Tanh)(onnx_node_t *n);
  void (*op_TfIdfVectorizer)(onnx_node_t *n);
  void (*op_ThresholdedRelu)(onnx_node_t *n);
  void (*op_Tile)(onnx_node_t *n);
  void (*op_TopK)(onnx_node_t *n);
  void (*op_Transpose)(onnx_node_t *n);
  void (*op_Trilu)(onnx_node_t *n);
  void (*op_Unique)(onnx_node_t *n);
  void (*op_Unsqueeze)(onnx_node_t *n);
  void (*op_Upsample)(onnx_node_t *n);
  void (*op_Where)(onnx_node_t *n);
  void (*op_Xor)(onnx_node_t *n);

  void (*op_Celu)(onnx_node_t *n);
  void (*op_DynamicQuantizeLinear)(onnx_node_t *n);
  void (*op_GreaterOrEqual)(onnx_node_t *n);
  void (*op_HardSwish)(onnx_node_t *n);
  void (*op_LessOrEqual)(onnx_node_t *n);
  void (*op_LogSoftmax)(onnx_node_t *n);
  void (*op_MeanVarianceNormalization)(onnx_node_t *n);
  void (*op_NegativeLogLikelihoodLoss)(onnx_node_t *n);
  void (*op_Range)(onnx_node_t *n);
  void (*op_Softmax)(onnx_node_t *n);
  void (*op_SoftmaxCrossEntropyLoss)(onnx_node_t *n);
} onnx_resolver_t;

onnx_context_t *onnx_context_alloc(const void *buf, size_t len,
                                   onnx_resolver_t **r, int rlen);
onnx_context_t *onnx_context_alloc_from_file(const char *filename,
                                             onnx_resolver_t **r, int rlen);
void onnx_context_free(onnx_context_t *ctx);

onnx_graph_t *onnx_graph_alloc(onnx_context_t *ctx, Onnx__GraphProto *graph);
void onnx_graph_free(onnx_graph_t *g);

const char *onnx_tensor_type_tostring(enum onnx_tensor_type_t type);
int onnx_tensor_type_sizeof(enum onnx_tensor_type_t type);
onnx_tensor_t *onnx_tensor_search(onnx_context_t *ctx, const char *name);
onnx_tensor_t *onnx_tensor_alloc(const char *name, enum onnx_tensor_type_t type,
                                 int *dims, int ndim);
onnx_tensor_t *onnx_tensor_alloc_from_file(const char *filename);
void onnx_tensor_free(onnx_tensor_t *t);
int onnx_tensor_equal(onnx_tensor_t *a, onnx_tensor_t *b);
void onnx_tensor_reinit(onnx_tensor_t *t, enum onnx_tensor_type_t type,
                        int *dims, int ndim);
void onnx_tensor_apply(onnx_tensor_t *t, void *buf, size_t len);

static inline int onnx_tensor_is_scalar(onnx_tensor_t *t) {
  return ((t->ndim == 0) && (t->ndata == 1)) ? 1 : 0;
}

static inline int onnx_tensor_broadcast_is_valid(onnx_tensor_t *x, int *dims,
                                                 int ndim) {
  int i;

  if (x->ndim > ndim)
    return 0;
  for (i = 1; i <= x->ndim; i++) {
    if ((x->dims[x->ndim - i] != 1) && (x->dims[x->ndim - i] != dims[ndim - i]))
      return 0;
  }
  return 1;
}

static inline int onnx_tensor_indices_to_offset(onnx_tensor_t *t,
                                                int *indices) {
  int offset, i;

  for (i = 0, offset = 0; i < t->ndim; i++)
    offset += indices[i] * t->strides[i];
  return offset;
}

static inline void onnx_tensor_offset_to_indices(onnx_tensor_t *t, int offset,
                                                 int *indices) {
  int i;

  for (i = t->ndim - 1; i >= 0; i--) {
    indices[i] = offset % t->dims[i];
    offset /= t->dims[i];
  }
}

static inline int onnx_tensor_reshape(onnx_tensor_t *y, int *dims, int ndim,
                                      enum onnx_tensor_type_t type) {
  if ((y->ndim != ndim) ||
      (dims && (memcmp(y->dims, dims, sizeof(int) * y->ndim) != 0)) ||
      (y->type != type))
    onnx_tensor_reinit(y, type, dims, ndim);
  return 1;
}

static inline int onnx_tensor_reshape_identity(onnx_tensor_t *y,
                                               onnx_tensor_t *x,
                                               enum onnx_tensor_type_t type) {
  if ((y->ndim != x->ndim) ||
      (memcmp(y->dims, x->dims, sizeof(int) * y->ndim) != 0) ||
      (y->type != type))
    onnx_tensor_reinit(y, type, x->dims, x->ndim);
  return 1;
}

static inline int
onnx_tensor_reshape_multi_broadcast(onnx_tensor_t *y, onnx_tensor_t *a,
                                    onnx_tensor_t *b,
                                    enum onnx_tensor_type_t type) {
  int ndim = max(a->ndim, b->ndim);
  int dims[ndim];
  int i, j, k;

  if (ndim > 0) {
    for (i = a->ndim - 1, j = b->ndim - 1, k = ndim - 1; k >= 0; k--) {
      if (i < 0)
        dims[k] = b->dims[j--];
      else if (j < 0)
        dims[k] = a->dims[i--];
      else {
        if (a->dims[i] == b->dims[j])
          dims[k] = a->dims[i];
        else if ((a->dims[i] == 1) || (b->dims[j] == 1))
          dims[k] = (a->dims[i] > b->dims[j]) ? a->dims[i] : b->dims[j];
        else
          return 0;
        i--;
        j--;
      }
    }
  }
  if ((y->type != type) || (y->ndim != ndim) ||
      (memcmp(y->dims, dims, sizeof(int) * ndim) != 0))
    onnx_tensor_reinit(y, type, dims, ndim);
  return 1;
}

static inline void *onnx_tensor_broadcast_map_address(onnx_tensor_t *x,
                                                      onnx_tensor_t *y,
                                                      int offset) {
  int xndim = x->ndim;
  int yndim = y->ndim;

  if ((xndim > 0) && (yndim > 0)) {
    int dndim = yndim - xndim;
    int ix[xndim];
    int iy[yndim];
    int i;

    onnx_tensor_offset_to_indices(y, offset, iy);
    for (i = 0; i < xndim; i++)
      ix[i] = iy[dndim + i] % x->dims[i];
    return x->datas + onnx_tensor_indices_to_offset(x, ix) *
                          onnx_tensor_type_sizeof(x->type);
  }
  return x->datas;
}

float onnx_attribute_read_float(onnx_node_t *n, const char *name, float def);
int64_t onnx_attribute_read_int(onnx_node_t *n, const char *name, int64_t def);
char *onnx_attribute_read_string(onnx_node_t *n, const char *name, char *def);
int onnx_attribute_read_ints(onnx_node_t *n, const char *name, int64_t **ints);
int onnx_attribute_read_floats(onnx_node_t *n, const char *name,
                               float **floats);
int onnx_attribute_read_tensor(onnx_node_t *n, const char *name,
                               onnx_tensor_t *t);
Onnx__GraphProto *onnx_attribute_read_graph(onnx_node_t *n, const char *name,
                                            Onnx__GraphProto *def);
Onnx__SparseTensorProto *
onnx_attribute_read_sparse_tensor(onnx_node_t *n, const char *name,
                                  Onnx__SparseTensorProto *def);

void onnx_tensor_dump(onnx_tensor_t *t, int detail);
void onnx_node_dump(onnx_node_t *n, int detail);
void onnx_graph_dump(onnx_graph_t *g, int detail);
void onnx_context_dump(onnx_context_t *ctx, int detail);

void onnx_run(onnx_context_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* __ONNX_H__ */
