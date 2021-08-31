#ifndef __DEFAULT_H__
#define __DEFAULT_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <onnx.h>

void *resolver_default_create(void);
void resolver_default_destroy(void *rctx);

void resolver_default_op_Abs(onnx_node_t *n);
void resolver_default_op_Acos(onnx_node_t *n);
void resolver_default_op_Acosh(onnx_node_t *n);
void resolver_default_op_Add(onnx_node_t *n);
void resolver_default_op_And(onnx_node_t *n);
void resolver_default_op_ArgMax(onnx_node_t *n);
void resolver_default_op_ArgMin(onnx_node_t *n);
void resolver_default_op_Asin(onnx_node_t *n);
void resolver_default_op_Asinh(onnx_node_t *n);
void resolver_default_op_Atan(onnx_node_t *n);
void resolver_default_op_Atanh(onnx_node_t *n);
void resolver_default_op_AveragePool(onnx_node_t *n);
void resolver_default_op_BatchNormalization(onnx_node_t *n);
void resolver_default_op_BitShift(onnx_node_t *n);
void resolver_default_op_Cast(onnx_node_t *n);
void resolver_default_op_Ceil(onnx_node_t *n);
void resolver_default_op_Clip(onnx_node_t *n);
void resolver_default_op_Compress(onnx_node_t *n);
void resolver_default_op_Concat(onnx_node_t *n);
void resolver_default_op_ConcatFromSequence(onnx_node_t *n);
void resolver_default_op_Constant(onnx_node_t *n);
void resolver_default_op_ConstantOfShape(onnx_node_t *n);
void resolver_default_op_Conv(onnx_node_t *n);
void resolver_default_op_ConvInteger(onnx_node_t *n);
void resolver_default_op_ConvTranspose(onnx_node_t *n);
void resolver_default_op_Cos(onnx_node_t *n);
void resolver_default_op_Cosh(onnx_node_t *n);
void resolver_default_op_CumSum(onnx_node_t *n);
void resolver_default_op_DepthToSpace(onnx_node_t *n);
void resolver_default_op_DequantizeLinear(onnx_node_t *n);
void resolver_default_op_Det(onnx_node_t *n);
void resolver_default_op_Div(onnx_node_t *n);
void resolver_default_op_Dropout(onnx_node_t *n);
void resolver_default_op_Einsum(onnx_node_t *n);
void resolver_default_op_Elu(onnx_node_t *n);
void resolver_default_op_Equal(onnx_node_t *n);
void resolver_default_op_Erf(onnx_node_t *n);
void resolver_default_op_Exp(onnx_node_t *n);
void resolver_default_op_Expand(onnx_node_t *n);
void resolver_default_op_EyeLike(onnx_node_t *n);
void resolver_default_op_Flatten(onnx_node_t *n);
void resolver_default_op_Floor(onnx_node_t *n);
void resolver_default_op_GRU(onnx_node_t *n);
void resolver_default_op_Gather(onnx_node_t *n);
void resolver_default_op_GatherElements(onnx_node_t *n);
void resolver_default_op_GatherND(onnx_node_t *n);
void resolver_default_op_Gemm(onnx_node_t *n);
void resolver_default_op_GlobalAveragePool(onnx_node_t *n);
void resolver_default_op_GlobalLpPool(onnx_node_t *n);
void resolver_default_op_GlobalMaxPool(onnx_node_t *n);
void resolver_default_op_Greater(onnx_node_t *n);
void resolver_default_op_HardSigmoid(onnx_node_t *n);
void resolver_default_op_Hardmax(onnx_node_t *n);
void resolver_default_op_Identity(onnx_node_t *n);
void resolver_default_op_If(onnx_node_t *n);
void resolver_default_op_InstanceNormalization(onnx_node_t *n);
void resolver_default_op_IsInf(onnx_node_t *n);
void resolver_default_op_IsNaN(onnx_node_t *n);
void resolver_default_op_LRN(onnx_node_t *n);
void resolver_default_op_LSTM(onnx_node_t *n);
void resolver_default_op_LeakyRelu(onnx_node_t *n);
void resolver_default_op_Less(onnx_node_t *n);
void resolver_default_op_Log(onnx_node_t *n);
void resolver_default_op_Loop(onnx_node_t *n);
void resolver_default_op_LpNormalization(onnx_node_t *n);
void resolver_default_op_LpPool(onnx_node_t *n);
void resolver_default_op_MatMul(onnx_node_t *n);
void resolver_default_op_MatMulInteger(onnx_node_t *n);
void resolver_default_op_Max(onnx_node_t *n);
void resolver_default_op_MaxPool(onnx_node_t *n);
void resolver_default_op_MaxRoiPool(onnx_node_t *n);
void resolver_default_op_MaxUnpool(onnx_node_t *n);
void resolver_default_op_Mean(onnx_node_t *n);
void resolver_default_op_Min(onnx_node_t *n);
void resolver_default_op_Mod(onnx_node_t *n);
void resolver_default_op_Mul(onnx_node_t *n);
void resolver_default_op_Multinomial(onnx_node_t *n);
void resolver_default_op_Neg(onnx_node_t *n);
void resolver_default_op_NonMaxSuppression(onnx_node_t *n);
void resolver_default_op_NonZero(onnx_node_t *n);
void resolver_default_op_Not(onnx_node_t *n);
void resolver_default_op_OneHot(onnx_node_t *n);
void resolver_default_op_Or(onnx_node_t *n);
void resolver_default_op_PRelu(onnx_node_t *n);
void resolver_default_op_Pad(onnx_node_t *n);
void resolver_default_op_Pow(onnx_node_t *n);
void resolver_default_op_QLinearConv(onnx_node_t *n);
void resolver_default_op_QLinearMatMul(onnx_node_t *n);
void resolver_default_op_QuantizeLinear(onnx_node_t *n);
void resolver_default_op_RNN(onnx_node_t *n);
void resolver_default_op_RandomNormal(onnx_node_t *n);
void resolver_default_op_RandomNormalLike(onnx_node_t *n);
void resolver_default_op_RandomUniform(onnx_node_t *n);
void resolver_default_op_RandomUniformLike(onnx_node_t *n);
void resolver_default_op_Reciprocal(onnx_node_t *n);
void resolver_default_op_ReduceL1(onnx_node_t *n);
void resolver_default_op_ReduceL2(onnx_node_t *n);
void resolver_default_op_ReduceLogSum(onnx_node_t *n);
void resolver_default_op_ReduceLogSumExp(onnx_node_t *n);
void resolver_default_op_ReduceMax(onnx_node_t *n);
void resolver_default_op_ReduceMean(onnx_node_t *n);
void resolver_default_op_ReduceMin(onnx_node_t *n);
void resolver_default_op_ReduceProd(onnx_node_t *n);
void resolver_default_op_ReduceSum(onnx_node_t *n);
void resolver_default_op_ReduceSumSquare(onnx_node_t *n);
void resolver_default_op_Relu(onnx_node_t *n);
void resolver_default_op_Reshape(onnx_node_t *n);
void resolver_default_op_Resize(onnx_node_t *n);
void resolver_default_op_ReverseSequence(onnx_node_t *n);
void resolver_default_op_RoiAlign(onnx_node_t *n);
void resolver_default_op_Round(onnx_node_t *n);
void resolver_default_op_Scan(onnx_node_t *n);
void resolver_default_op_Scatter(onnx_node_t *n);
void resolver_default_op_ScatterElements(onnx_node_t *n);
void resolver_default_op_ScatterND(onnx_node_t *n);
void resolver_default_op_Selu(onnx_node_t *n);
void resolver_default_op_SequenceAt(onnx_node_t *n);
void resolver_default_op_SequenceConstruct(onnx_node_t *n);
void resolver_default_op_SequenceEmpty(onnx_node_t *n);
void resolver_default_op_SequenceErase(onnx_node_t *n);
void resolver_default_op_SequenceInsert(onnx_node_t *n);
void resolver_default_op_SequenceLength(onnx_node_t *n);
void resolver_default_op_Shape(onnx_node_t *n);
void resolver_default_op_Shrink(onnx_node_t *n);
void resolver_default_op_Sigmoid(onnx_node_t *n);
void resolver_default_op_Sign(onnx_node_t *n);
void resolver_default_op_Sin(onnx_node_t *n);
void resolver_default_op_Sinh(onnx_node_t *n);
void resolver_default_op_Size(onnx_node_t *n);
void resolver_default_op_Slice(onnx_node_t *n);
void resolver_default_op_Softplus(onnx_node_t *n);
void resolver_default_op_Softsign(onnx_node_t *n);
void resolver_default_op_SpaceToDepth(onnx_node_t *n);
void resolver_default_op_Split(onnx_node_t *n);
void resolver_default_op_SplitToSequence(onnx_node_t *n);
void resolver_default_op_Sqrt(onnx_node_t *n);
void resolver_default_op_Squeeze(onnx_node_t *n);
void resolver_default_op_StringNormalizer(onnx_node_t *n);
void resolver_default_op_Sub(onnx_node_t *n);
void resolver_default_op_Sum(onnx_node_t *n);
void resolver_default_op_Tan(onnx_node_t *n);
void resolver_default_op_Tanh(onnx_node_t *n);
void resolver_default_op_TfIdfVectorizer(onnx_node_t *n);
void resolver_default_op_ThresholdedRelu(onnx_node_t *n);
void resolver_default_op_Tile(onnx_node_t *n);
void resolver_default_op_TopK(onnx_node_t *n);
void resolver_default_op_Transpose(onnx_node_t *n);
void resolver_default_op_Trilu(onnx_node_t *n);
void resolver_default_op_Unique(onnx_node_t *n);
void resolver_default_op_Unsqueeze(onnx_node_t *n);
void resolver_default_op_Upsample(onnx_node_t *n);
void resolver_default_op_Where(onnx_node_t *n);
void resolver_default_op_Xor(onnx_node_t *n);

void resolver_default_op_Celu(onnx_node_t *n);
void resolver_default_op_DynamicQuantizeLinear(onnx_node_t *n);
void resolver_default_op_GreaterOrEqual(onnx_node_t *n);
void resolver_default_op_HardSwish(onnx_node_t *n);
void resolver_default_op_LessOrEqual(onnx_node_t *n);
void resolver_default_op_LogSoftmax(onnx_node_t *n);
void resolver_default_op_MeanVarianceNormalization(onnx_node_t *n);
void resolver_default_op_NegativeLogLikelihoodLoss(onnx_node_t *n);
void resolver_default_op_Range(onnx_node_t *n);
void resolver_default_op_Softmax(onnx_node_t *n);
void resolver_default_op_SoftmaxCrossEntropyLoss(onnx_node_t *n);

extern onnx_resolver_t resolver_default;

#ifdef __cplusplus
}
#endif

#endif /* __DEFAULT_H__ */
