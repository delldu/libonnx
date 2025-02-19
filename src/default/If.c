#include <onnx.h>

typedef struct {
  onnx_graph_t *else_branch;
  onnx_graph_t *then_branch;
} operator_pdata_t;

static int If_init(onnx_node_t *n) {
  operator_pdata_t *pdat;

  if ((n->ninput == 1) && (n->noutput >= 1)) {
    pdat = malloc(sizeof(operator_pdata_t));
    if (pdat) {
      pdat->else_branch = onnx_graph_alloc(
          n->ctx, onnx_attribute_read_graph(n, "else_branch", NULL));
      pdat->then_branch = onnx_graph_alloc(
          n->ctx, onnx_attribute_read_graph(n, "then_branch", NULL));
      if (!pdat->else_branch || !pdat->then_branch) {
        if (pdat->else_branch)
          onnx_graph_free(pdat->else_branch);
        if (pdat->then_branch)
          onnx_graph_free(pdat->then_branch);
        free(pdat);
        return 0;
      }
      n->priv = pdat;
      return 1;
    }
  }
  return 0;
}

static int If_exit(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;

  if (pdat) {
    if (pdat->else_branch)
      onnx_graph_free(pdat->else_branch);
    if (pdat->then_branch)
      onnx_graph_free(pdat->then_branch);
    free(pdat);
  }
  return 1;
}

static int If_reshape(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;
  onnx_tensor_t *x = n->inputs[0];
  uint8_t *px = (uint8_t *)x->datas;
  onnx_graph_t *g;
  onnx_node_t *t;
  int i;

  if (px[0])
    g = pdat->then_branch;
  else
    g = pdat->else_branch;
  if (g->nlen > 0) {
    for (i = 0; i < g->nlen; i++) {
      t = &g->nodes[i];
      t->reshape(t);
    }
    if (t) {
      for (i = 0; i < min(t->noutput, n->noutput); i++) {
        onnx_tensor_t *a = t->outputs[i];
        onnx_tensor_t *b = n->outputs[i];
        onnx_tensor_reshape_identity(b, a, a->type);
      }
    }
  }
  return 1;
}

static void If_operator(onnx_node_t *n) {
  operator_pdata_t *pdat = (operator_pdata_t *)n->priv;
  onnx_tensor_t *x = n->inputs[0];
  uint8_t *px = (uint8_t *)x->datas;
  onnx_graph_t *g;
  onnx_node_t *t;
  int i;

  if (px[0])
    g = pdat->then_branch;
  else
    g = pdat->else_branch;
  if (g->nlen > 0) {
    for (i = 0; i < g->nlen; i++) {
      t = &g->nodes[i];
      t->operator(t);
    }
    if (t) {
      for (i = 0; i < min(t->noutput, n->noutput); i++) {
        onnx_tensor_t *a = t->outputs[i];
        onnx_tensor_t *b = n->outputs[i];
        if (x->type == ONNX_TENSOR_TYPE_STRING) {
          char **pa = (char **)a->datas;
          char **pb = (char **)b->datas;
          for (size_t o = 0; o < b->ndata; o++) {
            if (pb[o])
              free(pb[o]);
            pb[o] = strdup(pa[o]);
          }
        } else {
          memcpy(b->datas, a->datas,
                 a->ndata * onnx_tensor_type_sizeof(a->type));
        }
      }
    }
  }
}

void resolver_default_op_If(onnx_node_t *n) {
  n->init = If_init;
  n->exit = If_exit;
  n->reshape = If_reshape;

  if (n->opset >= 13) {
    n->operator= If_operator;
  } else if (n->opset >= 11) {
    n->operator= If_operator;
  } else if (n->opset >= 1) {
    n->operator= If_operator;
  }
}
