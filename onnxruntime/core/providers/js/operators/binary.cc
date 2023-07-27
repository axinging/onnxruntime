// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

#define REG_ELEMENTWISE_KERNEL(OP_TYPE, VERSION, TYPE, KERNEL_CLASS)               \
  ONNX_OPERATOR_KERNEL_EX(                                                         \
      OP_TYPE,                                                                     \
      kOnnxDomain,                                                                 \
      VERSION,                                                                     \
      kJsExecutionProvider,                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>()), \
      KERNEL_CLASS);

#define REG_ELEMENTWISE_TYPED_KERNEL(OP_TYPE, VERSION, TYPE, KERNEL_CLASS)               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                         \
      OP_TYPE,                                                                     \
      kOnnxDomain,                                                                 \
      VERSION,                                                                     \
      TYPE,                                                                                     \
      kJsExecutionProvider,                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>()), \
      KERNEL_CLASS);

#define REG_ELEMENTWISE_VERSIONED_KERNEL(OP_TYPE, VERSION_FROM, VERSION_TO, TYPE, KERNEL_CLASS) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                            \
      OP_TYPE,                                                                                  \
      kOnnxDomain,                                                                              \
      VERSION_FROM, VERSION_TO,                                                                 \
      kJsExecutionProvider,                                                                     \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>()),              \
      KERNEL_CLASS);

#define REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(OP_TYPE, VERSION_FROM, VERSION_TO, TYPE, KERNEL_CLASS) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                            \
      OP_TYPE,                                                                                  \
      kOnnxDomain,                                                                              \
      VERSION_FROM, VERSION_TO,                                                                 \
      TYPE,                                                                                     \
      kJsExecutionProvider,                                                                     \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>()),              \
      KERNEL_CLASS);

JSEP_KERNEL_IMPL(Add, Add)
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Add, 7, 12, float, Add);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Add, 13, 13, float, Add);
REG_ELEMENTWISE_TYPED_KERNEL(Add, 14, float, Add);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Add, 7, 12, int32_t, Add);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Add, 13, 13, int32_t, Add);
REG_ELEMENTWISE_TYPED_KERNEL(Add, 14, int32_t, Add);

JSEP_KERNEL_IMPL(Sub, Sub)
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Sub, 7, 12, float, Sub);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Sub, 13, 13, float, Sub);
REG_ELEMENTWISE_TYPED_KERNEL(Sub, 14, float, Sub);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Sub, 7, 12, int32_t, Sub);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Sub, 13, 13, int32_t, Sub);
REG_ELEMENTWISE_TYPED_KERNEL(Sub, 14, int32_t, Sub);

JSEP_KERNEL_IMPL(Mul, Mul)
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Mul, 7, 12, float, Mul);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Mul, 13, 13, float, Mul);
REG_ELEMENTWISE_TYPED_KERNEL(Mul, 14, float, Mul);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Mul, 7, 12, int32_t, Mul);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Mul, 13, 13, int32_t, Mul);
REG_ELEMENTWISE_TYPED_KERNEL(Mul, 14, int32_t, Mul);

JSEP_KERNEL_IMPL(Div, Div)
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Div, 7, 12, float, Div);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Div, 13, 13, float, Div);
REG_ELEMENTWISE_TYPED_KERNEL(Div, 14, float, Div);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Div, 7, 12, int32_t, Div);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Div, 13, 13, int32_t, Div);
REG_ELEMENTWISE_TYPED_KERNEL(Div, 14, int32_t, Div);

JSEP_KERNEL_IMPL(Pow, Pow)
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Pow, 7, 11, float, Pow);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Pow, 12, 12, float, Pow);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Pow, 13, 14, float, Pow);
REG_ELEMENTWISE_TYPED_KERNEL(Pow, 15, float, Pow);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Pow, 7, 11, int32_t, Pow);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Pow, 12, 12, int32_t, Pow);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Pow, 13, 14, int32_t, Pow);
REG_ELEMENTWISE_TYPED_KERNEL(Pow, 15, int32_t, Pow);

}  // namespace js
}  // namespace onnxruntime
