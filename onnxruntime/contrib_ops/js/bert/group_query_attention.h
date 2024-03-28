// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cpu/bert/attention_base.h"
#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace js {

using onnxruntime::contrib::AttentionBase;
using onnxruntime::js::JsKernel;

class GroupQueryAttention : public JsKernel, AttentionBase {
 public:
  explicit GroupQueryAttention(const OpKernelInfo& info) : JsKernel(info), AttentionBase(info, false) {
    JSEP_INIT_KERNEL_ATTRIBUTE(GroupQueryAttention, ({
                                 "numHeads" : $1,
                                 "kvNumHeads" : $2,
                               }),
                               static_cast<int32_t>(num_heads_),
                               static_cast<int32_t>(kv_num_heads_));
  }
};

}  // namespace js
}  // namespace contrib
}  // namespace onnxruntime
