// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor-view';
import {ShapeUtil} from '../../util';
import {createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType, ProgramUniform} from '../types';

import {inputVariable, outputVariable, ShaderHelper, UniformsArrayType} from './common';
import {createExpandProgramInfo} from './expand';
import {applyAttention, AttentionMaskType, AttentionParameters, AttentionQkvFormat, GroupQueryAttentionAttrs} from './group_query_attention_util';
import {createTransposeProgramInfo, TransposeAttributes} from './transpose';

const validateInputs = (inputs: readonly TensorView[], attributes: GroupQueryAttentionAttrs): AttentionParameters => {
  const query = inputs[0];
  const key = inputs[1];
  const value = inputs[2];
  const bias = inputs[3];
  const keyPaddingMask = inputs[4];
  const relativePositionBias = inputs[5];
  const pastKey = inputs[6];
  const pastValue = inputs[7];

  // Abbreviation and Meanings:
  //   B:    batch_size
  //   S:    sequence_length (input sequence length of query)
  //   P:    past_sequence_length (past sequence length of key or value)
  //   L:    kv_sequence_length (input sequence length of key or value)
  //   M:    max_sequence_length
  //   T:    total_sequence_length = past_sequence_length + kv_sequence_length
  //   N:    num_heads
  //   H:    head size for Q and K, aka q_head_size or k_head_size or qk_head_size
  //   H_v:  v_head_size
  //   D_i:  input hidden size
  //   D:    hidden size for Q and K (D = N * H), aka q_hidden_size or k_hidden_size or qk_hidden_size
  //   D_v:  v_hidden_size = num_heads * v_head_size

  //     key_padding_mask (K/V)     : (B) or (2*B + 1) or (B, L) or None
  //     relative_position_bias     : (B, 1, S, L)
  //     past_key                   : (B, N, S*, H)
  //     past_value                 : (B, N, S*, H)
  // When no packing for q/k/v:
  //     query            (Q)       : (B, S, D)
  //     key              (K)       : (B, L, D) or (B, N, S*, H)
  //     value            (V)       : (B, L, D_v) or (B, N, S*, H)
  //     bias             (Q/K/V)   : (D + D + D_v)
  // When packed kv is used:
  //     query            (Q)       : (B, S, D)
  //     key              (K)       : (B, L, N, 2, H)
  //     value            (V)       : None
  //     bias             (Q/K/V)   : None
  // When packed qkv is used:
  //     query            (Q)       : (B, L, N, 3, H) or (B, S, 3*D)
  //     key              (K)       : None
  //     value            (V)       : None
  //     bias             (Q/K/V)   : None or (D + D + D_v)

  if (query.dims.length !== 3 && query.dims.length !== 5) {
    throw new Error('Input query is expected to have 3 or 5 dimensions');
  }

  const dmmhaPacking = false;
  const batchSize = query.dims[0];
  const sequenceLength = query.dims[1];
  const hiddenSize = query.dims.length === 3 ? (dmmhaPacking ? query.dims[2] / 3 : query.dims[2]) :
                                               attributes.numHeads * query.dims[4];
  let kvSequenceLength = sequenceLength;

  let pastSequenceLength = 0;
  let maxSequenceLength = 0;
  const headSize = Math.floor(hiddenSize / attributes.numHeads);
  if (pastKey && pastValue) {
    if (pastKey.dims.length !== 4) {
      throw new Error('Input "past_key" is expected to have 4 dimensions');
    }
    if (pastValue.dims.length !== 4) {
      throw new Error('Input "past_value" is expected to have 4 dimensions');
    }
    pastSequenceLength = pastKey.dims[2];
    maxSequenceLength = pastKey.dims[2];
  } else if (pastKey || pastValue) {
    throw new Error('Input "past_key" and "past_value" shall be both present or both absent');
  }

  let qkvFormat: AttentionQkvFormat;
  if (key) {
    if (query.dims.length !== 3) {
      throw new Error('Input "query" is expected to have 3 dimensions when key is given');
    }
    if (key.dims.length < 3 || key.dims.length > 5) {
      console.warn('Input "key" is expected to have 3, 4, or 5 dimensions');
    }
    if (query.dims[0] !== key.dims[0]) {
      console.warn('Input "query" and "key" shall have same dim 0 (batch size)');
    }

    if (key.dims.length === 3) {
      if (key.dims[2] !== query.dims[2]) {
        console.warn('Input "query" and "key" shall have same dim 2 (hidden_size)');
      }
      qkvFormat = AttentionQkvFormat.qkvBSNH;
      kvSequenceLength = key.dims[1];
    } else if (key.dims.length === 5) {
      if (key.dims[2] !== attributes.numHeads || key.dims[3] !== 2 || key.dims[4] !== headSize) {
        throw new Error('Expect "key" shape (batch_size, kv_sequence_length, num_heads, 2, head_size) for packed kv');
      }
      if (value) {
        throw new Error('Expect "value" be none when "key" has packed kv format.');
      }
      qkvFormat = AttentionQkvFormat.qKvBSNHxBSN2H;
      kvSequenceLength = key.dims[1];
    } else {  // key_dims.size() == 4 (cross-attention with past_key)
      if (key.dims[1] !== attributes.numHeads || key.dims[3] !== headSize) {
        throw new Error('Expect "key" shape (batch_size, num_heads, kv_sequence_length, head_size) for past_key');
      }

      qkvFormat = AttentionQkvFormat.unknown;
      kvSequenceLength = key.dims[2];
    }
  } else {  // packed QKV
    if (query.dims.length !== 3 && query.dims.length !== 5) {
      throw new Error('Input "query" is expected to have 3 or 5 dimensions when key is empty');
    }
    if (query.dims.length === 5 && (query.dims[2] !== attributes.numHeads || query.dims[3] !== 3)) {
      throw new Error('Expect "query" shape (batch_size, kv_sequence_length, num_heads, 3, head_size) for packed kv');
    }

    qkvFormat = AttentionQkvFormat.qkvBSN3H;
  }

  if (bias) {
    if (bias.dims.length !== 1) {
      throw new Error('Input "bias" is expected to have 1 dimension');
    }

    if (value) {
      if (query.dims.length === 5 && query.dims[3] === 2) {
        throw new Error('bias is not allowed for packed kv.');
      }
    }
  }

  let maskType: AttentionMaskType = AttentionMaskType.none;
  if (keyPaddingMask) {
    maskType = AttentionMaskType.maskUnknown;
    const maskDims = keyPaddingMask.dims;
    if (maskDims.length === 1) {
      if (maskDims[0] === batchSize) {
        maskType = AttentionMaskType.mask1dKeySeqLen;
      } else if (maskDims[0] === 3 * batchSize + 2) {
        maskType = AttentionMaskType.mask1DKeySeqLenStart;
      }
    } else if (maskDims.length === 2 && maskDims[0] === batchSize && maskDims[1] === kvSequenceLength) {
      maskType = AttentionMaskType.mask2dKeyPadding;
    }
    if (maskType === AttentionMaskType.maskUnknown) {
      throw new Error('Input "key_padding_mask" shape shall be (batch_size) or (batch_size, kv_sequence_length)');
    }
    throw new Error('Mask not supported');
  }

  let passPastInKv = false;
  let vHiddenSize = hiddenSize;
  if (value) {
    if (value.dims.length !== 3 && value.dims.length !== 4) {
      throw new Error('Input "value" is expected to have 3 or 4 dimensions');
    }

    if (query.dims[0] !== value.dims[0]) {
      throw new Error('Input "query" and "value" shall have same dim 0 (batch_size)');
    }

    if (value.dims.length === 3) {
      if (kvSequenceLength !== value.dims[1]) {
        throw new Error('Input "key" and "value" shall have the same dim 1 (kv_sequence_length)');
      }
      vHiddenSize = value.dims[2];
    } else {
      if (kvSequenceLength !== value.dims[2]) {
        throw new Error('Input "past_key" and "past_value" shall have the same dim 2 (kv_sequence_length)');
      }
      vHiddenSize = value.dims[1] * value.dims[3];
      passPastInKv = true;
    }
  }

  const totalSequenceLength = pastSequenceLength + kvSequenceLength;
  const broadcastResPosBias = false;
  // if (extraAddQk) {
  //   if (extraAddQk.dims[0] === 1) {
  //     broadcastResPosBias = true;
  //   }
  // }

  if (keyPaddingMask) {
    throw new Error('Key padding mask is not supported');
  }
  if (relativePositionBias) {
    throw new Error('extraAddQk is not supported');
  }
  if (pastKey) {
    throw new Error('pastKey is not supported');
  }
  if (pastValue) {
    throw new Error('pastValue is not supported');
  }
  const kvNumHeads = 2;
  return {
    batchSize,
    sequenceLength,
    pastSequenceLength,
    kvSequenceLength,
    totalSequenceLength,
    maxSequenceLength,
    inputHiddenSize: 0,
    hiddenSize,
    vHiddenSize,
    headSize,
    vHeadSize: Math.floor(vHiddenSize / kvNumHeads),  //???
    numHeads: attributes.numHeads,
    kvNumHeads: kvNumHeads,  // attributes.kvNumHeads,
    isUnidirectional: false,
    pastPresentShareBuffer: false,
    maskFilterValue: attributes.maskFilterValue,
    maskType,
    scale: attributes.scale,
    broadcastResPosBias,
    passPastInKv,
    qkvFormat,
  };
};

export const parseGroupQueryAttentionAttributes = (attributes: GroupQueryAttentionAttrs): GroupQueryAttentionAttrs =>
    createAttributeWithCacheKey({...attributes});

const weightTransposeAttribute: TransposeAttributes = createAttributeWithCacheKey({perm: [0, 2, 1, 3]});

const addBiasTranspose =
    (context: ComputeContext, qkv: TensorView, bias: TensorView, batchSize: number, sequenceLength: number,
     hiddenSize: number, biasOffset: number) => {
      const outputShape = [batchSize, sequenceLength, hiddenSize];
      const outputSize = ShapeUtil.size(outputShape);
      const programUniforms: ProgramUniform[] = [
        {type: DataType.uint32, data: outputSize}, {type: DataType.uint32, data: biasOffset},
        {type: DataType.uint32, data: hiddenSize}
      ];

      const getShaderSource = (shaderHelper: ShaderHelper) => {
        const output = outputVariable('qkv_with_bias', qkv.dataType, outputShape);
        const qkvInput = inputVariable('qkv', qkv.dataType, outputShape);
        const biasInput = inputVariable('bias', bias.dataType, outputShape);

        const uniforms: UniformsArrayType = [
          {name: 'output_size', type: 'u32'}, {name: 'bias_offset', type: 'u32'}, {name: 'hidden_size', type: 'u32'}
        ];
        return `
  ${shaderHelper.registerUniforms(uniforms).declareVariables(qkvInput, biasInput, output)}
  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes('uniforms.output_size')}
    let bias_offset_idx = (global_idx % uniforms.hidden_size) + uniforms.bias_offset;

    qkv_with_bias[global_idx] = qkv[global_idx] + bias[bias_offset_idx];
  }`;
      };

      return context.compute(
          {
            name: 'GroupQueryAttentionAddBias',
            shaderCache: {inputDependencies: ['type', 'type']},
            getRunData: () => ({
              outputs: [{dims: outputShape, dataType: qkv.dataType, gpuDataType: GpuDataType.default}],
              dispatchGroup: {x: Math.ceil(outputSize / 64 /* workgroup size */)},
              programUniforms
            }),
            getShaderSource,
          },
          {inputs: [qkv, bias], outputs: [-1]})[0];
    };

const maybeTransposeToBNSHAndAddBias =
    (context: ComputeContext, batchSize: number, numHeads: number, sequenceLength: number, headSize: number,
     input: TensorView, bias?: TensorView, biasOffset?: number) => {
      // const newDims = [];
      console.log(
          'xxx  batchSize, sequenceLength, numHeads, headSize = ' + [batchSize, sequenceLength, numHeads, headSize] +
          ', input dim=' + input.dims);
      let reshapedInput = input;
      if (!bias) {
        if (input.dims.length === 3) {
          reshapedInput = input.reshape([batchSize, sequenceLength, numHeads, headSize]);
        }

        return context.compute(
            createTransposeProgramInfo(reshapedInput, weightTransposeAttribute.perm),
            {inputs: [reshapedInput], outputs: [-1]})[0];
      } else {
        if (sequenceLength === 1) {
          throw new Error('AddBiasReshape is not implemented. Please export your model with packed QKV or KV');
        } else {
          reshapedInput =
              addBiasTranspose(context, input, bias, batchSize, sequenceLength, numHeads * headSize, biasOffset!);
          reshapedInput = reshapedInput.reshape([batchSize, sequenceLength, numHeads, headSize]);
          return context.compute(
              createTransposeProgramInfo(reshapedInput, weightTransposeAttribute.perm),
              {inputs: [reshapedInput], outputs: [-1]})[0];
        }
      }
    };

const maybeTransposeToBNSHAndAddBias2 =
    (context: ComputeContext, batchSize: number, numHeads: number, sequenceLength: number, headSize: number,
     input: TensorView, nReps: number) => {
      // const newDims = [];
      console.log('xxx  input dim=' + input.dims);
      let reshapedInput = input;
      if (input.dims.length === 3) {
        reshapedInput = input.reshape([batchSize, sequenceLength, numHeads, headSize]);
      }
      console.log(
          'xxx  reshapedInput.dims = ' + reshapedInput.dims +
          ', [batchSize, sequenceLength, numHeads, headSize] = ' + [batchSize, sequenceLength, numHeads, headSize]);
      const expanedInput = context.compute(
          createExpandProgramInfo([reshapedInput], [batchSize, sequenceLength, numHeads, nReps, headSize]),
          {inputs: [reshapedInput], outputs: [-1]})[0];
      console.log('xxx  expanedInput.dims = ' + expanedInput.dims);
      const expanedInput2 = expanedInput.reshape([batchSize, sequenceLength, numHeads * nReps, headSize]);
      console.log('xxx  reshape expanedInput2.dims = ' + expanedInput2.dims);
      return context.compute(
          createTransposeProgramInfo(expanedInput2, weightTransposeAttribute.perm),
          {inputs: [expanedInput2], outputs: [-1]})[0];
    };

export const groupQueryAttention = (context: ComputeContext, attributes: GroupQueryAttentionAttrs): void => {
  const params = validateInputs(context.inputs, attributes);

  if (context.inputs[0].dims.length === 5) {
    throw new Error('Packed QKV is not implemented');
  }

  if (context.inputs[1]?.dims.length === 5) {
    throw new Error('Packed KV is not implemented');
  }

  // applyAttention expects BNSH inputs
  const kvBNSH = context.inputs[1] && context.inputs[2] && context.inputs[1].dims.length === 4 &&
      context.inputs[2].dims.length === 4;

  const Q = maybeTransposeToBNSHAndAddBias(
      context, params.batchSize, params.numHeads, params.sequenceLength, params.headSize, context.inputs[0],
      context.inputs[3], 0);
  console.log('xxx ' + kvBNSH);

  if (kvBNSH) {
    return applyAttention(
        context, Q, context.inputs[1], context.inputs[2], context.inputs[4], undefined, undefined, undefined,
        context.inputs[5], params, attributes);
  }

  const nRep = Math.floor(attributes.numHeads / params.kvNumHeads);
  const K = maybeTransposeToBNSHAndAddBias2(
      context, params.batchSize, params.kvNumHeads, params.kvSequenceLength, params.vHeadSize, context.inputs[1], nRep);

  const V = maybeTransposeToBNSHAndAddBias2(
      context, params.batchSize, params.kvNumHeads, params.kvSequenceLength, params.vHeadSize, context.inputs[2], nRep);
  console.log('xxx Q.dims = ' + Q.dims + ' K.dims = ' + K.dims + '  V.dims = ' + V.dims);
  console.log('xxx params = ' + JSON.stringify(params));
  applyAttention(
      context, Q, K, V, context.inputs[4], undefined, context.inputs[6], context.inputs[7], context.inputs[5], params,
      attributes);
};
