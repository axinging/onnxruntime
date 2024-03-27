// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor-view';
import {ComputeContext, GpuDataType, ProgramUniform} from '../types';

import {castToF32, fillVector, getMaxComponents, inputVariable, outputVariable, ShaderHelper, sumVector, tensorTypeToWsglStorageType, tensorTypeToWsglValueType, UniformDataElementType, UniformsArrayType} from './common';

export const enum AttentionQkvFormat {
  unknown,          // enum value not set, or depends on qkv projection implementation details
  qkvBNSH,          // for non-packed qkv, permuted
  qkvBSNH,          // for non-packed qkv, not permuted, used by memory efficient attention or MultiHeadAttention
  qkvBSN3H,         // for TRT fused attention, qkv are packed
  qkvBNSHqkvBS3NH,  // for TRT fused causal attention, data has two formats (qkv is 3BNSH, gemm_buffer is BS3NH)
  qKvBSNHxBSN2H,    // for TRT fused cross attention, kv are packed
  qkvTNH,           // for memory efficient attention, qkv are not packed, and paddings are removed.
  qkvTN3H,          // for TRT fused attention, qkv are packed and paddings are removed
}

export const enum AttentionMaskType {
  none,                  // No mask
  mask1dKeySeqLen,       // [batch_size], key sequence length
  mask1dEndStart,        // [2 * batch_size] with end positions and start positions
  mask1DKeySeqLenStart,  // [3 * batch_size + 2] with [key_len[0], ..., key_len[batch_size - 1], query_start[0],
                         // ..., query_start[batch_size - 1], query_end[batch_size - 1], key_start[0], ...,
                         // key_start[batch_size - 1], key_end[batch_size - 1]]
  mask2dDummy,           // dummy mask with shape [1, 1] or [batch_size, 1]. It has same effect as no mask.
  mask2dKeyPadding,      // [batch_size, total_sequence_length]
  mask3dAttention,       // [batch_size, sequence_length, total_sequence_length]
  mask4dMegatron,        // Megatron causal mask with shape [batch_size, 1, max_sequence_length, max_sequence_length]
  maskUnknown
}

export interface AttentionParameters {
  batchSize: number;
  sequenceLength: number;
  pastSequenceLength: number;
  kvSequenceLength: number;
  totalSequenceLength: number;
  maxSequenceLength: number;
  inputHiddenSize: number;
  hiddenSize: number;
  vHiddenSize: number;
  headSize: number;
  vHeadSize: number;
  numHeads: number;
  kvNumHeads: number
  isUnidirectional: boolean;
  pastPresentShareBuffer: boolean;
  maskFilterValue: number;
  maskType: AttentionMaskType;
  scale: number;
  broadcastResPosBias: boolean;
  passPastInKv: boolean;
  qkvFormat: AttentionQkvFormat;
}

export interface GroupQueryAttentionAttrs {
  numHeads: number;
  kvNumHeads: number
  isUnidirectional: number;
  maskFilterValue: number;
  scale: number;
  doRotary: number;
  qkvHiddenSizes: number[];
  pastPresentShareBuffer: boolean;
}

export const computeInPlaceSoftmax = (context: ComputeContext, input: TensorView, n: number, d: number) => {
  const components = getMaxComponents(d);
  let WG = 64;
  const dComp = d / components;
  if (dComp < WG) {
    WG = 1;
  } else if (dComp / 8 < 64) {
    WG = Math.ceil(dComp / 8);
  }
  const elementsPerWG = Math.ceil(d / components / WG);
  const programUniforms: ProgramUniform[] = [
    {type: input.dataType, data: 1 / d}, {type: DataType.uint32, data: dComp},
    {type: DataType.uint32, data: elementsPerWG}
  ];
  const dataType = tensorTypeToWsglStorageType(input.dataType, components);

  const getShaderSource = (shaderHelper: ShaderHelper) => {
    const inputHelper = outputVariable('x', input.dataType, input.dims, components);
    let threadMaxValue = 'thread_max_vector';
    if (components === 2) {
      threadMaxValue = 'max(thread_max_vector.x, thread_max_vector.y)';
    } else if (components === 4) {
      threadMaxValue =
          'max(max(thread_max_vector.x, thread_max_vector.y), max(thread_max_vector.z, thread_max_vector.w))';
    }
    const elemValueType = tensorTypeToWsglValueType(input.dataType);
    const uniforms: UniformsArrayType = [
      {name: 'd_inv', type: elemValueType as UniformDataElementType}, {name: 'd_comp', type: 'u32'},
      {name: 'elements_per_wg', type: 'u32'}
    ];

    return `
  var<workgroup> wgMax: array<f32, ${WG}>;
  var<workgroup> wgSum: array<f32, ${WG}>;
  ${shaderHelper.registerUniforms(uniforms).declareVariables(inputHelper)}
  ${shaderHelper.mainStart([
      WG, 1, 1
    ])}
    let localOffset = local_idx * uniforms.elements_per_wg;
    let offset: u32 = workgroup_id.x * uniforms.d_comp + localOffset;

    var thread_max_vector = ${fillVector('f32', components, '-3.402823e+38f')};
    for (var i: u32 = 0; i < uniforms.elements_per_wg && i + localOffset < uniforms.d_comp; i++) {
      thread_max_vector = max(${castToF32(elemValueType, components, 'x[offset + i]')}, thread_max_vector);
    }
    wgMax[local_idx] = ${threadMaxValue};
    workgroupBarrier();

    var maxValue = -3.402823e+38f;
    for (var i = 0u; i < ${WG}; i++) {
      maxValue = max(wgMax[i], maxValue);
    }

    var sumVector = ${fillVector('f32', components, '0')};
    for (var i: u32 = 0; i < uniforms.elements_per_wg && i + localOffset < uniforms.d_comp; i++) {
      sumVector += exp(${castToF32(elemValueType, components, 'x[offset + i]')} - maxValue);
    }
    wgSum[local_idx] = ${sumVector('sumVector', components)};
    workgroupBarrier();

    var sum: f32 = 0;
    for (var i = 0u; i < ${WG}; i++) {
      sum += wgSum[i];
    }

    if (sum == 0) {
      for (var i: u32 = 0; i < uniforms.elements_per_wg && i + localOffset < uniforms.d_comp; i++) {
        x[offset + i] = ${fillVector(elemValueType, components, 'uniforms.d_inv')};
      }
    } else {
      for (var i: u32 = 0; i < uniforms.elements_per_wg && i + localOffset < uniforms.d_comp; i++) {
        let f32input = ${castToF32(elemValueType, components, 'x[offset + i]')};
        x[offset + i] = ${inputHelper.type.value}(exp(f32input - maxValue) / sum);
      }
    }
  }`;
  };

  context.compute(
      {
        name: 'AttentionProbsSoftmax',
        shaderCache: {hint: `${WG};${dataType};${components}`},
        getShaderSource,
        getRunData: () => ({outputs: [], dispatchGroup: {x: n}, programUniforms}),
      },
      {inputs: [input], outputs: []});
};

const computeAttentionProbs =
    (context: ComputeContext, q: TensorView, key: TensorView, _bias: TensorView|undefined,
     parameters: AttentionParameters, attributes: GroupQueryAttentionAttrs) => {
      const probsShape = [
        parameters.batchSize, parameters.numHeads, parameters.sequenceLength,
        parameters.kvSequenceLength + parameters.pastSequenceLength
      ];
      // TODO: handle mask
      console.log(attributes);

      // const alpha = attributes.scale === 0 ? 1.0 / Math.sqrt(parameters.headSize) : attributes.scale;
      const alpha = 1.0 / Math.sqrt(parameters.headSize);
      const components = getMaxComponents(parameters.headSize);
      const vectorizedHeadSize = parameters.headSize / components;
      const TILE_SIZE = 12;
      const dispatch = {
        x: Math.ceil(parameters.totalSequenceLength / TILE_SIZE),
        y: Math.ceil(parameters.sequenceLength / TILE_SIZE),
        z: parameters.batchSize * parameters.numHeads
      };
      const programUniforms: ProgramUniform[] = [
        {type: DataType.uint32, data: parameters.sequenceLength}, {type: DataType.uint32, data: vectorizedHeadSize},
        {type: DataType.uint32, data: parameters.totalSequenceLength},
        {type: DataType.uint32, data: parameters.kvSequenceLength}, {type: q.dataType, data: alpha}
      ];
      console.log(JSON.stringify(programUniforms));

      const inputs = [q, key];

      const getShaderSource = (shaderHelper: ShaderHelper) => {
        const qInput = inputVariable('q', q.dataType, q.dims, components);
        const kInput = inputVariable('key', key.dataType, key.dims, components);
        const output = outputVariable('output', q.dataType, probsShape);
        const dataType = tensorTypeToWsglStorageType(q.dataType);

        const uniforms: UniformsArrayType = [
          {name: 'M', type: 'u32'}, {name: 'K', type: 'u32'}, {name: 'N', type: 'u32'},
          {name: 'kv_sequence_length', type: 'u32'}, {name: 'alpha', type: dataType as UniformDataElementType}
        ];
        return `
  const beta: ${dataType} = 1.0;
  const TILE_SIZE = ${TILE_SIZE}u;

  var<workgroup> tileQ: array<${qInput.type.storage}, ${TILE_SIZE * TILE_SIZE}>;
  var<workgroup> tileK: array<${qInput.type.storage}, ${TILE_SIZE * TILE_SIZE}>;
  ${shaderHelper.registerUniforms(uniforms).declareVariables(qInput, kInput, output)}
  ${shaderHelper.mainStart([
          TILE_SIZE, TILE_SIZE, 1
        ])}
    // x holds the N and y holds the M
    let headIdx = workgroup_id.z;
    let m = workgroup_id.y * TILE_SIZE;
    let n = workgroup_id.x * TILE_SIZE;
    let lm = global_id.y;
    let ln = global_id.x;

    let qOffset = uniforms.M * uniforms.K * headIdx + m * uniforms.K;
    let kOffset = uniforms.kv_sequence_length * uniforms.K * headIdx + n * uniforms.K;

    var value = ${fillVector(dataType, components)};
    for (var w: u32 = 0u; w < uniforms.K; w += TILE_SIZE) {
      if (m + local_id.y < uniforms.M && w + local_id.x < uniforms.K) {
        tileQ[TILE_SIZE * local_id.y + local_id.x] = q[qOffset + local_id.y * uniforms.K + w + local_id.x];
      }
      if (n + local_id.y < uniforms.N && w + local_id.x < uniforms.K) {
        tileK[TILE_SIZE * local_id.y + local_id.x] = key[kOffset + local_id.y * uniforms.K + w + local_id.x];
      }
      workgroupBarrier();

      for (var k: u32 = 0u; k<TILE_SIZE && w+k < uniforms.K; k++) {
        value += tileQ[TILE_SIZE * local_id.y + k] * tileK[TILE_SIZE * local_id.x + k];
      }

      workgroupBarrier();
    }

    let headOffset = headIdx * uniforms.M * uniforms.N;
    if (lm < uniforms.M && ln < uniforms.N) {
      let outputIdx = headOffset + lm * uniforms.N + ln;
      output[outputIdx] = ${sumVector('value', components)} * uniforms.alpha;
    }
  }`;
      };

      const probs = context.compute(
          {
            name: 'AttentionProbs',
            shaderCache: {hint: `${components}`, inputDependencies: ['type', 'type']},
            getRunData: () => ({
              outputs: [{dims: probsShape, dataType: q.dataType, gpuDataType: GpuDataType.default}],
              dispatchGroup: dispatch,
              programUniforms
            }),
            getShaderSource,
          },
          {inputs, outputs: [-1]})[0];

      computeInPlaceSoftmax(
          context, probs, parameters.batchSize * parameters.numHeads * parameters.sequenceLength,
          parameters.totalSequenceLength);

      return probs;
    };

const computeVxAttentionScore =
    (context: ComputeContext, probs: TensorView, v: TensorView, params: AttentionParameters) => {
      const outputShape = [params.batchSize, params.sequenceLength, params.vHiddenSize];
      const TILE_SIZE = 12;
      const dispatch = {
        x: Math.ceil(params.vHeadSize / TILE_SIZE),
        y: Math.ceil(params.sequenceLength / TILE_SIZE),
        z: params.batchSize * params.numHeads
      };
      const programUniforms: ProgramUniform[] = [
        {type: DataType.uint32, data: params.sequenceLength}, {type: DataType.uint32, data: params.totalSequenceLength},
        {type: DataType.uint32, data: params.vHeadSize}, {type: DataType.uint32, data: params.numHeads},
        {type: DataType.uint32, data: params.vHiddenSize}
      ];
      console.log(JSON.stringify(programUniforms));

      const getShaderSource = (shaderHelper: ShaderHelper) => {
        const probsHelper = inputVariable('probs', probs.dataType, probs.dims);
        const vHelper = inputVariable('v', v.dataType, v.dims);
        const output = outputVariable('output', probs.dataType, outputShape);
        const uniforms: UniformsArrayType = [
          {name: 'M', type: 'u32'}, {name: 'K', type: 'u32'}, {name: 'N', type: 'u32'},
          {name: 'num_heads', type: 'u32'}, {name: 'v_hidden_size', type: 'u32'}
        ];
        return `
  const TILE_SIZE = ${TILE_SIZE}u;
  var<workgroup> tileQ: array<${probsHelper.type.value}, ${TILE_SIZE * TILE_SIZE}>;
  var<workgroup> tileK: array<${probsHelper.type.value}, ${TILE_SIZE * TILE_SIZE}>;
  ${shaderHelper.registerUniforms(uniforms).declareVariables(probsHelper, vHelper, output)}
  ${shaderHelper.mainStart([
          TILE_SIZE, TILE_SIZE, 1
        ])}
   let headIdx = workgroup_id.z;
   let m = workgroup_id.y * TILE_SIZE + local_id.y;
   let n = workgroup_id.x * TILE_SIZE + local_id.x;

   let offsetA = headIdx * (uniforms.M * uniforms.K) + m * uniforms.K;
   let offsetB = headIdx * (uniforms.N * uniforms.K) + n;

   var value = ${probsHelper.type.storage}(0);
   for (var w: u32 = 0u; w < uniforms.K; w += TILE_SIZE) {
     if (m < uniforms.M && w + local_id.x < uniforms.K) {
       tileQ[TILE_SIZE * local_id.y + local_id.x] = probs[offsetA + w + local_id.x];
     }
     if (n < uniforms.N && w + local_id.y < uniforms.K) {
       tileK[TILE_SIZE * local_id.y + local_id.x] = v[offsetB + (w + local_id.y) * uniforms.N];
     }
     workgroupBarrier();
     for (var k: u32 = 0u; k<TILE_SIZE && w+k < uniforms.K; k++) {
       value += tileQ[TILE_SIZE * local_id.y + k] * tileK[TILE_SIZE * k + local_id.x];
     }
     workgroupBarrier();
   }

   // we need to transpose output from BNSH_v to BSND_v
   let batchIdx = workgroup_id.z / uniforms.num_heads;
   let currentBatchHeadNumber = workgroup_id.z % uniforms.num_heads;
   let headOffset = (batchIdx * uniforms.M * uniforms.num_heads + currentBatchHeadNumber) * uniforms.N;
   if (m < uniforms.M && n < uniforms.N) {
     let outputIdx = batchIdx * uniforms.M *uniforms.v_hidden_size + m * uniforms.v_hidden_size
       + currentBatchHeadNumber * uniforms.N + n;
     output[outputIdx] = value;
   }
  }`;
      };

      return context.compute(
          {
            name: 'AttentionScore',
            shaderCache: {inputDependencies: ['type', 'type']},
            getRunData: () => ({
              outputs: [{dims: outputShape, dataType: probs.dataType, gpuDataType: GpuDataType.default}],
              dispatchGroup: dispatch,
              programUniforms
            }),
            getShaderSource,
          },
          {inputs: [probs, v], outputs: [0]})[0];
    };

export const applyAttention =
    (context: ComputeContext, q: TensorView, k: TensorView, v: TensorView, _maskIndex: TensorView|undefined,
     _past: TensorView|undefined, _pastKey: TensorView|undefined, _pastValue: TensorView|undefined,
     relativePositionBias: TensorView|undefined, parameters: AttentionParameters, attributes: GroupQueryAttentionAttrs) => {
      const probs = computeAttentionProbs(context, q, k, relativePositionBias, parameters, attributes);

      computeVxAttentionScore(context, probs, v, parameters);
    };

/*

const prepare = (context: ComputeContext, parameters: AttentionParameters) => {
  const outputShape = [
    parameters.batchSize,
    parameters.numHeads,
    parameters.sequenceLength,
    parameters.headSize,
  ];
  const M = parameters.sequenceLength;
  const K = parameters.inputHiddenSize;
  const N = parameters.headSize;
  const TILE_SIZE = 12;
  const dispatch = {
    x: Math.ceil(parameters.headSize / TILE_SIZE),
    y: Math.ceil(parameters.sequenceLength / TILE_SIZE),
    z: parameters.batchSize * parameters.numHeads
  };
  const inputs = [context.inputs[0], context.inputs[1], context.inputs[2]];
  const programUniforms: ProgramUniform[] = [
    {type: DataType.uint32, data: M}, {type: DataType.uint32, data: K}, {type: DataType.uint32, data: N},
    {type: DataType.uint32, data: parameters.numHeads}, {type: DataType.uint32, data: parameters.headSize},
    {type: DataType.uint32, data: parameters.hiddenSize},
    {type: DataType.uint32, data: parameters.hiddenSize + parameters.hiddenSize + parameters.vHiddenSize}
  ];

  const getShaderSource = (shaderHelper: ShaderHelper) => {
    const outputQ = outputVariable('output_q', inputs[0].dataType, outputShape);
    const outputK = outputVariable('output_k', inputs[0].dataType, outputShape);
    const outputV = outputVariable('output_v', inputs[0].dataType, outputShape);
    const input = inputVariable('input', inputs[0].dataType, inputs[0].dims);
    const weight = inputVariable('weight', inputs[1].dataType, inputs[1].dims);
    const bias = inputVariable('bias', inputs[2].dataType, inputs[2].dims);
    const dataType = input.type.storage;

    const uniforms: UniformsArrayType = [
      {name: 'M', type: 'u32'}, {name: 'K', type: 'u32'}, {name: 'N', type: 'u32'}, {name: 'num_heads', type: 'u32'},
      {name: 'head_size', type: 'u32'}, {name: 'hidden_size', type: 'u32'}, {name: 'ldb', type: 'u32'}
    ];
    return `
  const TILE_SIZE = ${TILE_SIZE}u;
  var<workgroup> tileInput: array<${dataType}, ${TILE_SIZE * TILE_SIZE}>;
  var<workgroup> tileWeightQ: array<${dataType}, ${TILE_SIZE * TILE_SIZE}>;
  var<workgroup> tileWeightK: array<${dataType}, ${TILE_SIZE * TILE_SIZE}>;
  var<workgroup> tileWeightV: array<${dataType}, ${TILE_SIZE * TILE_SIZE}>;
  ${shaderHelper.registerUniforms(uniforms).declareVariables(input, weight, bias, outputQ, outputK, outputV)}
  ${shaderHelper.mainStart([
      TILE_SIZE, TILE_SIZE, 1
    ])}
    let batchIndex = workgroup_id.z / uniforms.num_heads;
    let headNumber = workgroup_id.z % uniforms.num_heads;
    let m = workgroup_id.y * TILE_SIZE + local_id.y;
    let n = workgroup_id.x * TILE_SIZE + local_id.x;

    let inputOffset = batchIndex * (uniforms.M * uniforms.K) + m * uniforms.K;
    let biasOffsetQ = headNumber * uniforms.head_size;
    let biasOffsetK = uniforms.hidden_size + biasOffsetQ;
    let biasOffsetV = uniforms.hidden_size + biasOffsetK;

    var valueQ = ${dataType}(0);
    var valueK = ${dataType}(0);
    var valueV = ${dataType}(0);
    for (var w: u32 = 0u; w < uniforms.K; w += TILE_SIZE) {
      if (m < uniforms.M && w + local_id.x < uniforms.K) {
        tileInput[TILE_SIZE * local_id.y + local_id.x] = input[inputOffset + w + local_id.x];
      }
      if (n < uniforms.N && w + local_id.y < uniforms.K) {
        let offset = n + (w + local_id.y) * uniforms.ldb;
        tileWeightQ[TILE_SIZE * local_id.y + local_id.x] = weight[biasOffsetQ + offset];
        tileWeightK[TILE_SIZE * local_id.y + local_id.x] = weight[biasOffsetK + offset];
        tileWeightV[TILE_SIZE * local_id.y + local_id.x] = weight[biasOffsetV + offset];
      }
      workgroupBarrier();
      for (var k: u32 = 0u; k<TILE_SIZE && w+k < uniforms.K; k++) {
        let inputTileOffset = TILE_SIZE * local_id.y + k;
        let weightTileOffset = TILE_SIZE * k + local_id.x;
        valueQ += tileInput[inputTileOffset] * tileWeightQ[weightTileOffset];
        valueK += tileInput[inputTileOffset] * tileWeightK[weightTileOffset];
        valueV += tileInput[inputTileOffset] * tileWeightV[weightTileOffset];
      }

      workgroupBarrier();
    }

    let headOffset = (m * uniforms.N + n) % uniforms.head_size;
    valueQ += bias[headOffset + biasOffsetQ];
    valueK += bias[headOffset + biasOffsetK];
    valueV += bias[headOffset + biasOffsetV];

    let offset = workgroup_id.z * uniforms.M * uniforms.N;
    if (m < uniforms.M && n < uniforms.N) {
      let outputIdx = offset + m * uniforms.N + n;
      output_q[outputIdx] = valueQ;
      output_k[outputIdx] = valueK;
      output_v[outputIdx] = valueV;
    }
  }`;
  };

  return context.compute(
      {
        name: 'AttentionPrepare',
        shaderCache: {inputDependencies: ['type', 'type', 'type']},
        getRunData: () => ({
          outputs: [
            {dims: outputShape, dataType: context.inputs[0].dataType, gpuDataType: GpuDataType.default},
            {dims: outputShape, dataType: context.inputs[0].dataType, gpuDataType: GpuDataType.default},
            {dims: outputShape, dataType: context.inputs[0].dataType, gpuDataType: GpuDataType.default},
          ],
          dispatchGroup: dispatch,
          programUniforms
        }),
        getShaderSource,
      },
      {inputs, outputs: [-1, -1, -1]});
};

*/
