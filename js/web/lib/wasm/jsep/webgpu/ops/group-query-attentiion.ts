// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor-view';
import {createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext} from '../types';
import {AttentionAttrs, AttentionParameters, computeAttentionProbs, computeVxAttentionScore} from './attention';
import {createExpandProgramInfo} from './expand';
import {maybeTransposeToBNSHAndAddBias, validateInputs} from './multi-head-attentiion';
import {createTransposeProgramInfo, TransposeAttributes} from './transpose';

export const applyAttention =
    (context: ComputeContext, q: TensorView, k: TensorView, v: TensorView, _maskIndex: TensorView|undefined,
     _past: TensorView|undefined, _pastKey: TensorView|undefined, _pastValue: TensorView|undefined,
     relativePositionBias: TensorView|undefined, parameters: AttentionParameters) => {
      const probs = computeAttentionProbs(context, q, k, relativePositionBias, parameters, 1.0);

      computeVxAttentionScore(context, probs, v, parameters);
    };

export const parseGroupQueryAttentionAttributes = (attributes: AttentionAttrs): AttentionAttrs =>
    createAttributeWithCacheKey({...attributes});

const weightTransposeAttribute: TransposeAttributes = createAttributeWithCacheKey({perm: [0, 2, 1, 3]});

const maybeExpandAndTransposeToBNSH =
    (context: ComputeContext, batchSize: number, numHeads: number, sequenceLength: number, headSize: number,
     input: TensorView, nReps: number) => {
      let reshapedInput = input;
      if (input.dims.length === 3) {
        reshapedInput = input.reshape([batchSize, sequenceLength, numHeads, headSize]);
      }
      let expanedInput = context.compute(
          createExpandProgramInfo([reshapedInput], [batchSize, sequenceLength, numHeads, nReps, headSize]),
          {inputs: [reshapedInput], outputs: [-1]})[0];
      expanedInput = expanedInput.reshape([batchSize, sequenceLength, numHeads * nReps, headSize]);
      return context.compute(
          createTransposeProgramInfo(expanedInput, weightTransposeAttribute.perm),
          {inputs: [expanedInput], outputs: [-1]})[0];
    };

export const groupQueryAttention = (context: ComputeContext, attributes: AttentionAttrs): void => {
  const params = validateInputs(context.inputs, attributes);
  params.kvNumHeads = attributes.kvNumHeads;
  params.vHeadSize = Math.floor(params.vHiddenSize / params.kvNumHeads!!);

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

  if (kvBNSH) {
    return applyAttention(
        context, Q, context.inputs[1], context.inputs[2], context.inputs[4], undefined, undefined, undefined,
        context.inputs[5], params);
  }

  const nRep = Math.floor(attributes.numHeads / params.kvNumHeads!!);
  const K = maybeExpandAndTransposeToBNSH(
      context, params.batchSize, params.kvNumHeads!!, params.kvSequenceLength, params.vHeadSize, context.inputs[1],
      nRep);

  const V = maybeExpandAndTransposeToBNSH(
      context, params.batchSize, params.kvNumHeads!!, params.kvSequenceLength, params.vHeadSize, context.inputs[2],
      nRep);
  applyAttention(
      context,
      Q,
      K,
      V,
      context.inputs[4],
      undefined,
      context.inputs[6],
      context.inputs[7],
      context.inputs[5],
      params,
  );
};
