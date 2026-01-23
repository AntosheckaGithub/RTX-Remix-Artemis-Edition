/*
* Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

// ReSTIR-FG Caustic Reservoir Resampling Pass
// Performs spatiotemporal resampling of caustic photon reservoirs

#include "rtx/pass/common_bindings.slangh"
#include "rtx/pass/restir_fg/restir_fg_binding_indices.h"
#include "rtx/pass/restir_fg/restir_fg_types.slangh"

#include "rtx/utility/common.slangh"
#include "rtx/utility/noise.slangh"
#include "rtx/utility/math.slangh"
#include "rtx/utility/packing.slangh"
#include "rtx/utility/color.slangh"

// Inputs
layout(rgba16f, binding = RESTIR_FG_CAUSTIC_RESAMPLE_BINDING_MVEC_INPUT)
Texture2D<float4> MotionVector;

layout(rgba32f, binding = RESTIR_FG_CAUSTIC_RESAMPLE_BINDING_WORLD_POSITION_INPUT)
Texture2D<float4> WorldPosition;

layout(r32ui, binding = RESTIR_FG_CAUSTIC_RESAMPLE_BINDING_WORLD_NORMAL_INPUT)
Texture2D<uint> WorldNormal;

layout(binding = RESTIR_FG_CAUSTIC_RESAMPLE_BINDING_RESERVOIR_PREV)
StructuredBuffer<PackedReSTIRFG_CausticReservoir> CausticReservoirPrev;

layout(binding = RESTIR_FG_CAUSTIC_RESAMPLE_BINDING_SAMPLE_PREV)
StructuredBuffer<PackedReSTIRFG_CausticSample> CausticSamplePrev;

// Inputs/Outputs
layout(binding = RESTIR_FG_CAUSTIC_RESAMPLE_BINDING_RESERVOIR_CURR)
RWStructuredBuffer<PackedReSTIRFG_CausticReservoir> CausticReservoirCurr;

layout(binding = RESTIR_FG_CAUSTIC_RESAMPLE_BINDING_SAMPLE_CURR)
RWStructuredBuffer<PackedReSTIRFG_CausticSample> CausticSampleCurr;

// Decode shading normal from packed format
float3 decodeNormal(uint packed)
{
  return signedOctahedralToSphereDirection(snorm2x16ToFloat2x32(packed));
}

// Combine caustic reservoirs using RIS
bool combineCausticReservoirs(
  inout ReSTIRFG_CausticReservoir combined,
  inout ReSTIRFG_CausticSample combinedSample,
  ReSTIRFG_CausticReservoir candidate,
  ReSTIRFG_CausticSample candidateSample,
  float targetAtCombined,
  inout RNG rng)
{
  if (candidate.M == 0 || candidate.targetFunc <= 0.0f) {
    return false;
  }
  
  float candidateWeight = targetAtCombined * candidate.wSum / max(candidate.targetFunc, 1e-6f);
  
  combined.wSum += candidateWeight;
  combined.M += candidate.M;
  
  float rand = getNextSampleBlueNoise(rng);
  if (rand * combined.wSum < candidateWeight) {
    combined.targetFunc = targetAtCombined;
    combinedSample = candidateSample;
    return true;
  }
  
  return false;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
  uint2 pixel = dispatchThreadId.xy;
  
  if (any(pixel >= cb.camera.resolution)) {
    return;
  }
  
  uint linearIdx = pixel.y * cb.camera.resolution.x + pixel.x;
  
  // Initialize RNG
  RNG rng = createRNG(uvec2(pixel), cb.frameIdx);
  
  // Load current data
  ReSTIRFG_CausticReservoir currentReservoir = CausticReservoirCurr[linearIdx].unpack();
  ReSTIRFG_CausticSample currentSample = CausticSampleCurr[linearIdx].unpack();
  
  // Load surface data for similarity checks
  float4 worldPosData = WorldPosition[pixel];
  float3 surfacePos = worldPosData.xyz;
  uint packedNormal = WorldNormal[pixel];
  float3 surfaceNormal = decodeNormal(packedNormal);
  
  // Skip invalid surfaces
  if (all(surfacePos == float3(0))) {
    return;
  }
  
  // Initialize combined reservoir
  ReSTIRFG_CausticReservoir combinedReservoir = currentReservoir;
  ReSTIRFG_CausticSample combinedSample = currentSample;
  
  // Temporal resampling
  float4 mvec = MotionVector[pixel];
  float2 prevPixelF = float2(pixel) + mvec.xy * float2(cb.camera.resolution);
  int2 prevPixel = int2(round(prevPixelF));
  
  if (all(prevPixel >= int2(0)) && all(prevPixel < int2(cb.camera.resolution))) {
    uint prevLinearIdx = prevPixel.y * cb.camera.resolution.x + prevPixel.x;
    
    ReSTIRFG_CausticReservoir prevReservoir = CausticReservoirPrev[prevLinearIdx].unpack();
    ReSTIRFG_CausticSample prevSample = CausticSamplePrev[prevLinearIdx].unpack();
    
    // Cap temporal history
    uint maxHistory = cb.restirFGTemporalHistoryLength;
    if (prevReservoir.M > maxHistory) {
      float scale = float(maxHistory) / float(prevReservoir.M);
      prevReservoir.wSum *= scale;
      prevReservoir.M = maxHistory;
    }
    
    // Re-evaluate target function at current surface
    float targetAtCurrent = calcBt709Luminance(prevSample.photonFlux);
    
    // Combine with temporal neighbor
    combineCausticReservoirs(combinedReservoir, combinedSample, prevReservoir, prevSample, targetAtCurrent, rng);
  }
  
  // Spatial resampling for caustics
  uint causticSpatialSamples = cb.restirFGCausticSpatialSamples;
  float causticSpatialRadius = cb.restirFGCausticSpatialRadius;
  
  for (uint i = 0; i < causticSpatialSamples; i++) {
    float2 offset;
    offset.x = (getNextSampleBlueNoise(rng) * 2.0f - 1.0f) * causticSpatialRadius;
    offset.y = (getNextSampleBlueNoise(rng) * 2.0f - 1.0f) * causticSpatialRadius;
    
    int2 neighborPixel = int2(pixel) + int2(round(offset));
    
    if (all(neighborPixel >= int2(0)) && all(neighborPixel < int2(cb.camera.resolution))) {
      uint neighborLinearIdx = neighborPixel.y * cb.camera.resolution.x + neighborPixel.x;
      
      ReSTIRFG_CausticReservoir neighborReservoir = CausticReservoirCurr[neighborLinearIdx].unpack();
      ReSTIRFG_CausticSample neighborSample = CausticSampleCurr[neighborLinearIdx].unpack();
      
      // Re-evaluate target function
      float targetAtCurrent = calcBt709Luminance(neighborSample.photonFlux);
      
      combineCausticReservoirs(combinedReservoir, combinedSample, neighborReservoir, neighborSample, targetAtCurrent, rng);
    }
  }
  
  // Normalize final weight
  if (combinedReservoir.M > 0 && combinedReservoir.targetFunc > 0) {
    combinedReservoir.wSum = combinedReservoir.wSum / (combinedReservoir.targetFunc * float(combinedReservoir.M));
  }
  
  // Write back results
  CausticReservoirCurr[linearIdx] = PackedReSTIRFG_CausticReservoir::pack(combinedReservoir);
  CausticSampleCurr[linearIdx] = PackedReSTIRFG_CausticSample::pack(combinedSample);
}
