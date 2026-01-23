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

// ReSTIR-FG Final Gather Reservoir Resampling Pass
// Performs spatiotemporal resampling of final gather reservoirs

#include "rtx/pass/common_bindings.slangh"
#include "rtx/pass/restir_fg/restir_fg_binding_indices.h"
#include "rtx/pass/restir_fg/restir_fg_types.slangh"

#include "rtx/utility/common.slangh"
#include "rtx/utility/noise.slangh"
#include "rtx/utility/math.slangh"
#include "rtx/utility/packing.slangh"
#include "rtx/utility/color.slangh"

// Inputs
layout(rgba16f, binding = RESTIR_FG_RESAMPLE_BINDING_MVEC_INPUT)
Texture2D<float4> MotionVector;

layout(rgba32f, binding = RESTIR_FG_RESAMPLE_BINDING_WORLD_POSITION_INPUT)
Texture2D<float4> WorldPosition;

layout(r32ui, binding = RESTIR_FG_RESAMPLE_BINDING_WORLD_NORMAL_INPUT)
Texture2D<uint> WorldNormal;

layout(binding = RESTIR_FG_RESAMPLE_BINDING_RESERVOIR_PREV)
StructuredBuffer<PackedReSTIRFG_Reservoir> FGReservoirPrev;

layout(binding = RESTIR_FG_RESAMPLE_BINDING_SAMPLE_PREV)
StructuredBuffer<PackedReSTIRFG_Sample> FGSamplePrev;

layout(binding = RESTIR_FG_RESAMPLE_BINDING_SURFACE_PREV)
StructuredBuffer<PackedReSTIRFG_SurfaceData> SurfaceDataPrev;

layout(binding = RESTIR_FG_RESAMPLE_BINDING_SURFACE_CURR)
StructuredBuffer<PackedReSTIRFG_SurfaceData> SurfaceDataCurr;

// Inputs/Outputs
layout(binding = RESTIR_FG_RESAMPLE_BINDING_RESERVOIR_CURR)
RWStructuredBuffer<PackedReSTIRFG_Reservoir> FGReservoirCurr;

layout(binding = RESTIR_FG_RESAMPLE_BINDING_SAMPLE_CURR)
RWStructuredBuffer<PackedReSTIRFG_Sample> FGSampleCurr;

// Decode shading normal from packed format
float3 decodeNormal(uint packed)
{
  return signedOctahedralToSphereDirection(snorm2x16ToFloat2x32(packed));
}

// Check if two surfaces are similar enough for resampling
bool areSurfacesSimilar(ReSTIRFG_SurfaceData curr, ReSTIRFG_SurfaceData prev)
{
  // Normal similarity check
  float normalSimilarity = dot(curr.normal, prev.normal);
  if (normalSimilarity < cb.restirFGNormalThreshold) {
    return false;
  }
  
  // Depth/position similarity check
  float distSq = dot(curr.position - prev.position, curr.position - prev.position);
  float avgDist = length(curr.position) + length(prev.position);
  float relDist = sqrt(distSq) / max(avgDist * 0.5f, 0.001f);
  if (relDist > cb.restirFGDepthThreshold) {
    return false;
  }
  
  return true;
}

// Combine two reservoirs using RIS
bool combineReservoirs(
  inout ReSTIRFG_Reservoir combined,
  inout ReSTIRFG_Sample combinedSample,
  ReSTIRFG_Reservoir candidate,
  ReSTIRFG_Sample candidateSample,
  float targetAtCombined,
  inout RNG rng)
{
  if (candidate.M == 0 || candidate.targetFunc <= 0.0f) {
    return false;
  }
  
  // Compute weight for this candidate
  float candidateWeight = targetAtCombined * candidate.wSum / max(candidate.targetFunc, 1e-6f);
  
  combined.wSum += candidateWeight;
  combined.M += candidate.M;
  
  // Reservoir selection
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
  
  // Load current reservoir and surface
  ReSTIRFG_Reservoir currentReservoir = FGReservoirCurr[linearIdx].unpack();
  ReSTIRFG_Sample currentSample = FGSampleCurr[linearIdx].unpack();
  ReSTIRFG_SurfaceData currentSurface = SurfaceDataCurr[linearIdx].unpack();
  
  // Skip invalid surfaces
  if (currentSurface.hitInfo == 0) {
    return;
  }
  
  // Initialize combined reservoir with current
  ReSTIRFG_Reservoir combinedReservoir = currentReservoir;
  ReSTIRFG_Sample combinedSample = currentSample;
  
  // Temporal resampling - use motion vectors to find previous pixel
  float4 mvec = MotionVector[pixel];
  float2 prevPixelF = float2(pixel) + mvec.xy * float2(cb.camera.resolution);
  int2 prevPixel = int2(round(prevPixelF));
  
  if (all(prevPixel >= int2(0)) && all(prevPixel < int2(cb.camera.resolution))) {
    uint prevLinearIdx = prevPixel.y * cb.camera.resolution.x + prevPixel.x;
    
    ReSTIRFG_Reservoir prevReservoir = FGReservoirPrev[prevLinearIdx].unpack();
    ReSTIRFG_Sample prevSample = FGSamplePrev[prevLinearIdx].unpack();
    ReSTIRFG_SurfaceData prevSurface = SurfaceDataPrev[prevLinearIdx].unpack();
    
    // Check surface similarity
    if (areSurfacesSimilar(currentSurface, prevSurface)) {
      // Cap temporal history
      uint maxHistory = cb.restirFGTemporalHistoryLength;
      if (prevReservoir.M > maxHistory) {
        float scale = float(maxHistory) / float(prevReservoir.M);
        prevReservoir.wSum *= scale;
        prevReservoir.M = maxHistory;
      }
      
      // Re-evaluate target function at current surface
      float targetAtCurrent = calcBt709Luminance(prevSample.radiance);
      
      // Combine with temporal neighbor
      combineReservoirs(combinedReservoir, combinedSample, prevReservoir, prevSample, targetAtCurrent, rng);
    }
  }
  
  // Spatial resampling
  uint spatialSamples = cb.restirFGSpatialSamples;
  float spatialRadius = cb.restirFGSpatialRadius;
  
  for (uint i = 0; i < spatialSamples; i++) {
    // Sample random neighbor in disk
    float2 offset;
    offset.x = (getNextSampleBlueNoise(rng) * 2.0f - 1.0f) * spatialRadius;
    offset.y = (getNextSampleBlueNoise(rng) * 2.0f - 1.0f) * spatialRadius;
    
    int2 neighborPixel = int2(pixel) + int2(round(offset));
    
    if (all(neighborPixel >= int2(0)) && all(neighborPixel < int2(cb.camera.resolution))) {
      uint neighborLinearIdx = neighborPixel.y * cb.camera.resolution.x + neighborPixel.x;
      
      ReSTIRFG_Reservoir neighborReservoir = FGReservoirCurr[neighborLinearIdx].unpack();
      ReSTIRFG_Sample neighborSample = FGSampleCurr[neighborLinearIdx].unpack();
      ReSTIRFG_SurfaceData neighborSurface = SurfaceDataCurr[neighborLinearIdx].unpack();
      
      // Check surface similarity
      if (areSurfacesSimilar(currentSurface, neighborSurface)) {
        // Re-evaluate target function at current surface
        float targetAtCurrent = calcBt709Luminance(neighborSample.radiance);
        
        // Combine with spatial neighbor
        combineReservoirs(combinedReservoir, combinedSample, neighborReservoir, neighborSample, targetAtCurrent, rng);
      }
    }
  }
  
  // Normalize final weight
  if (combinedReservoir.M > 0 && combinedReservoir.targetFunc > 0) {
    // This is the unbiased weight for the selected sample
    combinedReservoir.wSum = combinedReservoir.wSum / (combinedReservoir.targetFunc * float(combinedReservoir.M));
  }
  
  // Increment age
  combinedReservoir.age = currentReservoir.age + 1;
  
  // Write back results
  FGReservoirCurr[linearIdx] = PackedReSTIRFG_Reservoir::pack(combinedReservoir);
  FGSampleCurr[linearIdx] = PackedReSTIRFG_Sample::pack(combinedSample);
}
