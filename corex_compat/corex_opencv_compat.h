/* CoreX / ivcore11 shims for OpenCV CUDA (.cu) compilation. */
#pragma once

/* cuda_fp16.h on CoreX may not set __CUDA_FP16_HPP__ (half-raw redef case). */
#include <cuda_fp16.h>
#if defined(__CUDA_ARCH__) && !defined(__CUDA_FP16_HPP__)
#define __CUDA_FP16_HPP__
#endif
#include <cuda_fp16.hpp>

#ifndef __half2float
#define __half2float __nv_half2float
#endif
#ifndef __float2half_rn
#define __float2half_rn __nv_float2half_rn
#endif

/* Prefer explicit float shuffles to avoid ambiguous __shfl_up overloads. */
#if defined(__CUDACC__) || defined(__CUDA__)
#include <cuda_runtime.h>
#endif
