/*
 * texture_indirect_functions_old.h
 * Device-side emulation of old-style (reference-based) texture/surface API.
 *
 * Included at the end of texture_indirect_functions.h (bridge).
 *
 * How it works:
 *   The host-side bind (in texture_indirect_functions.h) always creates the
 *   texture object with cudaReadModeElementType; Corex hardware ignores
 *   cudaReadModeNormalizedFloat.  Manual normalization is performed here.
 *
 *   The handle is stored as two int32 halves in
 *   textureReference.__cudaReserved[0] (lo) and [1] (hi).
 *
 *   Surface device-ptr + pitch are stored in
 *   surfaceReference.channelDesc.{x,y,z}.
 *
 * NOTE: texture<T> globals are __device_builtin__ on Corex.
 *   After calling cudaBindTexture[ToArray] use a per-texture __global__
 *   setup kernel to propagate the handle into the device copy.
 *   See COR_PROPAGATE() in texture_bridge_test_old.cu.
 */

#pragma once

#ifdef __CUDA_ARCH__

/* ── Return-type trait for cudaReadModeNormalizedFloat ───────────────────── */
template<typename T, enum cudaTextureReadMode mode>
struct __cor_old_ret { typedef T type; };

template<> struct __cor_old_ret<unsigned char,  cudaReadModeNormalizedFloat> { typedef float  type; };
template<> struct __cor_old_ret<signed char,    cudaReadModeNormalizedFloat> { typedef float  type; };
template<> struct __cor_old_ret<unsigned short, cudaReadModeNormalizedFloat> { typedef float  type; };
template<> struct __cor_old_ret<short,          cudaReadModeNormalizedFloat> { typedef float  type; };
template<> struct __cor_old_ret<uchar2,         cudaReadModeNormalizedFloat> { typedef float2 type; };
template<> struct __cor_old_ret<char2,          cudaReadModeNormalizedFloat> { typedef float2 type; };
template<> struct __cor_old_ret<ushort2,        cudaReadModeNormalizedFloat> { typedef float2 type; };
template<> struct __cor_old_ret<short2,         cudaReadModeNormalizedFloat> { typedef float2 type; };
template<> struct __cor_old_ret<uchar4,         cudaReadModeNormalizedFloat> { typedef float4 type; };
template<> struct __cor_old_ret<char4,          cudaReadModeNormalizedFloat> { typedef float4 type; };
template<> struct __cor_old_ret<ushort4,        cudaReadModeNormalizedFloat> { typedef float4 type; };
template<> struct __cor_old_ret<short4,         cudaReadModeNormalizedFloat> { typedef float4 type; };

/* ── Manual normalization (Corex HW ignores cudaReadModeNormalizedFloat) ─── */
/* Plain overloads — no return-type varying template needed                   */
static __device__ __forceinline__ float  __cor_normalize(unsigned char  v) { return (float)v / 255.f; }
static __device__ __forceinline__ float  __cor_normalize(signed char    v) { return fmaxf((float)v / 127.f, -1.f); }
static __device__ __forceinline__ float  __cor_normalize(unsigned short v) { return (float)v / 65535.f; }
static __device__ __forceinline__ float  __cor_normalize(short          v) { return fmaxf((float)v / 32767.f, -1.f); }
static __device__ __forceinline__ float2 __cor_normalize(uchar2  v) {
    return make_float2(v.x / 255.f, v.y / 255.f); }
static __device__ __forceinline__ float2 __cor_normalize(char2   v) {
    return make_float2(fmaxf(v.x/127.f,-1.f), fmaxf(v.y/127.f,-1.f)); }
static __device__ __forceinline__ float2 __cor_normalize(ushort2 v) {
    return make_float2(v.x / 65535.f, v.y / 65535.f); }
static __device__ __forceinline__ float2 __cor_normalize(short2  v) {
    return make_float2(fmaxf(v.x/32767.f,-1.f), fmaxf(v.y/32767.f,-1.f)); }
static __device__ __forceinline__ float4 __cor_normalize(uchar4  v) {
    return make_float4(v.x/255.f, v.y/255.f, v.z/255.f, v.w/255.f); }
static __device__ __forceinline__ float4 __cor_normalize(char4   v) {
    return make_float4(fmaxf(v.x/127.f,-1.f), fmaxf(v.y/127.f,-1.f),
                       fmaxf(v.z/127.f,-1.f), fmaxf(v.w/127.f,-1.f)); }
static __device__ __forceinline__ float4 __cor_normalize(ushort4 v) {
    return make_float4(v.x/65535.f, v.y/65535.f, v.z/65535.f, v.w/65535.f); }
static __device__ __forceinline__ float4 __cor_normalize(short4  v) {
    return make_float4(fmaxf(v.x/32767.f,-1.f), fmaxf(v.y/32767.f,-1.f),
                       fmaxf(v.z/32767.f,-1.f), fmaxf(v.w/32767.f,-1.f)); }

/* ── Load cudaTextureObject_t from textureReference.__cudaReserved[0:1] ──── */
static __device__ __forceinline__
cudaTextureObject_t __cor_old_load_texobj(const textureReference &ref)
{
    uint64_t lo = (uint32_t)ref.__cudaReserved[0];
    uint64_t hi = (uint32_t)ref.__cudaReserved[1];
    return (cudaTextureObject_t)(lo | (hi << 32));
}

/* ── Load surface device pointer from surfaceReference.channelDesc.{x,y} ─── */
static __device__ __forceinline__
void *__cor_old_surf_ptr(const surfaceReference &ref)
{
    uint64_t lo = (uint32_t)ref.channelDesc.x;
    uint64_t hi = (uint32_t)ref.channelDesc.y;
    return (void *)(lo | (hi << 32));
}

static __device__ __forceinline__
uint32_t __cor_old_surf_pitch(const surfaceReference &ref) {
    return (uint32_t)ref.channelDesc.z;
}

/* ─────────────────────────── tex1Dfetch ────────────────────────────────── */
template<class T, enum cudaTextureReadMode readMode>
static __device__ __forceinline__
typename __cor_old_ret<T, readMode>::type
tex1Dfetch(const texture<T, 1, readMode> &ref, int i)
{
    cudaTextureObject_t obj = __cor_old_load_texobj(ref);
    if constexpr (readMode == cudaReadModeNormalizedFloat) {
        return __cor_normalize(tex1Dfetch<T>(obj, i));
    } else {
        return tex1Dfetch<T>(obj, i);
    }
}

/* ─────────────────────────── tex1D ─────────────────────────────────────── */
template<class T, enum cudaTextureReadMode readMode>
static __device__ __forceinline__
typename __cor_old_ret<T, readMode>::type
tex1D(const texture<T, 1, readMode> &ref, float x)
{
    cudaTextureObject_t obj = __cor_old_load_texobj(ref);
    if constexpr (readMode == cudaReadModeNormalizedFloat) {
        return __cor_normalize(tex1D<T>(obj, x));
    } else {
        return tex1D<T>(obj, x);
    }
}

/* ─────────────────────────── tex2D ─────────────────────────────────────── */
template<class T, enum cudaTextureReadMode readMode>
static __device__ __forceinline__
typename __cor_old_ret<T, readMode>::type
tex2D(const texture<T, 2, readMode> &ref, float x, float y)
{
    cudaTextureObject_t obj = __cor_old_load_texobj(ref);
    if constexpr (readMode == cudaReadModeNormalizedFloat) {
        return __cor_normalize(tex2D<T>(obj, x, y));
    } else {
        return tex2D<T>(obj, x, y);
    }
}

/* ─────────────────────────── surf1Dwrite ───────────────────────────────── */
template<class T>
static __device__ __forceinline__ void
surf1Dwrite(T val, const surface<void, 1> &ref, int byteOff,
            cudaSurfaceBoundaryMode /*mode*/ = cudaBoundaryModeTrap)
{
    char *p = (char *)__cor_old_surf_ptr(ref);
    *(T *)(p + byteOff) = val;
}

template<class T>
static __device__ __forceinline__ void
surf1Dwrite(T val, const surface<T, 1> &ref, int byteOff,
            cudaSurfaceBoundaryMode /*mode*/ = cudaBoundaryModeTrap)
{
    char *p = (char *)__cor_old_surf_ptr(ref);
    *(T *)(p + byteOff) = val;
}

/* ─────────────────────────── surf2Dwrite ───────────────────────────────── */
template<class T>
static __device__ __forceinline__ void
surf2Dwrite(T val, const surface<void, 2> &ref, int xBytes, int y,
            cudaSurfaceBoundaryMode /*mode*/ = cudaBoundaryModeTrap)
{
    char *p = (char *)__cor_old_surf_ptr(ref);
    *(T *)(p + (size_t)y * __cor_old_surf_pitch(ref) + xBytes) = val;
}

template<class T>
static __device__ __forceinline__ void
surf2Dwrite(T val, const surface<T, 2> &ref, int xBytes, int y,
            cudaSurfaceBoundaryMode /*mode*/ = cudaBoundaryModeTrap)
{
    char *p = (char *)__cor_old_surf_ptr(ref);
    *(T *)(p + (size_t)y * __cor_old_surf_pitch(ref) + xBytes) = val;
}

#endif /* __CUDA_ARCH__ */
