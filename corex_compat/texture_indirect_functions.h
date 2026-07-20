/*
 * Software texture implementation for Iluvatar Corex (no texture hardware).
 *
 * This header shadows /usr/local/corex/include/texture_indirect_functions.h
 * via the higher-priority -I. in the Makefile include path.
 *
 * Approach:
 *   - cudaMallocArray   → allocates flat device memory + metadata struct
 *   - cudaCreateTextureObject → packs {devPtr, pitch, w, h, modes} into a
 *     device-side __CorTexDesc, returns device pointer as the handle
 *   - tex2D<T>(handle, x, y) → casts handle back to __CorTexDesc*, performs
 *     software fetch (nearest-neighbor or bilinear)
 */
#pragma once

/* Corex SDK 缺少 __nv_tex_surf_handler，surface/texture 内部分发宏用空宏代替 */
#if defined(__ILUVATAR__)
#ifndef __nv_tex_surf_handler
#define __nv_tex_surf_handler(...) ((void)0)
#endif
#endif

#include <cstring>
#include <cstdint>
#include <cstdlib>

// Pull in the templated cudaMalloc<T>(T**, size_t) overload. The bridge calls
// cudaMalloc(&typedPtr, ...) below; in strict host C++ (g++) the C overload
// cudaMalloc(void**, size_t) (from cuda_runtime_api.h) rejects T** -> void**,
// so a host TU that only included cuda_runtime_api.h would fail to compile the
// bridge. cuda_runtime.h provides the templated overload and is guard-safe to
// re-include while it is itself being processed (CUDA device-mode path).
#include <cuda_runtime.h>

template <bool B, typename T = void> struct __cor_enable_if {};
template <typename T> struct __cor_enable_if<true, T> { typedef T type; };
template <typename T> struct __cor_is_scalar { static const bool value = false; };
template <> struct __cor_is_scalar<float>         { static const bool value = true; };
template <> struct __cor_is_scalar<unsigned char>  { static const bool value = true; };
template <> struct __cor_is_scalar<signed char>    { static const bool value = true; };
template <> struct __cor_is_scalar<unsigned short> { static const bool value = true; };
template <> struct __cor_is_scalar<short>          { static const bool value = true; };
template <> struct __cor_is_scalar<int>            { static const bool value = true; };
template <> struct __cor_is_scalar<unsigned int>   { static const bool value = true; };

/* Vector-type traits: component scalar type + count — used for per-channel bilinear */
template <typename T> struct __cor_vec_traits;
template <> struct __cor_vec_traits<float2>  { typedef float          elem; static const int n = 2; };
template <> struct __cor_vec_traits<float4>  { typedef float          elem; static const int n = 4; };
template <> struct __cor_vec_traits<uchar2>  { typedef unsigned char  elem; static const int n = 2; };
template <> struct __cor_vec_traits<uchar4>  { typedef unsigned char  elem; static const int n = 4; };
template <> struct __cor_vec_traits<char2>   { typedef signed char    elem; static const int n = 2; };
template <> struct __cor_vec_traits<char4>   { typedef signed char    elem; static const int n = 4; };
template <> struct __cor_vec_traits<ushort2> { typedef unsigned short elem; static const int n = 2; };
template <> struct __cor_vec_traits<ushort4> { typedef unsigned short elem; static const int n = 4; };
template <> struct __cor_vec_traits<short2>  { typedef short          elem; static const int n = 2; };
template <> struct __cor_vec_traits<short4>  { typedef short          elem; static const int n = 4; };
template <> struct __cor_vec_traits<int2>    { typedef int            elem; static const int n = 2; };
template <> struct __cor_vec_traits<int4>    { typedef int            elem; static const int n = 4; };
template <> struct __cor_vec_traits<uint2>   { typedef unsigned int   elem; static const int n = 2; };
template <> struct __cor_vec_traits<uint4>   { typedef unsigned int   elem; static const int n = 4; };

/* is floating-point element? (bilinear rounds integer elements, passes float through) */
template <typename E> struct __cor_is_fp { static const bool value = false; };
template <> struct __cor_is_fp<float>    { static const bool value = true; };

/* ═══════════════════════════════════════════════════════════════════════════
 *  Descriptor struct — lives in DEVICE memory, readable from kernels
 * ═══════════════════════════════════════════════════════════════════════════ */
struct __CorTexDesc {
    void     *devPtr;
    uint32_t  width;
    uint32_t  height;
    uint32_t  depth;
    uint32_t  pitch;          // bytes per row
    uint8_t   elemSize;       // bytes per element
    uint8_t   numChannels;
    uint8_t   filterMode;     // 0=Point, 1=Linear
    uint8_t   normalizedCoords;
    uint8_t   addressMode[3]; // 0=Wrap, 1=Clamp, 2=Mirror, 3=Border
    uint8_t   readMode;       // 0=ElementType, 1=NormalizedFloat
    uint8_t   isSigned;       // 1 = signed channel (int8/int16) → NormalizedFloat maps to [-1,1]
};

/* ═══════════════════════════════════════════════════════════════════════════
 *  Custom cudaArray — replaces the opaque Corex one
 * ═══════════════════════════════════════════════════════════════════════════ */
struct __CorArray {
    void     *devPtr;
    uint32_t  width;
    uint32_t  height;
    uint32_t  depth;
    uint8_t   elemSize;
    uint8_t   numChannels;
    uint8_t   isSigned;       // 1 = signed channel (int8/int16)
    uint32_t  flags;
};

/* ── Fix broken cudaCreateChannelDesc on Corex ───────────────────────────── */
#define __COR_MKCD(X,Y,Z,W,F) []()->cudaChannelFormatDesc{ cudaChannelFormatDesc d; d.x=X;d.y=Y;d.z=Z;d.w=W;d.f=F; return d; }()

static inline cudaChannelFormatDesc __cor_CCD(
    int x, int y, int z, int w, enum cudaChannelFormatKind f) {
    cudaChannelFormatDesc d;
    d.x = x; d.y = y; d.z = z; d.w = w; d.f = f;
    return d;
}

template <typename T> struct __cor_ChanDescHelper;
template <> struct __cor_ChanDescHelper<float>          { static cudaChannelFormatDesc get() { return __COR_MKCD(32,0,0,0,cudaChannelFormatKindFloat); } };
template <> struct __cor_ChanDescHelper<int>            { static cudaChannelFormatDesc get() { return __COR_MKCD(32,0,0,0,cudaChannelFormatKindSigned); } };
template <> struct __cor_ChanDescHelper<unsigned int>   { static cudaChannelFormatDesc get() { return __COR_MKCD(32,0,0,0,cudaChannelFormatKindUnsigned); } };
template <> struct __cor_ChanDescHelper<short>          { static cudaChannelFormatDesc get() { return __COR_MKCD(16,0,0,0,cudaChannelFormatKindSigned); } };
template <> struct __cor_ChanDescHelper<unsigned short> { static cudaChannelFormatDesc get() { return __COR_MKCD(16,0,0,0,cudaChannelFormatKindUnsigned); } };
template <> struct __cor_ChanDescHelper<char>           { static cudaChannelFormatDesc get() { return __COR_MKCD(8,0,0,0,cudaChannelFormatKindSigned); } };
template <> struct __cor_ChanDescHelper<unsigned char>  { static cudaChannelFormatDesc get() { return __COR_MKCD(8,0,0,0,cudaChannelFormatKindUnsigned); } };
template <> struct __cor_ChanDescHelper<signed char>    { static cudaChannelFormatDesc get() { return __COR_MKCD(8,0,0,0,cudaChannelFormatKindSigned); } };
template <> struct __cor_ChanDescHelper<short2>         { static cudaChannelFormatDesc get() { return __COR_MKCD(16,16,0,0,cudaChannelFormatKindSigned); } };
template <> struct __cor_ChanDescHelper<ushort2>        { static cudaChannelFormatDesc get() { return __COR_MKCD(16,16,0,0,cudaChannelFormatKindUnsigned); } };
template <> struct __cor_ChanDescHelper<short4>         { static cudaChannelFormatDesc get() { return __COR_MKCD(16,16,16,16,cudaChannelFormatKindSigned); } };
template <> struct __cor_ChanDescHelper<ushort4>        { static cudaChannelFormatDesc get() { return __COR_MKCD(16,16,16,16,cudaChannelFormatKindUnsigned); } };
template <> struct __cor_ChanDescHelper<char2>          { static cudaChannelFormatDesc get() { return __COR_MKCD(8,8,0,0,cudaChannelFormatKindSigned); } };
template <> struct __cor_ChanDescHelper<char4>          { static cudaChannelFormatDesc get() { return __COR_MKCD(8,8,8,8,cudaChannelFormatKindSigned); } };
template <> struct __cor_ChanDescHelper<float2>         { static cudaChannelFormatDesc get() { return __COR_MKCD(32,32,0,0,cudaChannelFormatKindFloat); } };
template <> struct __cor_ChanDescHelper<float4>         { static cudaChannelFormatDesc get() { return __COR_MKCD(32,32,32,32,cudaChannelFormatKindFloat); } };
template <> struct __cor_ChanDescHelper<uchar2>         { static cudaChannelFormatDesc get() { return __COR_MKCD(8,8,0,0,cudaChannelFormatKindUnsigned); } };
template <> struct __cor_ChanDescHelper<uchar4>         { static cudaChannelFormatDesc get() { return __COR_MKCD(8,8,8,8,cudaChannelFormatKindUnsigned); } };
template <> struct __cor_ChanDescHelper<int2>           { static cudaChannelFormatDesc get() { return __COR_MKCD(32,32,0,0,cudaChannelFormatKindSigned); } };
template <> struct __cor_ChanDescHelper<int4>           { static cudaChannelFormatDesc get() { return __COR_MKCD(32,32,32,32,cudaChannelFormatKindSigned); } };
template <> struct __cor_ChanDescHelper<uint2>          { static cudaChannelFormatDesc get() { return __COR_MKCD(32,32,0,0,cudaChannelFormatKindUnsigned); } };
template <> struct __cor_ChanDescHelper<uint4>          { static cudaChannelFormatDesc get() { return __COR_MKCD(32,32,32,32,cudaChannelFormatKindUnsigned); } };

template <typename T>
static inline cudaChannelFormatDesc __cor_CCD() {
    return __cor_ChanDescHelper<T>::get();
}
#define cudaCreateChannelDesc __cor_CCD

/* ── Helper: channel desc → element size ─────────────────────────────────── */
static inline int __cor_chanDescElemSize(const cudaChannelFormatDesc &d) {
    int bits = d.x + d.y + d.z + d.w;
    return bits / 8;
}

static inline int __cor_chanDescNumCh(const cudaChannelFormatDesc &d) {
    int n = 0;
    if (d.x) n++;
    if (d.y) n++;
    if (d.z) n++;
    if (d.w) n++;
    return n ? n : 1;
}

/* signed channel? (int8/int16 → NormalizedFloat maps to [-1,1]) */
static inline int __cor_chanDescIsSigned(const cudaChannelFormatDesc &d) {
    return d.f == cudaChannelFormatKindSigned ? 1 : 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  Host-side API overrides (inline, header-only)
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ── cudaMallocArray ─────────────────────────────────────────────────────── */
static inline cudaError_t __cor_MallocArray(
    cudaArray_t *array,
    const cudaChannelFormatDesc *desc,
    size_t width, size_t height = 0, unsigned int flags = 0)
{
    if (!array || !desc) return cudaErrorInvalidValue;
    if (height == 0) height = 1;

    __CorArray *ca = (__CorArray *)malloc(sizeof(__CorArray));
    ca->width  = (uint32_t)width;
    ca->height = (uint32_t)height;
    ca->depth  = 1;
    ca->elemSize = (uint8_t)__cor_chanDescElemSize(*desc);
    ca->numChannels = (uint8_t)__cor_chanDescNumCh(*desc);
    ca->isSigned = (uint8_t)__cor_chanDescIsSigned(*desc);
    ca->flags  = flags;

    size_t bytes = width * height * ca->elemSize;
    cudaError_t err = cudaMalloc(&ca->devPtr, bytes);
    if (err != cudaSuccess) { free(ca); return err; }

    *array = (cudaArray_t)ca;
    return cudaSuccess;
}
#define cudaMallocArray __cor_MallocArray

/* ── cudaFreeArray ───────────────────────────────────────────────────────── */
static inline cudaError_t __cor_FreeArray(cudaArray_t array) {
    if (!array) return cudaSuccess;
    __CorArray *ca = (__CorArray *)array;
    cudaFree(ca->devPtr);
    free(ca);
    return cudaSuccess;
}
#define cudaFreeArray __cor_FreeArray

/* ── cudaMemcpyToArray ───────────────────────────────────────────────────── */
static inline cudaError_t __cor_MemcpyToArray(
    cudaArray_t dst, size_t wOffset, size_t hOffset,
    const void *src, size_t count, cudaMemcpyKind kind)
{
    __CorArray *ca = (__CorArray *)dst;
    size_t offset = hOffset * ca->width * ca->elemSize + wOffset;
    return cudaMemcpy((char *)ca->devPtr + offset, src, count, kind);
}
#define cudaMemcpyToArray __cor_MemcpyToArray

/* ── cudaMemcpy2DToArray ─────────────────────────────────────────────────── */
static inline cudaError_t __cor_Memcpy2DToArray(
    cudaArray_t dst, size_t wOffset, size_t hOffset,
    const void *src, size_t spitch,
    size_t width, size_t height, cudaMemcpyKind kind)
{
    __CorArray *ca = (__CorArray *)dst;
    size_t dstPitch = ca->width * ca->elemSize;
    return cudaMemcpy2D(
        (char *)ca->devPtr + hOffset * dstPitch + wOffset,
        dstPitch, src, spitch, width, height, kind);
}
#define cudaMemcpy2DToArray __cor_Memcpy2DToArray

/* ── cudaMemcpyFromArray ─────────────────────────────────────────────────── */
static inline cudaError_t __cor_MemcpyFromArray(
    void *dst, cudaArray_const_t src,
    size_t wOffset, size_t hOffset,
    size_t count, cudaMemcpyKind kind)
{
    const __CorArray *ca = (const __CorArray *)src;
    size_t offset = hOffset * ca->width * ca->elemSize + wOffset;
    return cudaMemcpy(dst, (const char *)ca->devPtr + offset, count, kind);
}
#define cudaMemcpyFromArray __cor_MemcpyFromArray

/* ── cudaMemcpy(cudaArray_t) overloads ──────────────────────────────────── */
/* Intercept cudaMemcpy when dst/src is cudaArray_t (= cudaArray*).         */
/* Without these, Corex treats the __CorArray* host ptr as a GPU ptr        */
/* and corrupts runtime state. C++ overload resolution picks these          */
/* automatically; normal void* calls are unaffected.                        */
static inline cudaError_t cudaMemcpy(
    cudaArray_t dst, const void *src, size_t count, cudaMemcpyKind kind)
{
    return __cor_MemcpyToArray(dst, 0, 0, src, count, kind);
}
static inline cudaError_t cudaMemcpy(
    void *dst, cudaArray_t src, size_t count, cudaMemcpyKind kind)
{
    return __cor_MemcpyFromArray(dst, src, 0, 0, count, kind);
}

/* ── cudaCreateTextureObject ─────────────────────────────────────────────── */
static inline cudaError_t __cor_CreateTextureObject(
    cudaTextureObject_t *pTexObj,
    const cudaResourceDesc *rd,
    const cudaTextureDesc *td,
    const cudaResourceViewDesc * /*rv*/)
{
    if (!pTexObj || !rd) return cudaErrorInvalidValue;

    __CorTexDesc h;
    memset(&h, 0, sizeof(h));

    switch (rd->resType) {
    case cudaResourceTypeArray: {
        const __CorArray *ca = (const __CorArray *)rd->res.array.array;
        h.devPtr   = ca->devPtr;
        h.width    = ca->width;
        h.height   = ca->height;
        h.depth    = ca->depth;
        h.elemSize = ca->elemSize;
        h.numChannels = ca->numChannels;
        h.isSigned = ca->isSigned;
        h.pitch    = ca->width * ca->elemSize;
        break;
    }
    case cudaResourceTypePitch2D:
        h.devPtr   = rd->res.pitch2D.devPtr;
        h.width    = (uint32_t)rd->res.pitch2D.width;
        h.height   = (uint32_t)rd->res.pitch2D.height;
        h.elemSize = (uint8_t)__cor_chanDescElemSize(rd->res.pitch2D.desc);
        h.numChannels = (uint8_t)__cor_chanDescNumCh(rd->res.pitch2D.desc);
        h.isSigned = (uint8_t)__cor_chanDescIsSigned(rd->res.pitch2D.desc);
        h.pitch    = (uint32_t)rd->res.pitch2D.pitchInBytes;
        h.depth    = 1;
        break;
    case cudaResourceTypeLinear:
        h.devPtr   = rd->res.linear.devPtr;
        h.elemSize = (uint8_t)__cor_chanDescElemSize(rd->res.linear.desc);
        h.numChannels = (uint8_t)__cor_chanDescNumCh(rd->res.linear.desc);
        h.isSigned = (uint8_t)__cor_chanDescIsSigned(rd->res.linear.desc);
        h.width    = (uint32_t)(rd->res.linear.sizeInBytes / h.elemSize);
        h.height   = 1;
        h.pitch    = h.width * h.elemSize;
        h.depth    = 1;
        break;
    default:
        return cudaErrorInvalidValue;
    }

    if (td) {
        h.filterMode       = (uint8_t)td->filterMode;
        h.normalizedCoords = (uint8_t)td->normalizedCoords;
        h.readMode         = (uint8_t)td->readMode;
        for (int i = 0; i < 3; i++)
            h.addressMode[i] = (uint8_t)td->addressMode[i];
    }

    __CorTexDesc *d_desc;
    cudaError_t err = cudaMalloc(&d_desc, sizeof(__CorTexDesc));
    if (err != cudaSuccess) return err;
    err = cudaMemcpy(d_desc, &h, sizeof(__CorTexDesc), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cudaFree(d_desc); return err; }

    *pTexObj = (cudaTextureObject_t)(uintptr_t)d_desc;
    return cudaSuccess;
}
#define cudaCreateTextureObject __cor_CreateTextureObject

/* ── cudaDestroyTextureObject ────────────────────────────────────────────── */
static inline cudaError_t __cor_DestroyTextureObject(cudaTextureObject_t tex) {
    if (tex) cudaFree((void *)(uintptr_t)tex);
    return cudaSuccess;
}
#define cudaDestroyTextureObject __cor_DestroyTextureObject

/* ── cudaCreateSurfaceObject ─────────────────────────────────────────────── */
static inline cudaError_t __cor_CreateSurfaceObject(
    cudaSurfaceObject_t *pSurfObj,
    const cudaResourceDesc *rd)
{
    if (!pSurfObj || !rd) return cudaErrorInvalidValue;
    __CorTexDesc h;
    memset(&h, 0, sizeof(h));
    if (rd->resType == cudaResourceTypeArray) {
        const __CorArray *ca = (const __CorArray *)rd->res.array.array;
        h.devPtr   = ca->devPtr;
        h.width    = ca->width;
        h.height   = ca->height;
        h.elemSize = ca->elemSize;
        h.numChannels = ca->numChannels;
        h.pitch    = ca->width * ca->elemSize;
    }
    __CorTexDesc *d;
    cudaError_t err = cudaMalloc(&d, sizeof(__CorTexDesc));
    if (err != cudaSuccess) return err;
    cudaMemcpy(d, &h, sizeof(__CorTexDesc), cudaMemcpyHostToDevice);
    *pSurfObj = (cudaSurfaceObject_t)(uintptr_t)d;
    return cudaSuccess;
}
#define cudaCreateSurfaceObject __cor_CreateSurfaceObject

/* ── cudaDestroySurfaceObject ────────────────────────────────────────────── */
static inline cudaError_t __cor_DestroySurfaceObject(cudaSurfaceObject_t s) {
    if (s) cudaFree((void *)(uintptr_t)s);
    return cudaSuccess;
}
#define cudaDestroySurfaceObject __cor_DestroySurfaceObject

/* ═══════════════════════════════════════════════════════════════════════════
 *  Device-side software texture fetch
 * ═══════════════════════════════════════════════════════════════════════════ */
#ifdef __CUDA_ARCH__

/* ── Address modes ───────────────────────────────────────────────────────── */
static __device__ __forceinline__
int __cor_addr_clamp(int i, int n) { return max(0, min(i, n - 1)); }

static __device__ __forceinline__
int __cor_addr_wrap(int i, int n) { return ((i % n) + n) % n; }

static __device__ __forceinline__
int __cor_addr_mirror(int i, int n) {
    int p = ((i % (2 * n)) + 2 * n) % (2 * n);
    return p < n ? p : 2 * n - 1 - p;
}

static __device__ __forceinline__
int __cor_addr(int mode, int i, int n) {
    if (mode == 1) return __cor_addr_clamp(i, n);
    if (mode == 2) return __cor_addr_mirror(i, n);
    if (mode == 3) return i;  // Border: pass through, bounds checked in fetch
    return __cor_addr_wrap(i, n);  // default=Wrap
}

/* ── Raw element fetch (byte-addressed) ──────────────────────────────────── */
template <typename T>
static __device__ __forceinline__
T __cor_fetch(const __CorTexDesc *d, int ix, int iy) {
    ix = __cor_addr(d->addressMode[0], ix, d->width);
    iy = __cor_addr(d->addressMode[1], iy, d->height);
    if (ix < 0 || (unsigned)ix >= d->width ||
        iy < 0 || (unsigned)iy >= d->height)
        return T{};
    const char *row = (const char *)d->devPtr + (size_t)iy * d->pitch;
    return ((const T *)row)[ix];
}

/* float -> element conversion used by bilinear: round-to-nearest for integer
 * elements (≈ NV texture-unit fixed-point interpolation; plain truncation biases
 * results low), exact pass-through for float. */
template <typename E>
static __device__ __forceinline__ E __cor_round_cast(float x) {
    if (__cor_is_fp<E>::value) return (E)x;
    return (E)(x >= 0.0f ? x + 0.5f : x - 0.5f);
}

template <typename T>
static __device__ __forceinline__
typename __cor_enable_if<__cor_is_scalar<T>::value, T>::type
__cor_bilinear(const __CorTexDesc *d, int ix0, int iy0, float ax, float ay) {
    float v00 = (float)__cor_fetch<T>(d, ix0,   iy0);
    float v10 = (float)__cor_fetch<T>(d, ix0+1, iy0);
    float v01 = (float)__cor_fetch<T>(d, ix0,   iy0+1);
    float v11 = (float)__cor_fetch<T>(d, ix0+1, iy0+1);
    return __cor_round_cast<T>((1-ay)*((1-ax)*v00 + ax*v10) + ay*((1-ax)*v01 + ax*v11));
}

/* Vector types: true per-channel bilinear (interpolate each component as float). */
template <typename T>
static __device__ __forceinline__
typename __cor_enable_if<!__cor_is_scalar<T>::value, T>::type
__cor_bilinear(const __CorTexDesc *d, int ix0, int iy0, float ax, float ay) {
    typedef typename __cor_vec_traits<T>::elem E;
    const int N = __cor_vec_traits<T>::n;
    T t00 = __cor_fetch<T>(d, ix0,   iy0);
    T t10 = __cor_fetch<T>(d, ix0+1, iy0);
    T t01 = __cor_fetch<T>(d, ix0,   iy0+1);
    T t11 = __cor_fetch<T>(d, ix0+1, iy0+1);
    T out;
    const E *p00 = (const E*)&t00, *p10 = (const E*)&t10;
    const E *p01 = (const E*)&t01, *p11 = (const E*)&t11;
    E *po = (E*)&out;
    for (int c = 0; c < N; c++) {
        float v = (1-ay)*((1-ax)*(float)p00[c] + ax*(float)p10[c])
                + ay    *((1-ax)*(float)p01[c] + ax*(float)p11[c]);
        po[c] = __cor_round_cast<E>(v);
    }
    return out;
}

/* ── NormalizedFloat point-fetch helper (float return, width+sign aware) ── *
 * On CUDA, tex2D<float> with cudaReadModeNormalizedFloat reads the raw       *
 * integer element and divides by the type max:                              *
 *   unsigned uint8  -> v/255      uint16 -> v/65535      → [0,1]            *
 *   signed   int8   -> max(v/127,-1)  int16 -> max(v/32767,-1)  → [-1,1]   *
 * The generic __cor_fetch<T> reads sizeof(T) bytes; for T=float that is 4   *
 * bytes, which mismatches a uint8/uint16 texture.  This helper reads the    *
 * correct element width (d->elemSize) and signedness (d->isSigned).         */
static __device__ __forceinline__
float __cor_fetch_normalized(const __CorTexDesc *d, int ix, int iy) {
    if (d->elemSize == 1) {
        if (d->isSigned)
            return fmaxf((float)__cor_fetch<int8_t>(d, ix, iy) * (1.0f / 127.0f), -1.0f);
        return (float)__cor_fetch<uint8_t>(d, ix, iy) * (1.0f / 255.0f);
    }
    if (d->elemSize == 2) {
        if (d->isSigned)
            return fmaxf((float)__cor_fetch<int16_t>(d, ix, iy) * (1.0f / 32767.0f), -1.0f);
        return (float)__cor_fetch<uint16_t>(d, ix, iy) * (1.0f / 65535.0f);
    }
    return __cor_fetch<float>(d, ix, iy);
}

/* ── tex2D fetch/bilinear dispatcher ────────────────────────────────────── *
 * Using a helper struct instead of an explicit function template             *
 * specialisation avoids ODR issues when the header is included from         *
 * multiple translation units (explicit function specialisations can only be  *
 * static-linkage through primary-template inheritance, which clang-cuda     *
 * does not always honour when the primary carries __device__ qualifiers).   *
 * Member functions inside a class template definition are implicitly inline, *
 * so every TU gets its own copy - no ODR violation.                         */
template <typename T>
struct __cor_tex2d_dispatch {
    static __device__ __forceinline__
    T point(const __CorTexDesc *d, int ix, int iy) {
        return __cor_fetch<T>(d, ix, iy);
    }
    static __device__ __forceinline__
    T linear(const __CorTexDesc *d, int ix0, int iy0, float ax, float ay) {
        return __cor_bilinear<T>(d, ix0, iy0, ax, ay);
    }
};

/* float specialisation: applies NormalizedFloat normalisation when needed */
template <>
struct __cor_tex2d_dispatch<float> {
    static __device__ __forceinline__
    float point(const __CorTexDesc *d, int ix, int iy) {
        if (d->readMode == 1) // cudaReadModeNormalizedFloat
            return __cor_fetch_normalized(d, ix, iy);
        return __cor_fetch<float>(d, ix, iy);
    }
    static __device__ __forceinline__
    float linear(const __CorTexDesc *d, int ix0, int iy0, float ax, float ay) {
        if (d->readMode == 1) { // cudaReadModeNormalizedFloat
            float v00 = __cor_fetch_normalized(d, ix0,   iy0);
            float v10 = __cor_fetch_normalized(d, ix0+1, iy0);
            float v01 = __cor_fetch_normalized(d, ix0,   iy0+1);
            float v11 = __cor_fetch_normalized(d, ix0+1, iy0+1);
            return (1.0f-ay)*((1.0f-ax)*v00 + ax*v10) + ay*((1.0f-ax)*v01 + ax*v11);
        }
        return __cor_bilinear<float>(d, ix0, iy0, ax, ay);
    }
};

/* ── tex2D<T> ────────────────────────────────────────────────────────────── */
template <typename T>
static __device__ __forceinline__
T tex2D(cudaTextureObject_t obj, float x, float y) {
    const __CorTexDesc *d = (const __CorTexDesc *)(uintptr_t)obj;
    float fx = d->normalizedCoords ? x * d->width  : x;
    float fy = d->normalizedCoords ? y * d->height : y;

    if (d->filterMode == 0) { // Point
        int ix = (int)floorf(fx);
        int iy = (int)floorf(fy);
        return __cor_tex2d_dispatch<T>::point(d, ix, iy);
    }
    // Linear (bilinear)
    float bx = fx - 0.5f, by = fy - 0.5f;
    int ix0 = (int)floorf(bx), iy0 = (int)floorf(by);
    return __cor_tex2d_dispatch<T>::linear(d, ix0, iy0, bx - ix0, by - iy0);
}

/* ── tex1D<T> ────────────────────────────────────────────────────────────── */
template <typename T>
static __device__ __forceinline__
T tex1D(cudaTextureObject_t obj, float x) {
    return tex2D<T>(obj, x, 0.0f);
}

/* ── tex1Dfetch<T> (integer index, no filtering) ─────────────────────────── */
template <typename T>
static __device__ __forceinline__
T tex1Dfetch(cudaTextureObject_t obj, int x) {
    const __CorTexDesc *d = (const __CorTexDesc *)(uintptr_t)obj;
    return ((const T *)d->devPtr)[x];
}

/* ── tex3D<T> ────────────────────────────────────────────────────────────── */
template <typename T>
static __device__ __forceinline__
T tex3D(cudaTextureObject_t obj, float x, float y, float z) {
    const __CorTexDesc *d = (const __CorTexDesc *)(uintptr_t)obj;
    float fx = d->normalizedCoords ? x * d->width  : x;
    float fy = d->normalizedCoords ? y * d->height : y;
    float fz = d->normalizedCoords ? z * d->depth  : z;
    int ix = __cor_addr(d->addressMode[0], (int)floorf(fx), d->width);
    int iy = __cor_addr(d->addressMode[1], (int)floorf(fy), d->height);
    int iz = __cor_addr(d->addressMode[2], (int)floorf(fz), d->depth);
    size_t slicePitch = (size_t)d->pitch * d->height;
    const char *p = (const char *)d->devPtr + iz * slicePitch + iy * d->pitch;
    return ((const T *)p)[ix];
}

/* ── tex2DLayered<T> ─────────────────────────────────────────────────────── */
template <typename T>
static __device__ __forceinline__
T tex2DLayered(cudaTextureObject_t obj, float x, float y, int layer) {
    const __CorTexDesc *d = (const __CorTexDesc *)(uintptr_t)obj;
    float fx = d->normalizedCoords ? x * d->width  : x;
    float fy = d->normalizedCoords ? y * d->height : y;
    size_t layerPitch = (size_t)d->pitch * d->height;
    int ix = __cor_addr(d->addressMode[0], (int)floorf(fx), d->width);
    int iy = __cor_addr(d->addressMode[1], (int)floorf(fy), d->height);
    const char *p = (const char *)d->devPtr + layer * layerPitch + iy * d->pitch;
    return ((const T *)p)[ix];
}

/* ── texCubemap<T> — direction (x,y,z) → face + UV → layered fetch ────────── */
template <typename T>
static __device__ __forceinline__
T texCubemap(cudaTextureObject_t obj, float x, float y, float z) {
    float ax = fabsf(x), ay = fabsf(y), az = fabsf(z);
    int face; float sc, tc, ma;
    if (ax >= ay && ax >= az) {
        if (x > 0) { face = 0; sc = -z; tc = -y; ma = x; }
        else       { face = 1; sc =  z; tc = -y; ma = -x; }
    } else if (ay >= ax && ay >= az) {
        if (y > 0) { face = 2; sc =  x; tc =  z; ma = y; }
        else       { face = 3; sc =  x; tc = -z; ma = -y; }
    } else {
        if (z > 0) { face = 4; sc =  x; tc = -y; ma = z; }
        else       { face = 5; sc = -x; tc = -y; ma = -z; }
    }
    float u = 0.5f * (sc / ma + 1.0f);
    float v = 0.5f * (tc / ma + 1.0f);
    return tex2DLayered<T>(obj, u, v, face);
}

/* ── surf2Dwrite ─────────────────────────────────────────────────────────── */
template <typename T>
static __device__ __forceinline__
void surf2Dwrite(T val, cudaSurfaceObject_t obj, int x, int y,
                 cudaSurfaceBoundaryMode /*mode*/ = cudaBoundaryModeZero) {
    const __CorTexDesc *d = (const __CorTexDesc *)(uintptr_t)obj;
    if (x >= 0 && (unsigned)x < d->width * d->elemSize &&
        y >= 0 && (unsigned)y < d->height) {
        char *row = (char *)d->devPtr + y * d->pitch;
        *(T *)(row + x) = val;
    }
}

/* ── surf2Dread ──────────────────────────────────────────────────────────── */
template <typename T>
static __device__ __forceinline__
T surf2Dread(cudaSurfaceObject_t obj, int x, int y,
             cudaSurfaceBoundaryMode /*mode*/ = cudaBoundaryModeZero) {
    const __CorTexDesc *d = (const __CorTexDesc *)(uintptr_t)obj;
    if (x >= 0 && (unsigned)x < d->width * d->elemSize &&
        y >= 0 && (unsigned)y < d->height) {
        const char *row = (const char *)d->devPtr + y * d->pitch;
        return *(const T *)(row + x);
    }
    return T{};
}


/* ── surf2Dread: output-pointer overload (NV CUDA 风格) ──────────────────── */
template <typename T>
static __device__ __forceinline__
void surf2Dread(T *out, cudaSurfaceObject_t obj, int x, int y,
                cudaSurfaceBoundaryMode /*mode*/ = cudaBoundaryModeZero) {
    const __CorTexDesc *d = (const __CorTexDesc *)(uintptr_t)obj;
    if (x >= 0 && (unsigned)x < d->width * d->elemSize &&
        y >= 0 && (unsigned)y < d->height) {
        const char *row = (const char *)d->devPtr + y * d->pitch;
        *out = *(const T *)(row + x);
    } else {
        *out = T{};
    }
}

/* ── surf1Dwrite ─────────────────────────────────────────────────────────── */
template <typename T>
static __device__ __forceinline__
void surf1Dwrite(T val, cudaSurfaceObject_t obj, int x,
                 cudaSurfaceBoundaryMode /*mode*/ = cudaBoundaryModeZero) {
    const __CorTexDesc *d = (const __CorTexDesc *)(uintptr_t)obj;
    if (x >= 0 && (unsigned)x < d->width * d->elemSize) {
        *(T *)((char *)d->devPtr + x) = val;
    }
}

/* ── surf1Dread (return-by-value + output-pointer overloads) ─────────────── */
template <typename T>
static __device__ __forceinline__
T surf1Dread(cudaSurfaceObject_t obj, int x,
             cudaSurfaceBoundaryMode /*mode*/ = cudaBoundaryModeZero) {
    const __CorTexDesc *d = (const __CorTexDesc *)(uintptr_t)obj;
    if (x >= 0 && (unsigned)x < d->width * d->elemSize) {
        return *(const T *)((const char *)d->devPtr + x);
    }
    return T{};
}
template <typename T>
static __device__ __forceinline__
void surf1Dread(T *out, cudaSurfaceObject_t obj, int x,
                cudaSurfaceBoundaryMode mode = cudaBoundaryModeZero) {
    *out = surf1Dread<T>(obj, x, mode);
}

/* ── surf3Dwrite ─────────────────────────────────────────────────────────── *
 * 3D layout: dense row-major slabs. slicePitch = pitch * height (dense).
 * x is in BYTES (NV CUDA 约定), y/z in elements.
 */
template <typename T>
static __device__ __forceinline__
void surf3Dwrite(T val, cudaSurfaceObject_t obj, int x, int y, int z,
                 cudaSurfaceBoundaryMode /*mode*/ = cudaBoundaryModeZero) {
    const __CorTexDesc *d = (const __CorTexDesc *)(uintptr_t)obj;
    if (x >= 0 && (unsigned)x < d->width * d->elemSize &&
        y >= 0 && (unsigned)y < d->height &&
        z >= 0 && (unsigned)z < d->depth) {
        size_t slicePitch = (size_t)d->pitch * d->height;
        char *plane = (char *)d->devPtr + (size_t)z * slicePitch;
        char *row   = plane + (size_t)y * d->pitch;
        *(T *)(row + x) = val;
    }
}

/* ── surf3Dread (return-by-value + output-pointer overloads) ─────────────── */
template <typename T>
static __device__ __forceinline__
T surf3Dread(cudaSurfaceObject_t obj, int x, int y, int z,
             cudaSurfaceBoundaryMode /*mode*/ = cudaBoundaryModeZero) {
    const __CorTexDesc *d = (const __CorTexDesc *)(uintptr_t)obj;
    if (x >= 0 && (unsigned)x < d->width * d->elemSize &&
        y >= 0 && (unsigned)y < d->height &&
        z >= 0 && (unsigned)z < d->depth) {
        size_t slicePitch = (size_t)d->pitch * d->height;
        const char *plane = (const char *)d->devPtr + (size_t)z * slicePitch;
        const char *row   = plane + (size_t)y * d->pitch;
        return *(const T *)(row + x);
    }
    return T{};
}
template <typename T>
static __device__ __forceinline__
void surf3Dread(T *out, cudaSurfaceObject_t obj, int x, int y, int z,
                cudaSurfaceBoundaryMode mode = cudaBoundaryModeZero) {
    *out = surf3Dread<T>(obj, x, y, z, mode);
}


/* ── surf2DLayeredwrite / surf2DLayeredread ──────────────────────────────── *
 * Layered = 2D images stacked along layer dim. depth==numLayers; each layer
 * is a dense 2D plane with `pitch` row stride. slicePitch = pitch * height.
 */
template <typename T>
static __device__ __forceinline__
void surf2DLayeredwrite(T val, cudaSurfaceObject_t obj, int x, int y, int layer,
                        cudaSurfaceBoundaryMode /*mode*/ = cudaBoundaryModeZero) {
    const __CorTexDesc *d = (const __CorTexDesc *)(uintptr_t)obj;
    if (x >= 0 && (unsigned)x < d->width * d->elemSize &&
        y >= 0 && (unsigned)y < d->height &&
        layer >= 0 && (unsigned)layer < d->depth) {
        size_t slicePitch = (size_t)d->pitch * d->height;
        char *plane = (char *)d->devPtr + (size_t)layer * slicePitch;
        char *row   = plane + (size_t)y * d->pitch;
        *(T *)(row + x) = val;
    }
}
template <typename T>
static __device__ __forceinline__
T surf2DLayeredread(cudaSurfaceObject_t obj, int x, int y, int layer,
                    cudaSurfaceBoundaryMode /*mode*/ = cudaBoundaryModeZero) {
    const __CorTexDesc *d = (const __CorTexDesc *)(uintptr_t)obj;
    if (x >= 0 && (unsigned)x < d->width * d->elemSize &&
        y >= 0 && (unsigned)y < d->height &&
        layer >= 0 && (unsigned)layer < d->depth) {
        size_t slicePitch = (size_t)d->pitch * d->height;
        const char *plane = (const char *)d->devPtr + (size_t)layer * slicePitch;
        const char *row   = plane + (size_t)y * d->pitch;
        return *(const T *)(row + x);
    }
    return T{};
}
template <typename T>
static __device__ __forceinline__
void surf2DLayeredread(T *out, cudaSurfaceObject_t obj, int x, int y, int layer,
                       cudaSurfaceBoundaryMode mode = cudaBoundaryModeZero) {
    *out = surf2DLayeredread<T>(obj, x, y, layer, mode);
}

#endif /* __CUDA_ARCH__ */

/* ═══════════════════════════════════════════════════════════════════════════
 *  Host-side 3D array & memcpy (for layered / cubemap / 3D textures)
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ── cudaMalloc3DArray ───────────────────────────────────────────────────── */
static inline cudaError_t __cor_Malloc3DArray(
    cudaArray_t *array,
    const cudaChannelFormatDesc *desc,
    struct cudaExtent extent,
    unsigned int flags = 0)
{
    if (!array || !desc) return cudaErrorInvalidValue;
    size_t w = extent.width, h = extent.height ? extent.height : 1;
    size_t d = extent.depth ? extent.depth : 1;

    __CorArray *ca = (__CorArray *)malloc(sizeof(__CorArray));
    ca->width  = (uint32_t)w;
    ca->height = (uint32_t)h;
    ca->depth  = (uint32_t)d;
    ca->elemSize = (uint8_t)__cor_chanDescElemSize(*desc);
    ca->numChannels = (uint8_t)__cor_chanDescNumCh(*desc);
    ca->isSigned = (uint8_t)__cor_chanDescIsSigned(*desc);
    ca->flags  = flags;

    size_t bytes = w * h * d * ca->elemSize;
    cudaError_t err = cudaMalloc(&ca->devPtr, bytes);
    if (err != cudaSuccess) { free(ca); return err; }

    *array = (cudaArray_t)ca;
    return cudaSuccess;
}
#define cudaMalloc3DArray __cor_Malloc3DArray

/* ── cudaMemcpy3D ────────────────────────────────────────────────────────── */
static inline cudaError_t __cor_Memcpy3D(const struct cudaMemcpy3DParms *p) {
    if (!p) return cudaErrorInvalidValue;

    if (p->dstArray && p->srcPtr.ptr) {
        __CorArray *ca = (__CorArray *)p->dstArray;
        size_t rowBytes = p->extent.width * ca->elemSize;
        size_t dstPitch = (size_t)ca->width * ca->elemSize;
        size_t srcYSize = p->srcPtr.ysize ? p->srcPtr.ysize : p->extent.height;

        for (size_t z = 0; z < p->extent.depth; z++) {
            for (size_t y = 0; y < p->extent.height; y++) {
                const char *src = (const char *)p->srcPtr.ptr
                    + (z + p->srcPos.z) * p->srcPtr.pitch * srcYSize
                    + (y + p->srcPos.y) * p->srcPtr.pitch
                    + p->srcPos.x * ca->elemSize;
                size_t dstOff = (z + p->dstPos.z) * dstPitch * ca->height
                    + (y + p->dstPos.y) * dstPitch
                    + p->dstPos.x * ca->elemSize;
                cudaError_t err = cudaMemcpy(
                    (char *)ca->devPtr + dstOff, src, rowBytes, p->kind);
                if (err != cudaSuccess) return err;
            }
        }
        return cudaSuccess;
    }

    if (p->srcArray && p->dstPtr.ptr) {
        const __CorArray *ca = (const __CorArray *)p->srcArray;
        size_t rowBytes = p->extent.width * ca->elemSize;
        size_t srcPitch = (size_t)ca->width * ca->elemSize;
        size_t dstYSize = p->dstPtr.ysize ? p->dstPtr.ysize : p->extent.height;

        for (size_t z = 0; z < p->extent.depth; z++) {
            for (size_t y = 0; y < p->extent.height; y++) {
                const char *src = (const char *)ca->devPtr
                    + (z + p->srcPos.z) * srcPitch * ca->height
                    + (y + p->srcPos.y) * srcPitch
                    + p->srcPos.x * ca->elemSize;
                size_t dstOff = (z + p->dstPos.z) * p->dstPtr.pitch * dstYSize
                    + (y + p->dstPos.y) * p->dstPtr.pitch
                    + p->dstPos.x * ca->elemSize;
                cudaError_t err = cudaMemcpy(
                    (char *)p->dstPtr.ptr + dstOff, src, rowBytes, p->kind);
                if (err != cudaSuccess) return err;
            }
        }
        return cudaSuccess;
    }

    return cudaErrorInvalidValue;
}
#define cudaMemcpy3D __cor_Memcpy3D

/* ─────────────────────────────────────────────────────────────────────────────
 * Old-style texture<>/surface<> reference bind API. These reference types are
 * CUDA-compiler builtins declared only under __CUDACC__ (see cuda_texture_types.h:
 * `#if defined(__cplusplus) && defined(__CUDACC__)`); in plain host C++ (g++ /
 * clang without -x ivcore) `texture`/`surface` are not templates, so this whole
 * block fails to compile. Guard it with __CUDACC__ so a host-only TU can still
 * #include this header and use the modern texture-OBJECT path (the software
 * cudaCreateTextureObject / tex2D above), which does not depend on __CUDACC__.
 * ───────────────────────────────────────────────────────────────────────────── */
#if defined(__CUDACC__)
#include "texture_indirect_functions_old.h"

/* ═══════════════════════════════════════════════════════════════════════════
 *  Host-side overrides for old-style texture/surface bind API
 *  (included here so they're defined after __cor_CreateTextureObject etc.)
 * ═══════════════════════════════════════════════════════════════════════════ */

#ifdef __cplusplus

static inline void __cor_old_store_texobj(textureReference &tex,
                                          cudaTextureObject_t obj) {
    uintptr_t p = (uintptr_t)obj;
    tex.__cudaReserved[0] = (int)(uint32_t)(p & 0xFFFFFFFFu);
    tex.__cudaReserved[1] = (int)(uint32_t)((p >> 32) & 0xFFFFFFFFu);
}

static inline void __cor_old_store_surfdata(surfaceReference &surf,
                                            void *devPtr, uint32_t pitch) {
    uintptr_t p = (uintptr_t)devPtr;
    surf.channelDesc.x = (int)(uint32_t)(p & 0xFFFFFFFFu);
    surf.channelDesc.y = (int)(uint32_t)((p >> 32) & 0xFFFFFFFFu);
    surf.channelDesc.z = (int)pitch;
}

template<class T, int dim, enum cudaTextureReadMode readMode>
static inline cudaError_t
__cor_old_BindTextureToArray(const struct texture<T, dim, readMode> &tex,
                             cudaArray_const_t array,
                             const struct cudaChannelFormatDesc &/*desc*/)
{
    cudaResourceDesc rd; memset(&rd, 0, sizeof(rd));
    rd.resType = cudaResourceTypeArray;
    rd.res.array.array = const_cast<cudaArray_t>(array);
    cudaTextureDesc td; memset(&td, 0, sizeof(td));
    td.filterMode       = tex.filterMode;
    td.normalizedCoords = tex.normalized;
    td.readMode         = cudaReadModeElementType; // Corex HW ignores NormFloat; device handles manually
    for (int i = 0; i < 3; i++) td.addressMode[i] = tex.addressMode[i];
    cudaTextureObject_t obj = 0;
    cudaError_t err = __cor_CreateTextureObject(&obj, &rd, &td, nullptr);
    if (err == cudaSuccess)
        __cor_old_store_texobj(const_cast<texture<T,dim,readMode>&>(tex), obj);
    return err;
}

template<class T, int dim, enum cudaTextureReadMode readMode>
static inline cudaError_t
__cor_old_BindTextureToArray(const struct texture<T, dim, readMode> &tex,
                             cudaArray_const_t array)
{
    return __cor_old_BindTextureToArray(tex, array, tex.channelDesc);
}

template<class T, int dim, enum cudaTextureReadMode readMode>
static inline cudaError_t
__cor_old_BindTexture(size_t *offset,
                      const struct texture<T, dim, readMode> &tex,
                      const void *devPtr,
                      const struct cudaChannelFormatDesc &desc,
                      size_t size = UINT_MAX)
{
    if (offset) *offset = 0;
    cudaResourceDesc rd; memset(&rd, 0, sizeof(rd));
    rd.resType = cudaResourceTypeLinear;
    rd.res.linear.devPtr      = const_cast<void *>(devPtr);
    rd.res.linear.desc        = desc;
    rd.res.linear.sizeInBytes = size;
    cudaTextureDesc td; memset(&td, 0, sizeof(td));
    td.filterMode       = tex.filterMode;
    td.normalizedCoords = 0;
    td.readMode         = cudaReadModeElementType; // Corex HW ignores NormFloat; device handles manually
    for (int i = 0; i < 3; i++) td.addressMode[i] = tex.addressMode[i];
    cudaTextureObject_t obj = 0;
    cudaError_t err = __cor_CreateTextureObject(&obj, &rd, &td, nullptr);
    if (err == cudaSuccess)
        __cor_old_store_texobj(const_cast<texture<T,dim,readMode>&>(tex), obj);
    return err;
}

template<class T, int dim, enum cudaTextureReadMode readMode>
static inline cudaError_t
__cor_old_BindTexture(size_t *offset,
                      const struct texture<T, dim, readMode> &tex,
                      const void *devPtr,
                      size_t size = UINT_MAX)
{
    return __cor_old_BindTexture(offset, tex, devPtr, tex.channelDesc, size);
}

template<class T, int dim, enum cudaTextureReadMode readMode>
static inline cudaError_t
__cor_old_BindTexture2D(size_t *offset,
                        const struct texture<T, dim, readMode> &tex,
                        const void *devPtr,
                        const struct cudaChannelFormatDesc &desc,
                        size_t width, size_t height, size_t pitch)
{
    if (offset) *offset = 0;
    cudaResourceDesc rd; memset(&rd, 0, sizeof(rd));
    rd.resType = cudaResourceTypePitch2D;
    rd.res.pitch2D.devPtr       = const_cast<void *>(devPtr);
    rd.res.pitch2D.desc         = desc;
    rd.res.pitch2D.width        = width;
    rd.res.pitch2D.height       = height;
    rd.res.pitch2D.pitchInBytes = pitch;
    cudaTextureDesc td; memset(&td, 0, sizeof(td));
    td.filterMode       = tex.filterMode;
    td.normalizedCoords = tex.normalized;
    td.readMode         = cudaReadModeElementType; // Corex HW ignores NormFloat; device handles manually
    for (int i = 0; i < 3; i++) td.addressMode[i] = tex.addressMode[i];
    cudaTextureObject_t obj = 0;
    cudaError_t err = __cor_CreateTextureObject(&obj, &rd, &td, nullptr);
    if (err == cudaSuccess)
        __cor_old_store_texobj(const_cast<texture<T,dim,readMode>&>(tex), obj);
    return err;
}

template<class T, int dim, enum cudaTextureReadMode readMode>
static inline cudaError_t
__cor_old_BindTexture2D(size_t *offset,
                        const struct texture<T, dim, readMode> &tex,
                        const void *devPtr,
                        size_t width, size_t height, size_t pitch)
{
    return __cor_old_BindTexture2D(offset, tex, devPtr, tex.channelDesc,
                                   width, height, pitch);
}

template<class T, int dim>
static inline cudaError_t
__cor_old_BindSurfaceToArray(const struct surface<T, dim> &surf,
                             cudaArray_const_t array,
                             const struct cudaChannelFormatDesc & /*desc*/)
{
    const __CorArray *ca = (const __CorArray *)(const void *)array;
    uint32_t pitch = ca->width * ca->elemSize;
    __cor_old_store_surfdata(const_cast<surface<T,dim>&>(surf),
                             ca->devPtr, pitch);
    return cudaSuccess;
}

template<class T, int dim>
static inline cudaError_t
__cor_old_BindSurfaceToArray(const struct surface<T, dim> &surf,
                             cudaArray_const_t array)
{
    return __cor_old_BindSurfaceToArray(surf, array, surf.channelDesc);
}

template<class T, int dim, enum cudaTextureReadMode readMode>
static inline cudaError_t
__cor_old_UnbindTexture(const struct texture<T, dim, readMode> &tex) {
    uintptr_t p =
        (uintptr_t)(uint32_t)tex.__cudaReserved[0] |
        ((uintptr_t)(uint32_t)tex.__cudaReserved[1] << 32);
    if (p) cudaFree((void *)p);
    return cudaSuccess;
}


/* ─── Pointer overloads for C API callers (pass &tex instead of tex) ─────── */
template<class T, int dim, enum cudaTextureReadMode readMode>
static inline cudaError_t
__cor_old_BindTexture2D(size_t *offset,
                        const struct texture<T, dim, readMode> *tex,
                        const void *devPtr,
                        const struct cudaChannelFormatDesc *desc,
                        size_t width, size_t height, size_t pitch)
{
    return __cor_old_BindTexture2D(offset, *tex, devPtr, *desc, width, height, pitch);
}

template<class T, int dim, enum cudaTextureReadMode readMode>
static inline cudaError_t
__cor_old_BindTexture(size_t *offset,
                      const struct texture<T, dim, readMode> *tex,
                      const void *devPtr,
                      const struct cudaChannelFormatDesc *desc,
                      size_t size = UINT_MAX)
{
    return __cor_old_BindTexture(offset, *tex, devPtr, *desc, size);
}

template<class T, int dim, enum cudaTextureReadMode readMode>
static inline cudaError_t
__cor_old_BindTextureToArray(const struct texture<T, dim, readMode> *tex,
                             cudaArray_const_t array,
                             const struct cudaChannelFormatDesc *desc)
{
    return __cor_old_BindTextureToArray(*tex, array, *desc);
}

#define cudaBindTextureToArray __cor_old_BindTextureToArray
#define cudaBindTexture        __cor_old_BindTexture
#define cudaBindTexture2D      __cor_old_BindTexture2D
#define cudaBindSurfaceToArray __cor_old_BindSurfaceToArray
#define cudaUnbindTexture      __cor_old_UnbindTexture
#endif /* __CUDACC__ : old-style texture<>/surface<> bind API */

#endif /* __cplusplus */
