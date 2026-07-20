# gocv CoreX migration session notes

- **Source:** hybridgroup/gocv `release` @ `274ac8a2d6fc025a9dd566ae10b7a54644582c40` (v0.43.0)
- **Goal:** Make gocv `cuda/` package build/test against Iluvatar CoreX (ivcore11), clang++ not nvcc.

## Survey

- gocv does **not** compile CUDA kernels itself; `cuda/` is CGO wrappers around OpenCV CUDA modules.
- Build surface: `Makefile` `build_cuda` / Dockerfiles `Dockerfile.opencv-gpu-cuda-*` (NVIDIA base + nvcc-oriented cmake).
- Verification: `go test ./cuda`, `go run -tags cuda ./cmd/cuda/main.go`.

## Work performed

1. Preflight: CoreX clang 22.1; `ixsmi` → no GPUs (sysfs shows iluvatar0/1 but `/dev/iluvatar*` missing; `mknod` denied).
2. Installed Go 1.25.5 under `.tools/go` (sandbox cannot write `/usr/local`).
3. Downloaded OpenCV/opencv_contrib 4.13.0; wrote `scripts/build_opencv_corex.sh` + `Dockerfile.opencv-gpu-corex` + `make build_cuda_corex`.
4. CMake WITH_CUDA against `/usr/local/corex`: CUDA **detected 10.2**; empty arch autodetection fixed with `CUDA_ARCH_BIN=7.5` + `--cuda-gpu-arch=ivcore11`.
5. NPP missing → local stub `.so` files; omit NPP-heavy contrib modules from default `BUILD_LIST` (keep `cudev` for GpuMat).
6. `-Xcompiler=-*` rejected → `CCC_OVERRIDE_OPTIONS` unwrap (iluvatar-cuda-base nvcc-flag-translation).
7. Patched cudev headers (`saturate_cast`, `shuffle`, `texture`) for `__ILUVATAR__`.
8. **Wall:** `gpu_mat.cu` reaches CoreX llc then fails on NV PTX (`%laneid`, `cvt.sat.*`, `ld.global.cg` / asm constraints) from CUDA/cudev include chain.
9. Standalone CoreX smoke: `clang++ -x ivcore ... -lcudart` **compiles**; runtime `cudaMalloc=100` (no device nodes).

## Outcome

- Port scaffolding + documented CoreX OpenCV build path committed.
- Full gocv CUDA package compile/test **blocked** until OpenCV CUDA .cu can be built on CoreX (PTX) and GPU nodes are present.
