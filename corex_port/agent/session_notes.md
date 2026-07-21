# gocv CoreX 迁移记录

- **来源：** hybridgroup/gocv `release` @ `274ac8a2d6fc025a9dd566ae10b7a54644582c40`（v0.43.0）
- **目标：** 让 gocv `cuda/` 包能针对 Iluvatar CoreX（ivcore11）用 clang++（而非 nvcc）构建/测试。

## 调研

- gocv 本身**不编译** CUDA kernel；`cuda/` 是对 OpenCV CUDA 模块的 CGO 封装。
- 构建面：`Makefile` 的 `build_cuda` / Dockerfile `Dockerfile.opencv-gpu-cuda-*`（NVIDIA 基础镜像 + 面向 nvcc 的 cmake）。
- 验证方式：`go test ./cuda`、`go run -tags cuda ./cmd/cuda/main.go`。

## 已执行工作

1. 预检：CoreX clang 22.1；`ixsmi` → 无 GPU（sysfs 有 iluvatar0/1，但 `/dev/iluvatar*` 缺失，`mknod` 被拒）。
2. 在 `.tools/go` 下安装 Go 1.25.5（沙箱无法写 `/usr/local`）。
3. 下载 OpenCV/opencv_contrib 4.13.0；编写 `scripts/build_opencv_corex.sh` + `Dockerfile.opencv-gpu-corex` + `make build_cuda_corex`。
4. CMake WITH_CUDA 指向 `/usr/local/corex`：CUDA **识别为 10.2**；arch 自动探测为空，用 `CUDA_ARCH_BIN=7.5` + `--cuda-gpu-arch=ivcore11` 修复。
5. 缺 NPP → 本地 stub `.so`；默认 `BUILD_LIST` 中剔除重度依赖 NPP 的 contrib 模块（保留 `cudev` 供 GpuMat）。
6. `-Xcompiler=-*` 被拒 → 用 `CCC_OVERRIDE_OPTIONS` 拆解（iluvatar-cuda-base 的 nvcc-flag-translation 用例）。
7. 为 `__ILUVATAR__` 打了 cudev 头补丁（`saturate_cast`、`shuffle`、`texture`）。
8. **卡点：** `gpu_mat.cu` 进到 CoreX llc 后，因 CUDA/cudev 头链里的 NV PTX（`%laneid`、`cvt.sat.*`、`ld.global.cg` / asm 约束）编译失败。
9. 独立 CoreX 冒烟：`clang++ -x ivcore ... -lcudart` **可编译**；运行时 `cudaMalloc=100`（无设备节点）。

## 结论

- 已提交移植脚手架 + CoreX OpenCV 构建路径文档。
- gocv CUDA 包的完整编译/测试**被阻塞**，需先在 CoreX 上编出 OpenCV CUDA `.cu`（PTX 问题）且具备 GPU 设备节点。
