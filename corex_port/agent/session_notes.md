# gocv CoreX 迁移记录

- **来源：** hybridgroup/gocv `release` @ `c7a0736`（v0.31.0，对应 OpenCV 4.6.0）
- **目标：** 让 gocv `cuda/` 包在 Iluvatar CoreX（ivcore11）上用 CoreX 工具链构建、并在真实 GPU 上跑测试。

## CUDA 使用性质

- gocv **本身不编译任何 CUDA kernel**；`cuda/` 只是对 OpenCV CUDA 模块的 CGO（C++）封装。
- 因此本次迁移的本质不是 `nvcc→clang` kernel 移植，而是**让 Go+CGO 正确链接到已经用 CoreX 构建好的 OpenCV**。

## 环境（本次关键变化）

- 环境新装了 **CoreX 构建版 OpenCV 4.6.0**，位于 `/usr/local/corex`（头文件 `include/opencv4`，含 `libopencv_cuda*.so`）。
- GPU 正常：`ixsmi` 识别 2× Iluvatar BI-V150；`CUDA_VISIBLE_DEVICES=0,1`。
- 工具链：CoreX clang++ 22.1.0git，Go 1.25.5（用户目录 `.tools/go`）。

## 适配内容

1. **版本对齐**：gocv HEAD（v0.43.0）要求 OpenCV 4.12，与已装的 4.6.0 API 不符（aruco/dnn 报错）。改用匹配 4.6.0 的 gocv **v0.31.0**，以 git worktree 放在 `.tools/gocv-0.31`。
2. **pkg-config 路径**：`opencv4.pc` 把 `prefix` 写死为 `/usr/local`，实际在 `/usr/local/corex`。用 `pkg-config --define-prefix` + `CGO_LDFLAGS=-L/usr/local/corex/lib64` 修正。
3. **libstdc++ ABI**：CoreX OpenCV 库导出的是 **pre-C++11 std::string ABI**（`nm -D` 全是 `RKSs`、无 `__cxx11`）。给 CGO 加 `-D_GLIBCXX_USE_CXX11_ABI=0`，否则所有 `cv::` 字符串符号链接期未定义。

## 结果

- **编译/链接：成功**；`cuda.test` 在 CoreX OpenCV 上构建通过。
- **测试：45 例中 38 例通过**（在真实 BI-V150 上运行）。核心 GpuMat / arithm / 滤波 / resize / 金字塔 / remap / MOG 背景建模 / Canny 全部通过。
- **7 例失败，全部源于 CoreX OpenCV 库本身，与 gocv 绑定无关：**
  - `TestFlip` / `TestFlipWithStream`：CoreX `cv::cuda::flip` 是抛异常的占位实现（`need to support!!!`）。
  - `TestSparsePyrLKOpticalFlow_Calc`：CoreX cudaoptflow 的 PyrLK 未启用 CUDA（`throw_no_cuda: No CUDA support`）。
  - `TestHoughSegment_Calc` / `WithStream`：CoreX cudaimgproc 的 HoughSegment kernel 触发 `SIGILL`。
  - `TestHoughLines_Calc` / `WithStream`：能跑通，但结果列数 1598 vs 上游硬编码期望 1588（数值分布略有差异，断言是按 NVIDIA 结果写死的精确匹配）。

## Failure Gate

- 所有失败均已**实跑复现**并归类：`flip` / PyrLK / HoughSegment 为 **terminal**（属 CoreX OpenCV 供应商的库缺陷，受"零改动 SDK"约束不能本地绕过）；HoughLines 数值差异为 **workaround-able**（换成带容差的断言即可通过，功能本身可用）。
- 已验证 GPU 可用、可执行，非"推断失败"。
