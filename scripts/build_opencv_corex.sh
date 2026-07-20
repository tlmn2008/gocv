#!/usr/bin/env bash
# Build OpenCV (+ cudev) against Iluvatar CoreX (clang++, ivcore11).
# Installs to a repo-local prefix (never touches /usr/local/corex).
#
# Notes:
# - CoreX has no NPP; we link against local stub .so files so CMake can
#   configure. Contrib modules that call nppi* (cudaarithm, etc.) are omitted
#   from the default BUILD_LIST — they need a real NPP or CV-CUDA rewrite.
# - OpenCV still wants numeric CUDA_ARCH_BIN; device code is forced to
#   ivcore11 via CMAKE_CUDA_FLAGS + CCC_OVERRIDE_OPTIONS.
set -euo pipefail

. /etc/profile.d/corex.sh

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OPENCV_VERSION="${OPENCV_VERSION:-4.13.0}"
SRC_ROOT="${SRC_ROOT:-$ROOT/.tools/opencv-src}"
BUILD_DIR="${BUILD_DIR:-$ROOT/.tools/opencv-build}"
PREFIX="${PREFIX:-$ROOT/.tools/opencv-install}"
STUB_DIR="${STUB_DIR:-$ROOT/.tools/npp-stubs}"
JOBS="${JOBS:-$(nproc --all --ignore 1 2>/dev/null || nproc)}"
BUILD_LIST="${BUILD_LIST:-core,imgproc,imgcodecs,highgui,videoio,video,calib3d,features2d,objdetect,dnn,photo,cudev}"

export PATH="${COREX_PATH}/bin:${PATH}"
export CC="${COREX_PATH}/bin/clang"
export CXX="${COREX_PATH}/bin/clang++"
export CUDACXX="${COREX_PATH}/bin/clang++"
export CUDA_HOME="${COREX_PATH}"
export CUDA_PATH="${COREX_PATH}"
export CUDA_TOOLKIT_ROOT_DIR="${COREX_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export LD_LIBRARY_PATH="${COREX_PATH}/lib64:${COREX_PATH}/lib:${STUB_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="${COREX_PATH}/lib64:${COREX_PATH}/lib:${STUB_DIR}${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CCC_OVERRIDE_OPTIONS="${CCC_OVERRIDE_OPTIONS:-x--use_fast_math x-lineinfo x--generate-line-info X--threads X-maxrregcount X-Xptxas}"
export CMAKE_TLS_VERIFY="${CMAKE_TLS_VERIFY:-0}"

mkdir -p "$STUB_DIR" "$BUILD_DIR" "$PREFIX"
# Minimal NPP stubs so OpenCV CMake's REQUIRED NPP vars are non-NOTFOUND.
for lib in nppc nppial nppicc nppicom nppidei nppif nppig nppim nppist nppisu nppitc npps; do
  if [[ ! -f "$STUB_DIR/lib${lib}.so" ]]; then
    echo "int ${lib}_corex_stub(void){return 0;}" | \
      "$CC" -shared -fPIC -o "$STUB_DIR/lib${lib}.so" -x c - -Wl,-soname,lib${lib}.so
  fi
done

NPP_ARGS=()
for lib in nppc nppial nppicc nppicom nppidei nppif nppig nppim nppist nppisu nppitc npps; do
  NPP_ARGS+=(-D "CUDA_${lib}_LIBRARY=$STUB_DIR/lib${lib}.so")
done

cd "$BUILD_DIR"
cmake "$SRC_ROOT/opencv-${OPENCV_VERSION}" \
  -D CMAKE_BUILD_TYPE=RELEASE \
  -D CMAKE_INSTALL_PREFIX="$PREFIX" \
  -D CMAKE_C_COMPILER="$CC" \
  -D CMAKE_CXX_COMPILER="$CXX" \
  -D CMAKE_CUDA_COMPILER="$CUDACXX" \
  -D CMAKE_CUDA_FLAGS="--cuda-gpu-arch=ivcore11 -std=c++14" \
  -D BUILD_SHARED_LIBS=ON \
  -D BUILD_LIST="$BUILD_LIST" \
  -D OPENCV_EXTRA_MODULES_PATH="$SRC_ROOT/opencv_contrib-${OPENCV_VERSION}/modules" \
  -D BUILD_DOCS=OFF \
  -D BUILD_EXAMPLES=OFF \
  -D BUILD_TESTS=OFF \
  -D BUILD_PERF_TESTS=OFF \
  -D BUILD_opencv_apps=OFF \
  -D BUILD_opencv_java=OFF \
  -D BUILD_opencv_python=OFF \
  -D BUILD_opencv_python2=OFF \
  -D BUILD_opencv_python3=OFF \
  -D WITH_IPP=OFF \
  -D WITH_OPENCL=OFF \
  -D WITH_QT=OFF \
  -D WITH_GTK=OFF \
  -D WITH_FFMPEG=OFF \
  -D WITH_GSTREAMER=OFF \
  -D WITH_TBB=OFF \
  -D WITH_JASPER=OFF \
  -D WITH_1394=OFF \
  -D WITH_V4L=OFF \
  -D BUILD_JPEG=ON \
  -D BUILD_PNG=ON \
  -D BUILD_TIFF=ON \
  -D BUILD_WEBP=ON \
  -D WITH_CUDA=ON \
  -D WITH_CUBLAS=ON \
  -D WITH_CUDNN=OFF \
  -D OPENCV_DNN_CUDA=OFF \
  -D ENABLE_FAST_MATH=OFF \
  -D CUDA_FAST_MATH=OFF \
  -D CUDA_TOOLKIT_ROOT_DIR="$COREX_PATH" \
  -D CUDA_ARCH_BIN=7.5 \
  -D CUDA_ARCH_PTX= \
  -D BUILD_opencv_cudacodec=OFF \
  -D BUILD_opencv_wechat_qrcode=OFF \
  -D OPENCV_GENERATE_PKGCONFIG=ON \
  -D OPENCV_ENABLE_NONFREE=ON \
  "${NPP_ARGS[@]}" \
  "$@"

# OpenCV FindCUDA may still leave nppc/npps as NOTFOUND on first pass — force stubs.
cmake . \
  -D CUDA_nppc_LIBRARY="$STUB_DIR/libnppc.so" \
  -D CUDA_npps_LIBRARY="$STUB_DIR/libnpps.so" \
  -D CMAKE_CUDA_FLAGS="--cuda-gpu-arch=ivcore11 -std=c++14"

cmake --build . -j "$JOBS"
cmake --install .
echo "OpenCV CoreX install: $PREFIX"
