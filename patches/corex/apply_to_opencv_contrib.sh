#!/usr/bin/env bash
# Apply CoreX cudev header patches onto an opencv_contrib tree.
set -euo pipefail
CONTRIB="${1:?usage: $0 /path/to/opencv_contrib-<ver>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cp "$HERE/saturate_cast.hpp" "$CONTRIB/modules/cudev/include/opencv2/cudev/util/saturate_cast.hpp"
cp "$HERE/shuffle.hpp" "$CONTRIB/modules/cudev/include/opencv2/cudev/warp/shuffle.hpp"
cp "$HERE/texture.hpp" "$CONTRIB/modules/cudev/include/opencv2/cudev/ptr2d/texture.hpp"
echo "Applied CoreX cudev patches to $CONTRIB"
