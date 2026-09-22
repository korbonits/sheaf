#!/usr/bin/env bash
# Build a Sheaf serving image on the ESM C optimization kit's pinned stack.
#
#   1. clone the kits tree at a pinned commit (SHEAF_KITS_REPO / SHEAF_KITS_REF)
#   2. build the kit's OWN image from its esmc/environment/Dockerfile, unchanged
#      (Ubuntu 24.04 + CUDA 13.0.1, Python 3.12.1, torch 2.11.0+cu130,
#      esm 3.4.0 @ 43ccece2, flash-attn 2.7.4.post1, TE 2.15.0, the kit
#      installed and its CUDA extension built)
#   3. layer sheaf-serve on top without moving any pinned package
#      (Dockerfile.esmc next to this script; the kit's pin check runs last)
#
# Usage (from anywhere; needs docker + git):
#   bash examples/docker/model-opt/build_esmc.sh [tag]      # default tag sheaf-esmc-kit:dev
#
# Optional: a pre-filled compile cache tar placed at
#   $SHEAF_KITS_DIR/_jitcache/esmc-torch2.11.0-cu130-sm90-jit.tar
# is baked into /opt/jit_cache by the kit's Dockerfile (see its STOCK.md).
set -euo pipefail

TAG=${1:-sheaf-esmc-kit:dev}
KITS_REPO=${SHEAF_KITS_REPO:-https://github.com/anthropics/uplifting-biomolecular-modeling.git}
KITS_REF=${SHEAF_KITS_REF:-f4f62fa6592ae4938d49b1757bea0cfeff9f468e}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
KITS_DIR=${SHEAF_KITS_DIR:-$HERE/.kits}

if [ ! -d "$KITS_DIR/.git" ]; then
  git clone --filter=blob:none --no-checkout "$KITS_REPO" "$KITS_DIR"
  git -C "$KITS_DIR" sparse-checkout set --no-cone /esmc /LICENSE /NOTICE /README.md
fi
git -C "$KITS_DIR" fetch --quiet origin "$KITS_REF" || true
git -C "$KITS_DIR" checkout --quiet "$KITS_REF"
echo "kits: $KITS_REPO @ $(git -C "$KITS_DIR" rev-parse HEAD)"

KIT_IMAGE="esmc-kit:${KITS_REF:0:12}"
docker build -f "$KITS_DIR/esmc/environment/Dockerfile" -t "$KIT_IMAGE" "$KITS_DIR"

docker build -f "$HERE/Dockerfile.esmc" \
  --build-arg KIT_IMAGE="$KIT_IMAGE" \
  --build-arg KITS_REPO="$KITS_REPO" \
  --build-arg KITS_REF="$KITS_REF" \
  --build-context kits="$KITS_DIR" \
  -t "$TAG" "$REPO"
echo "built $TAG on $KIT_IMAGE"
