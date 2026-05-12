#!/usr/bin/env bash
set -euo pipefail

# Download the optional NYU CCD datasets that are not included in the default
# full-query download: rounded queries and full-scene TOI packages.
#
# Usage:
#   bash download_remaining_ccd_datasets.sh
#   bash download_remaining_ccd_datasets.sh --root ./datasets/continuous-collision-detection
#   bash download_remaining_ccd_datasets.sh --root /data/continuous-collision-detection --extract
#   bash download_remaining_ccd_datasets.sh --dry-run

ROOT="./datasets/continuous-collision-detection"
EXTRACT=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash download_remaining_ccd_datasets.sh [--root PATH] [--extract] [--dry-run]

Options:
  --root PATH   Dataset root. Default: ./datasets/continuous-collision-detection
  --extract     Extract each downloaded .tar.gz into ROOT after download.
  --dry-run     Print missing package list and total size, do not download.
  -h, --help    Show this help.

This script downloads only the optional remaining packages:
  - rounded CCD query archives
  - full-scene TOI archives
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT="$2"
      shift 2
      ;;
    --extract)
      EXTRACT=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      ROOT="$1"
      shift
      ;;
  esac
done

ARCHIVE_DIR="${ROOT}/nyu-full-dataset-archives"
MANIFEST="${ROOT}/download-remaining-manifest.csv"

TOTAL_BYTES=15056889937
ROUNDED_BYTES=5757762609
FULL_SCENE_TOI_BYTES=9299127328

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

file_size() {
  local path="$1"
  if have_cmd stat; then
    stat -c '%s' "$path" 2>/dev/null || stat -f '%z' "$path"
  else
    wc -c < "$path" | tr -d ' '
  fi
}

human_gib() {
  awk -v b="$1" 'BEGIN { printf "%.3f GiB", b / 1024 / 1024 / 1024 }'
}

write_manifest() {
  mkdir -p "$ROOT" "$ARCHIVE_DIR"
  {
    printf '"name","group","url","bytes"\n'
    while IFS='|' read -r group name url bytes; do
      [[ -z "${group}" || "${group}" =~ ^# ]] && continue
      printf '"%s","%s","%s","%s"\n' "$name" "$group" "$url" "$bytes"
    done <<'EOF'
rounded|rounded-ccd-queries-handcrafted.tar.gz|http://archive.nyu.edu/bitstream/2451/63808/2/rounded-ccd-queries-handcrafted.tar.gz|2697776
rounded|rounded-ccd-queries-simulation-chain-edge-edge.tar.gz|http://archive.nyu.edu/bitstream/2451/63808/6/rounded-ccd-queries-simulation-chain-edge-edge.tar.gz|2053652437
rounded|rounded-ccd-queries-simulation-chain-vertex-face.tar.gz|http://archive.nyu.edu/bitstream/2451/63808/7/rounded-ccd-queries-simulation-chain-vertex-face.tar.gz|485364865
rounded|rounded-ccd-queries-simulation-cow-heads.tar.gz|http://archive.nyu.edu/bitstream/2451/63808/3/rounded-ccd-queries-simulation-cow-heads.tar.gz|1215898062
rounded|rounded-ccd-queries-simulation-golf-ball.tar.gz|http://archive.nyu.edu/bitstream/2451/63808/4/rounded-ccd-queries-simulation-golf-ball.tar.gz|1168620332
rounded|rounded-ccd-queries-simulation-mat-twist.tar.gz|http://archive.nyu.edu/bitstream/2451/63808/5/rounded-ccd-queries-simulation-mat-twist.tar.gz|831529137
full-scene-toi|full-scene-README.md|http://archive.nyu.edu/bitstream/2451/74508/2/README.md|9778
full-scene-toi|armadillo-rollers.tar.gz|http://archive.nyu.edu/bitstream/2451/74508/3/armadillo-rollers.tar.gz|245015887
full-scene-toi|cloth-ball.tar.gz|http://archive.nyu.edu/bitstream/2451/74508/4/cloth-ball.tar.gz|317747290
full-scene-toi|cloth-funnel.tar.gz|http://archive.nyu.edu/bitstream/2451/74508/5/cloth-funnel.tar.gz|108956110
full-scene-toi|n-body-simulation.tar.gz|http://archive.nyu.edu/bitstream/2451/74508/6/n-body-simulation.tar.gz|561078239
full-scene-toi|puffer-ball-boxes+queries+mma_bool+roots.tar.gz|http://archive.nyu.edu/bitstream/2451/74508/8/puffer-ball-boxes%2bqueries%2bmma_bool%2broots.tar.gz|835698476
full-scene-toi|puffer-ball-frames.tar.gz|http://archive.nyu.edu/bitstream/2451/74508/9/puffer-ball-frames.tar.gz|2143336545
full-scene-toi|rod-twist-boxes+queries+mma_bool+roots.tar.gz|http://archive.nyu.edu/bitstream/2451/74508/7/rod-twist-boxes%2bqueries%2bmma_bool%2broots.tar.gz|252475210
full-scene-toi|rod-twist-frames-0-999.tar.gz|http://archive.nyu.edu/bitstream/2451/74508/10/rod-twist-frames-0-999.tar.gz|1207761770
full-scene-toi|rod-twist-frames-1000-1999.tar.gz|http://archive.nyu.edu/bitstream/2451/74508/11/rod-twist-frames-1000-1999.tar.gz|1208786531
full-scene-toi|rod-twist-frames-2000-2999.tar.gz|http://archive.nyu.edu/bitstream/2451/74508/12/rod-twist-frames-2000-2999.tar.gz|1208769684
full-scene-toi|rod-twist-frames-3000-4000.tar.gz|http://archive.nyu.edu/bitstream/2451/74508/13/rod-twist-frames-3000-4000.tar.gz|1209491808
EOF
  } > "$MANIFEST"
}

download_one() {
  local group="$1"
  local name="$2"
  local url="$3"
  local expected_bytes="$4"
  local dst="${ARCHIVE_DIR}/${name}"
  local part="${dst}.part"

  if [[ -f "$dst" ]]; then
    local existing_size
    existing_size="$(file_size "$dst")"
    if [[ "$existing_size" == "$expected_bytes" ]]; then
      printf '[skip] %s already exists (%s)\n' "$name" "$(human_gib "$existing_size")"
      return
    fi
    printf '[warn] %s exists but size is %s, expected %s. Keeping existing file; remove it to redownload.\n' \
      "$name" "$existing_size" "$expected_bytes" >&2
    return
  fi

  printf '[download] %s/%s %s\n' "$group" "$name" "$(human_gib "$expected_bytes")"
  if have_cmd curl; then
    curl -L --fail --retry 20 --retry-delay 10 --retry-all-errors -C - \
      --connect-timeout 60 -o "$part" "$url"
  elif have_cmd wget; then
    wget -c --tries=20 --waitretry=10 -O "$part" "$url"
  else
    printf '[error] curl or wget is required.\n' >&2
    exit 2
  fi

  local actual_bytes
  actual_bytes="$(file_size "$part")"
  if [[ "$actual_bytes" != "$expected_bytes" ]]; then
    printf '[error] size mismatch for %s: got %s, expected %s. Partial file kept: %s\n' \
      "$name" "$actual_bytes" "$expected_bytes" "$part" >&2
    exit 3
  fi

  mv -f "$part" "$dst"
  printf '[done] %s\n' "$dst"

  if [[ "$EXTRACT" -eq 1 && "$name" == *.tar.gz ]]; then
    printf '[extract] %s -> %s\n' "$name" "$ROOT"
    tar -xzf "$dst" -C "$ROOT"
  fi
}

write_manifest

printf 'Dataset root: %s\n' "$ROOT"
printf 'Archive dir : %s\n' "$ARCHIVE_DIR"
printf 'Manifest    : %s\n' "$MANIFEST"
printf 'Remaining packages: 18\n'
printf 'Rounded total      : %s\n' "$(human_gib "$ROUNDED_BYTES")"
printf 'Full-scene TOI     : %s\n' "$(human_gib "$FULL_SCENE_TOI_BYTES")"
printf 'Total download     : %s\n' "$(human_gib "$TOTAL_BYTES")"

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf '\nDry run package list:\n'
  column -s, -t < "$MANIFEST" 2>/dev/null || cat "$MANIFEST"
  exit 0
fi

while IFS='|' read -r group name url bytes; do
  [[ -z "${group}" || "${group}" =~ ^# ]] && continue
  download_one "$group" "$name" "$url" "$bytes"
done <<'EOF'
rounded|rounded-ccd-queries-handcrafted.tar.gz|http://archive.nyu.edu/bitstream/2451/63808/2/rounded-ccd-queries-handcrafted.tar.gz|2697776
rounded|rounded-ccd-queries-simulation-chain-edge-edge.tar.gz|http://archive.nyu.edu/bitstream/2451/63808/6/rounded-ccd-queries-simulation-chain-edge-edge.tar.gz|2053652437
rounded|rounded-ccd-queries-simulation-chain-vertex-face.tar.gz|http://archive.nyu.edu/bitstream/2451/63808/7/rounded-ccd-queries-simulation-chain-vertex-face.tar.gz|485364865
rounded|rounded-ccd-queries-simulation-cow-heads.tar.gz|http://archive.nyu.edu/bitstream/2451/63808/3/rounded-ccd-queries-simulation-cow-heads.tar.gz|1215898062
rounded|rounded-ccd-queries-simulation-golf-ball.tar.gz|http://archive.nyu.edu/bitstream/2451/63808/4/rounded-ccd-queries-simulation-golf-ball.tar.gz|1168620332
rounded|rounded-ccd-queries-simulation-mat-twist.tar.gz|http://archive.nyu.edu/bitstream/2451/63808/5/rounded-ccd-queries-simulation-mat-twist.tar.gz|831529137
full-scene-toi|full-scene-README.md|http://archive.nyu.edu/bitstream/2451/74508/2/README.md|9778
full-scene-toi|armadillo-rollers.tar.gz|http://archive.nyu.edu/bitstream/2451/74508/3/armadillo-rollers.tar.gz|245015887
full-scene-toi|cloth-ball.tar.gz|http://archive.nyu.edu/bitstream/2451/74508/4/cloth-ball.tar.gz|317747290
full-scene-toi|cloth-funnel.tar.gz|http://archive.nyu.edu/bitstream/2451/74508/5/cloth-funnel.tar.gz|108956110
full-scene-toi|n-body-simulation.tar.gz|http://archive.nyu.edu/bitstream/2451/74508/6/n-body-simulation.tar.gz|561078239
full-scene-toi|puffer-ball-boxes+queries+mma_bool+roots.tar.gz|http://archive.nyu.edu/bitstream/2451/74508/8/puffer-ball-boxes%2bqueries%2bmma_bool%2broots.tar.gz|835698476
full-scene-toi|puffer-ball-frames.tar.gz|http://archive.nyu.edu/bitstream/2451/74508/9/puffer-ball-frames.tar.gz|2143336545
full-scene-toi|rod-twist-boxes+queries+mma_bool+roots.tar.gz|http://archive.nyu.edu/bitstream/2451/74508/7/rod-twist-boxes%2bqueries%2bmma_bool%2broots.tar.gz|252475210
full-scene-toi|rod-twist-frames-0-999.tar.gz|http://archive.nyu.edu/bitstream/2451/74508/10/rod-twist-frames-0-999.tar.gz|1207761770
full-scene-toi|rod-twist-frames-1000-1999.tar.gz|http://archive.nyu.edu/bitstream/2451/74508/11/rod-twist-frames-1000-1999.tar.gz|1208786531
full-scene-toi|rod-twist-frames-2000-2999.tar.gz|http://archive.nyu.edu/bitstream/2451/74508/12/rod-twist-frames-2000-2999.tar.gz|1208769684
full-scene-toi|rod-twist-frames-3000-4000.tar.gz|http://archive.nyu.edu/bitstream/2451/74508/13/rod-twist-frames-3000-4000.tar.gz|1209491808
EOF

printf '[complete] all remaining optional CCD dataset packages downloaded.\n'
