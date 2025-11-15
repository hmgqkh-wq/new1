#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"
STAGE="$BUILD/stage"
rm -rf "$STAGE"
mkdir -p "$STAGE/usr/lib" "$STAGE/usr/etc/exynosx940" "$STAGE/usr/share/doc/exynosx940"
find "$BUILD" -type f -name "libexynosx940complete*.so" -exec cp {} "$STAGE/usr/lib/libexynosx940complete.so" \; || true
cp -r "$ROOT/etc" "$STAGE/usr/" || true
cp -r "$ROOT/profiles" "$STAGE/usr/etc/" || true
cp "$ROOT/README.md" "$STAGE/usr/share/doc/"
cp "$ROOT/LICENSE" "$STAGE/usr/share/doc/"
OUT="$ROOT/exynosx940-completev3-android-arm64.tar.gz"
tar -C "$STAGE" -czf "$OUT" .
echo "Created $OUT"
