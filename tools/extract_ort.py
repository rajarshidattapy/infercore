"""Extract the ONNX Runtime native library from a downloaded PyPI wheel.

Usage: python3 tools/extract_ort.py <wheel_dir> <out_dir>

CI uses this because the hosted runners in this account cannot reach
github.com release assets outside the owner's namespace; PyPI is reachable.
The C/C++ headers are vendored under third_party/onnxruntime/include (the
C API is stable across versions) and copied in alongside by the caller.
The resulting <out_dir> has the classic ORT layout:
  <out_dir>/lib/libonnxruntime.so*   (with .so / .so.1 symlinks)
  <out_dir>/include/onnxruntime/core/session/*.h
"""
import glob
import os
import re
import shutil
import sys
import zipfile

def main() -> None:
    pkg_dir, out_dir = sys.argv[1], sys.argv[2]
    whl = sorted(glob.glob(os.path.join(pkg_dir, "*.whl")))[-1]
    ver = re.search(r"onnxruntime-([0-9.]+)-", os.path.basename(whl)).group(1)

    lib = os.path.join(out_dir, "lib")
    os.makedirs(lib, exist_ok=True)

    with zipfile.ZipFile(whl) as z:
        for name in z.namelist():
            base = os.path.basename(name)
            if name.startswith("onnxruntime/capi/libonnxruntime") and base:
                with z.open(name) as src, open(os.path.join(lib, base), "wb") as dst:
                    shutil.copyfileobj(src, dst)

    real = next(f for f in os.listdir(lib) if f.startswith("libonnxruntime.so.1."))
    for alias in ("libonnxruntime.so.1", "libonnxruntime.so"):
        p = os.path.join(lib, alias)
        if os.path.lexists(p):
            os.remove(p)
        os.symlink(real, p)

    print(f"ONNX Runtime {ver} library extracted to {out_dir}")


if __name__ == "__main__":
    main()
