"""Extract the ONNX Runtime native library from a downloaded PyPI wheel and
fetch the matching C/C++ headers from jsDelivr's GitHub mirror.

Usage: python3 tools/extract_ort.py <wheel_dir> <out_dir>

CI uses this because the hosted runners in this account cannot reach
github.com release assets outside the owner's namespace; PyPI and jsDelivr
are reachable. The resulting <out_dir> has the classic ORT layout:
  <out_dir>/lib/libonnxruntime.so*   (with .so / .so.1 symlinks)
  <out_dir>/include/*.h
"""
import glob
import os
import re
import shutil
import sys
import urllib.request
import zipfile

HEADERS = [
    "onnxruntime_c_api.h",
    "onnxruntime_cxx_api.h",
    "onnxruntime_cxx_inline.h",
    "onnxruntime_ep_c_api.h",
    "onnxruntime_float16.h",
    "onnxruntime_lite_custom_op.h",
    "onnxruntime_run_options_config_keys.h",
]


def main() -> None:
    pkg_dir, out_dir = sys.argv[1], sys.argv[2]
    whl = sorted(glob.glob(os.path.join(pkg_dir, "*.whl")))[-1]
    ver = re.search(r"onnxruntime-([0-9.]+)-", os.path.basename(whl)).group(1)

    lib = os.path.join(out_dir, "lib")
    inc = os.path.join(out_dir, "include")
    os.makedirs(lib, exist_ok=True)
    os.makedirs(inc, exist_ok=True)

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

    for header in HEADERS:
        url = (
            "https://cdn.jsdelivr.net/gh/microsoft/onnxruntime@v"
            + ver
            + "/include/onnxruntime/core/session/"
            + header
        )
        urllib.request.urlretrieve(url, os.path.join(inc, header))

    print(f"ONNX Runtime {ver} extracted to {out_dir}")


if __name__ == "__main__":
    main()
