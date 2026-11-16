#!/usr/bin/env python3
"""Export FlashInfer's SM12x GDN kernel for the optional llama.cpp CUDA provider."""

import argparse
import ctypes
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def build(args):
    import cutlass  # ty: ignore[unresolved-import]
    import cutlass.cute as cute  # ty: ignore[unresolved-import]
    import torch  # ty: ignore[unresolved-import]
    from cuda.bindings import driver as cuda_driver  # ty: ignore[unresolved-import]
    from cutlass.runtime import find_runtime_libraries  # ty: ignore[unresolved-import]

    output = args.output_dir.resolve()
    checkout = Path(__file__).resolve().parents[1]
    if output == checkout or checkout in output.parents:
        raise ValueError("Build output must be outside the source checkout")
    if not args.runtime_shim.is_file():
        raise FileNotFoundError(args.runtime_shim)
    cuda_root = args.cuda_root.resolve()
    if not (cuda_root / "include/cuda_runtime.h").is_file():
        raise FileNotFoundError(cuda_root / "include/cuda_runtime.h")
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) not in ((12, 0), (12, 1)):
        raise ValueError("This provider supports SM120 and SM121 only")
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    output.mkdir(parents=True, exist_ok=False)
    for library in find_runtime_libraries(enable_tvm_ffi=False):
        ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)

    original = importlib.import_module("flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_sm120")
    if original.__file__ is None:
        raise ImportError("FlashInfer GDN source is unavailable")
    source = Path(original.__file__).read_text()
    grouped_lookup = "q_head_idx = o_head_idx * num_q_heads // num_sab_heads"
    if source.count(grouped_lookup) != 1:
        raise RuntimeError("Unsupported FlashInfer source: expected one grouped Q-head lookup")
    # ggml repeats Q/K heads by modulo, rather than adjacent head groups.
    adapted = source.replace(grouped_lookup, "q_head_idx = o_head_idx % num_q_heads")
    source_path = output / "delta_rule_native_layout.py"
    source_path.write_text(adapted)
    module_name = "flashinfer.gdn_kernels.delta_rule_dsl.llama_native_layout"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(source_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    kernel = module._FullyFusedDeltaRuleSm120(
        True, True, True, False, cutlass.BFloat16, cu_seqlens_dtype=torch.int32
    )

    @cute.jit
    def host(qp: cute.Pointer, kp: cute.Pointer, vp: cute.Pointer, op: cute.Pointer,
             gp: cute.Pointer, bp: cute.Pointer, initialp: cute.Pointer, statep: cute.Pointer,
             workspacep: cute.Pointer, seqp: cute.Pointer,
             t: cutlass.Int32, hq: cutlass.Int32, hv: cutlass.Int32,
             sv1: cutlass.Int64, sv2: cutlass.Int64, scale: cutlass.Float32,
             workspace_bytes: cutlass.Constexpr, implementation: cutlass.Constexpr,
             stream: cuda_driver.CUstream):
        q = cute.make_tensor(qp, cute.make_layout((t, 128, hq), stride=(cutlass.Int64(hq) * 128, 1, 128)))
        k = cute.make_tensor(kp, cute.make_layout((128, t, hq), stride=(1, cutlass.Int64(hq) * 128, 128)))
        v = cute.make_tensor(vp, cute.make_layout((128, t, hv), stride=(1, sv2, sv1)))
        o = cute.make_tensor(op, cute.make_layout((128, t, hv), stride=(1, cutlass.Int64(hv) * 128, 128)))
        gates = cute.make_layout((cutlass.Int64(t) * hv,), stride=(1,))
        g = cute.make_tensor(gp, gates)
        beta = cute.make_tensor(bp, gates)
        state_layout = cute.make_layout((1, hv, 128, 128), stride=(cutlass.Int64(hv) * 16384, 16384, 128, 1))
        initial = cute.make_tensor(initialp, state_layout)
        state = cute.make_tensor(statep, state_layout)
        workspace = cute.make_tensor(workspacep, cute.make_layout((workspace_bytes,), stride=(1,)))
        seq = cute.make_tensor(seqp, cute.make_layout((2,), stride=(1,)))
        implementation(q, k, v, o, g, beta, state, initial, None, None, None,
                       workspace, seq, scale, hq, hq, hv, hv,
                       cutlass.Int32(1), cutlass.Int32(1), cutlass.Int32(0), hv, stream)

    def pointer(dtype, alignment=16):
        return cute.runtime.make_ptr(dtype, 256, cute.AddressSpace.gmem, assumed_align=alignment)

    name = "llama_gdn_bf16"
    type(kernel).kernel.set_name_prefix(name, remove_cutlass_symbol=True, keep_mangled_name=False)
    options = module.sm12x_compile_options(device)
    compiled = cute.compile(
        host, *[pointer(cutlass.BFloat16) for _ in range(4)],
        *[pointer(cutlass.Float32) for _ in range(4)],
        pointer(cutlass.Uint8, 128), pointer(cutlass.Int32, 8),
        cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0),
        cutlass.Int64(0), cutlass.Int64(0), cutlass.Float32(0), sm_count * 128, kernel,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False), options=options,
    )
    compiled.export_to_c(str(output), name, function_prefix=name)
    provider = r'''#include <array>
#include <mutex>
#include "llama_gdn_bf16.h"

struct provider {
    llama_gdn_bf16_Kernel_Module_t module;
    std::once_flag init;
};
static std::array<provider, 64> modules;

extern "C" int ggml_gdn_bf16_arch() { return NATIVE_ARCH; }

static int prepare(provider * & p) {
    int device = -1;
    auto rc = cudaGetDevice(&device);
    if (rc != cudaSuccess) { return int(rc); }
    if (device < 0 || device >= int(modules.size())) { return int(cudaErrorInvalidDevice); }
    p = &modules[device];
    std::call_once(p->init, [p] { llama_gdn_bf16_Kernel_Module_Load(&p->module); });
    return 0;
}

extern "C" int ggml_gdn_bf16_init() { provider * p; return prepare(p); }

extern "C" int ggml_gdn_bf16_launch(
        void * q, void * k, void * v, void * o, void * g, void * b,
        void * initial, void * state, void * workspace, void * seq,
        int32_t t, int32_t hq, int32_t hv, int64_t sv1, int64_t sv2,
        float scale, cudaStream_t stream) {
    provider * p;
    int rc = prepare(p);
    if (rc) { return rc; }
    return cute_dsl_llama_gdn_bf16_wrapper(
        &p->module, q, k, v, o, g, b, initial, state, workspace, seq,
        t, hq, hv, sv1, sv2, scale, stream);
}
'''.replace("NATIVE_ARCH", str(major * 100 + minor * 10))
    provider_path = output / "provider.cpp"
    provider_path.write_text(provider)
    library = output / "libggml-gdn-bf16.so"
    command = [
        args.cxx, "-std=c++17", "-O2", "-fPIC", "-shared",
        "-I" + str(cuda_root / "include"), "-I" + str(output),
        str(provider_path), str(args.runtime_shim.resolve()), str(output / (name + ".o")),
        "-L" + str(cuda_root / "lib64"), "-lcudart", "-ldl", "-pthread", "-Wl,-z,defs",
        "-Wl,-soname,libggml-gdn-bf16.so", "-o", str(library),
    ]
    subprocess.run(command, check=True)
    manifest = {
        "device_name": torch.cuda.get_device_name(device), "compute_capability": [major, minor],
        "sm_count": sm_count, "compile_options": repr(options), "link_command": command,
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "flashinfer-python", "nvidia-cutlass-dsl")},
        "original_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "adapted_source_sha256": hashlib.sha256(adapted.encode()).hexdigest(),
        "runtime_shim_sha256": hashlib.sha256(args.runtime_shim.read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(library)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory outside the checkout")
    parser.add_argument("--runtime-shim", type=Path, required=True,
                        help="cuDNN Frontend aot/sm120/src/runtime_shim.cpp")
    parser.add_argument("--cuda-root", type=Path, required=True, help="CUDA toolkit root")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--cxx", default="g++", help="C++ compiler")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
