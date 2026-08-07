# A4 TPU SMOKE (=M9a XLA gate): can Kaggle's free TPU train our ternary QAT
# architecture? Probes framework availability honestly: JAX first (usually
# preinstalled on TPU images), torch_xla second. Then times a mini ternary-QAT
# train step. The finding = which path is viable + measured step time.
import subprocess, sys, time
t0 = time.time()

print("=== A4 TPU smoke v2 ===", flush=True)
import os, glob
print("TPU env hints:", {k: v for k, v in os.environ.items() if "TPU" in k.upper() or "XLA" in k.upper()}, flush=True)
print("accel devfiles:", glob.glob("/dev/accel*"), "| vfio:", glob.glob("/dev/vfio/*"), flush=True)
if glob.glob("/dev/accel*") or any("TPU" in k.upper() for k in os.environ):
    print("TPU hardware hints PRESENT -> installing jax[tpu] runtime", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jax[tpu]",
                    "-f", "https://storage.googleapis.com/jax-releases/libtpu_releases.html"])
else:
    print("NO TPU hardware hints -> this worker is NOT a TPU VM (silent downgrade)", flush=True)

r = subprocess.run([sys.executable, "-c", "import jax; print(jax.__version__)"],
                   capture_output=True, text=True)
print("jax probe:", r.stdout.strip() or r.stderr.strip()[:200], flush=True)

JAX_OK = False
try:
    import jax, jax.numpy as jnp
    devs = jax.devices()
    print("jax devices:", devs, flush=True)
    JAX_OK = any("TPU" in str(d).upper() for d in devs)
except Exception as e:
    print("jax import/device failed:", repr(e)[:300], flush=True)

if JAX_OK:
    import numpy as np
    # mini ternary QAT step in pure JAX: absmean ternary + STE, tied-emb decoder-ish MLP proxy
    key = jax.random.PRNGKey(0)
    D, F, V, B, T = 512, 1408, 8192, 8, 512   # ~A4-class layer shapes
    params = {
        "emb": jax.random.normal(key, (V, D)) * 0.02,
        "w1": jax.random.normal(key, (D, F)) * 0.02,
        "w2": jax.random.normal(key, (F, D)) * 0.02,
    }
    def tern(w):
        g = jnp.mean(jnp.abs(w)) + 1e-8
        wq = jnp.clip(jnp.round(w / g), -1, 1) * g
        return w + jax.lax.stop_gradient(wq - w)   # STE
    def loss_fn(p, ids):
        x = p["emb"][ids]                            # B,T,D
        h = jax.nn.silu(x @ tern(p["w1"])) @ tern(p["w2"])
        logits = h @ p["emb"].T
        tgt = jnp.roll(ids, -1, axis=1)
        lp = jax.nn.log_softmax(logits, -1)
        return -jnp.mean(jnp.take_along_axis(lp, tgt[..., None], -1))
    ids = jax.random.randint(key, (B, T), 0, V)
    step = jax.jit(jax.grad(loss_fn))
    g = step(params, ids); jax.block_until_ready(g)  # compile
    t1 = time.time()
    N = 20
    for _ in range(N):
        g = step(params, ids)
    jax.block_until_ready(g)
    dt = (time.time() - t1) / N
    tok_s = B * T / dt
    print(f"JAX_TPU_QAT_STEP: {dt*1000:.1f} ms/step -> {tok_s:,.0f} tok/s (proxy layer, ternary STE)", flush=True)
    print("A4_SMOKE_JAX_PASS", flush=True)
else:
    print("A4_SMOKE_JAX_FAIL (no TPU device via jax)", flush=True)

r = subprocess.run([sys.executable, "-c", "import torch_xla; print(torch_xla.__version__)"],
                   capture_output=True, text=True)
print("torch_xla probe:", (r.stdout.strip() or r.stderr.strip().splitlines()[-1] if r.stderr else "none")[:200], flush=True)
print(f"done {time.time()-t0:.0f}s", flush=True)
