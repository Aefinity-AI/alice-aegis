import os
import glob

# Remove #![no_std] and extern crate alloc from all submodules
for file in glob.glob("/home/killboxincorporated/aegis-uefi/src/*.rs"):
    if file.endswith("main.rs"):
        continue
    with open(file, "r") as f:
        content = f.read()
    content = content.replace("#![no_std]\n", "")
    content = content.replace("extern crate alloc;\n", "")
    with open(file, "w") as f:
        f.write(content)

# attention.rs (libm)
with open("/home/killboxincorporated/aegis-uefi/src/attention.rs", "r") as f:
    content = f.read()
content = content.replace("base.powf(dim_ratio)", "libm::powf(base, dim_ratio)")
content = content.replace("theta.cos()", "libm::cosf(theta)")
content = content.replace("theta.sin()", "libm::sinf(theta)")
with open("/home/killboxincorporated/aegis-uefi/src/attention.rs", "w") as f:
    f.write(content)

# inference.rs (libm)
with open("/home/killboxincorporated/aegis-uefi/src/inference.rs", "r") as f:
    content = f.read()
content = content.replace("(head_dim as f32).sqrt()", "libm::sqrtf(head_dim as f32)")
with open("/home/killboxincorporated/aegis-uefi/src/inference.rs", "w") as f:
    f.write(content)

# ops.rs (libm)
with open("/home/killboxincorporated/aegis-uefi/src/ops.rs", "r") as f:
    content = f.read()
content = content.replace("((sum_sq / input.len() as f32) + eps).sqrt()", "libm::sqrtf((sum_sq / input.len() as f32) + eps)")
content = content.replace("(*v - max_val).exp()", "libm::expf(*v - max_val)")
with open("/home/killboxincorporated/aegis-uefi/src/ops.rs", "w") as f:
    f.write(content)

# Cargo.toml - add serde to hashbrown
with open("/home/killboxincorporated/aegis-uefi/Cargo.toml", "r") as f:
    content = f.read()
content = content.replace("hashbrown = { version = \"0.17.1\" }", "hashbrown = { version = \"0.17.1\", features = [\"serde\"] }")
with open("/home/killboxincorporated/aegis-uefi/Cargo.toml", "w") as f:
    f.write(content)
