#![cfg_attr(not(feature = "parallel"), no_std)]

extern crate alloc;

#[cfg(feature = "parallel")]
extern crate std;

// The CIS-1 chain — reference semantics, integer attention, the full-integer
// engine — and everything it needs (model/tokenizer/kv/json/arena/sampler) is
// deliberately ISA-independent: pure integer or plain Rust, no intrinsics.
// That portability IS the A25/A28 claim, so the whole chain compiles on every
// architecture. Ledger A28: the selftest digest reproduced on aarch64 with
// zero changes to any of these modules.
//
// Below the gate: the five modules that use x86 intrinsics directly
// (`ops`, its colskip/bitplane variants, the AVX2 CIS kernel) or lean on them
// (`inference`, the f32 production engine). A new architecture gets the
// portable chain first and earns its fast kernels second — bit-identity
// before speed, which is the project's whole thesis.
pub mod arena;
pub mod attention;
pub mod cis;
pub mod cis_attn;
pub mod cis_infer;
pub mod json;
pub mod kvcache;
pub mod model;
pub mod sampler;
pub mod tokenizer;

#[cfg(all(target_arch = "x86_64", not(target_os = "uefi")))]
pub mod cis_avx2;
// aarch64's first earned fast kernel (see doctrine above): NEON CIS-1 TMV,
// bit-identical to the reference by the same test discipline as cis_avx2.
#[cfg(target_arch = "aarch64")]
pub mod cis_neon;
#[cfg(target_arch = "x86_64")]
pub mod inference;
#[cfg(target_arch = "x86_64")]
pub mod ops;
#[cfg(target_arch = "x86_64")]
pub mod ops_bitplane;
#[cfg(target_arch = "x86_64")]
pub mod ops_colskip;

#[cfg(feature = "parallel")]
pub mod pool;

/// True when THIS build of the engine quantizes activations onto the int8
/// grid before each BitLinear (feature `int8_act`, on by default). Exposed
/// as a const because a downstream crate's own `int8_act` feature flag says
/// nothing about what the engine it links against was compiled with.
pub const INT8_ACT_ENABLED: bool = cfg!(feature = "int8_act");
