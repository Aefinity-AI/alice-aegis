//! Captures the git commit this crate was BUILT from and bakes it into the
//! binary as `env!("AEGIS_GIT_COMMIT")`, so `agent_trace`'s receipt
//! provenance no longer depends on the *generating process's* current
//! working directory at run time (see aegis-linux/examples/agent_trace.rs's
//! `commit_hash()`).
//!
//! Resolution order:
//!   1. A pre-set `AEGIS_GIT_COMMIT` env var (e.g. from a packaged build
//!      with no `.git` directory, pinned explicitly by whoever built it).
//!   2. `git rev-parse HEAD` run with cwd = `CARGO_MANIFEST_DIR` (this
//!      crate's own directory), i.e. the repo the binary is built from —
//!      never the directory `cargo` or the resulting binary happens to be
//!      invoked from later.
//!   3. The literal `unknown` if neither is available.
//!
//! Also registers `cargo:rerun-if-changed` on `.git/HEAD` and whatever ref
//! file it points to, so the baked-in commit is refreshed on the next
//! build after a commit/checkout — resolving `.git` as a file (worktrees)
//! by reading its `gitdir:` line.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=AEGIS_GIT_COMMIT");

    if let Ok(pinned) = std::env::var("AEGIS_GIT_COMMIT") {
        println!("cargo:rustc-env=AEGIS_GIT_COMMIT={pinned}");
        return;
    }

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let manifest_dir = Path::new(&manifest_dir);

    for p in git_watch_paths(manifest_dir) {
        println!("cargo:rerun-if-changed={}", p.display());
    }

    let commit = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(manifest_dir)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=AEGIS_GIT_COMMIT={commit}");
}

/// Find `.git/HEAD` (resolving `.git` as a file, i.e. a worktree, by
/// reading its `gitdir:` line) plus the ref file `HEAD` points to (e.g.
/// `.git/refs/heads/main` or, in a worktree, the shared repo's ref), so
/// `cargo:rerun-if-changed` fires on both a plain commit/checkout and a
/// worktree branch switch. Returns whatever paths could be resolved; never
/// panics if the repo layout is unexpected — the build still proceeds and
/// just misses fine-grained rebuild triggering in that case.
fn git_watch_paths(start: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();

    let dot_git = start.join(".git");
    let git_dir: PathBuf = if dot_git.is_dir() {
        dot_git
    } else if dot_git.is_file() {
        match std::fs::read_to_string(&dot_git) {
            Ok(contents) => {
                let resolved = contents
                    .lines()
                    .find_map(|l| l.strip_prefix("gitdir:"))
                    .map(str::trim)
                    .map(|p| {
                        let pb = PathBuf::from(p);
                        if pb.is_absolute() { pb } else { start.join(pb) }
                    });
                match resolved {
                    Some(p) => p,
                    None => return out,
                }
            }
            Err(_) => return out,
        }
    } else {
        // Not in a git checkout at all; nothing to watch.
        return out;
    };

    let head = git_dir.join("HEAD");
    if let Ok(contents) = std::fs::read_to_string(&head) {
        out.push(head.clone());
        if let Some(rest) = contents.trim().strip_prefix("ref:") {
            let ref_path = git_dir.join(rest.trim());
            out.push(ref_path);
        }
    }

    out
}
