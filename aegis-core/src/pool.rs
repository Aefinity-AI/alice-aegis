//! A minimal persistent worker pool for the row-parallel kernels.
//!
//! Decode issues ~210 parallel matvec calls per token. Spawning OS threads per
//! call costs ~30 us each — measured 52 ms/token at 8 threads, a large fraction
//! of a decode step. The workers here are spawned once and park on a condvar;
//! dispatching a job is a lock, a pointer store, and a notify_all.
//!
//! Only compiled with the `parallel` feature, which implies std. The no_std
//! UEFI unikernel never sees this file.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

type DynJob = dyn Fn(usize, usize) + Send + Sync;

/// Raw pointer to a job that lives on the dispatching thread's stack.
///
/// SAFETY: `broadcast` does not return until every worker has finished with
/// this pointer, so the referent outlives all uses. This is the same contract
/// `std::thread::scope` provides, hand-rolled because we reuse parked threads.
#[derive(Clone, Copy)]
struct JobPtr(*const DynJob);
unsafe impl Send for JobPtr {}
unsafe impl Sync for JobPtr {}

struct Shared {
    job: Mutex<Option<(JobPtr, u64)>>,
    cv: Condvar,
    done: Mutex<usize>,
    done_cv: Condvar,
    panicked: AtomicBool,
    shutdown: AtomicBool,
}

pub struct Pool {
    shared: Arc<Shared>,
    workers: usize,
    generation: AtomicUsize,
    handles: Mutex<Vec<std::thread::JoinHandle<()>>>,
}

impl Pool {
    fn new(workers: usize) -> Self {
        let shared = Arc::new(Shared {
            job: Mutex::new(None),
            cv: Condvar::new(),
            done: Mutex::new(0),
            done_cv: Condvar::new(),
            panicked: AtomicBool::new(false),
            shutdown: AtomicBool::new(false),
        });

        let mut handles = Vec::new();
        // Worker 0 is the calling thread; spawn workers 1..n.
        for id in 1..workers {
            let sh = Arc::clone(&shared);
            let h = std::thread::Builder::new()
                .name(std::format!("aegis-worker-{id}"))
                .spawn(move || {
                    let mut last_gen: u64 = 0;
                    loop {
                        let ptr = {
                            let mut guard = sh.job.lock().unwrap();
                            loop {
                                if sh.shutdown.load(Ordering::Acquire) {
                                    return;
                                }
                                match *guard {
                                    Some((p, job_gen)) if job_gen != last_gen => {
                                        last_gen = job_gen;
                                        break p;
                                    }
                                    _ => guard = sh.cv.wait(guard).unwrap(),
                                }
                            }
                        };

                        // SAFETY: the dispatcher blocks until `done` reaches
                        // workers-1, so the job outlives this call.
                        let result =
                            catch_unwind(AssertUnwindSafe(|| unsafe { (*ptr.0)(id, workers) }));
                        if result.is_err() {
                            sh.panicked.store(true, Ordering::Release);
                        }

                        // Always count down, even on panic, or the dispatcher
                        // would wait forever.
                        let mut d = sh.done.lock().unwrap();
                        *d += 1;
                        sh.done_cv.notify_all();
                    }
                })
                .expect("failed to spawn aegis worker");
            handles.push(h);
        }

        Pool {
            shared,
            workers,
            generation: AtomicUsize::new(0),
            handles: Mutex::new(handles),
        }
    }

    pub fn workers(&self) -> usize {
        self.workers
    }

    /// Run `f(worker_index, worker_count)` on every worker, including the
    /// calling thread, and return only once all have finished.
    ///
    /// `f` may borrow from the caller's stack: this call is a barrier.
    pub fn broadcast<'env, F>(&self, f: F)
    where
        F: Fn(usize, usize) + Send + Sync + 'env,
    {
        if self.workers <= 1 {
            f(0, 1);
            return;
        }

        // Erase the borrow lifetime so the parked workers can hold a pointer to
        // `f`. SAFETY: `broadcast` is a barrier — it does not return until every
        // worker has finished calling through this pointer, so `f` and anything
        // it borrows outlive all uses. Same contract as `std::thread::scope`.
        let job: &(dyn Fn(usize, usize) + Send + Sync + 'env) = &f;
        let ptr = JobPtr(unsafe {
            core::mem::transmute::<&(dyn Fn(usize, usize) + Send + Sync + 'env), *const DynJob>(job)
        });

        let job_gen = self.generation.fetch_add(1, Ordering::Relaxed) as u64 + 1;
        self.shared.panicked.store(false, Ordering::Release);
        *self.shared.done.lock().unwrap() = 0;
        {
            let mut guard = self.shared.job.lock().unwrap();
            *guard = Some((ptr, job_gen));
            self.shared.cv.notify_all();
        }

        // The dispatching thread is worker 0 — it takes a share of the work.
        let local = catch_unwind(AssertUnwindSafe(|| f(0, self.workers)));

        let mut d = self.shared.done.lock().unwrap();
        while *d < self.workers - 1 {
            d = self.shared.done_cv.wait(d).unwrap();
        }
        drop(d);

        if let Err(e) = local {
            std::panic::resume_unwind(e);
        }
        if self.shared.panicked.load(Ordering::Acquire) {
            panic!("aegis worker thread panicked");
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        {
            let _guard = self.shared.job.lock().unwrap();
            self.shared.shutdown.store(true, Ordering::Release);
            self.shared.cv.notify_all();
        }
        for h in self.handles.lock().unwrap().drain(..) {
            let _ = h.join();
        }
    }
}

/// Process-wide pool, sized from AEGIS_THREADS or available parallelism.
pub fn global() -> &'static Pool {
    use std::sync::OnceLock;
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| Pool::new(crate::ops::worker_threads()))
}
