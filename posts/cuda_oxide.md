# Safe and Fast CUDA in Rust 

### date: 14 Jul, 2026
Recently, Nvidia released a cuda backend, called
[cuda-oxide](https://github.com/NVlabs/cuda-oxide), And its super cool! I have
finally jumped into the world of "Speed of Light"(SoL) kernels and started
learning how to write my own. I am personally very excited about cuda-oxide
because I have already spending a fairly large amount of my time learning rust
for the last several months, and being able to write cuda kernels in rust is
right at the intersection of the two biggest areas I have been learning / want
to learn recently. On my journey to write a SoL GEMM kernel I have been writing
different implementations starting from the most naive approach and building up
to one that takes full advantage of the newest hardware (it turns out, writing
SoL kernels is largely about taking advantage of the newest hardware features).

## Cuda Oxide
>cuda-oxide is an experimental compiler that demonstrates how CUDA SIMT kernels
 can be written natively in pure Rust -- no DSLs, no foreign language bindings
 -- and made available to the broader Rust community. The project is in an early
 stage (alpha) and under active development: you should expect bugs, incomplete
 features, and API breakage as we work to improve it. That said, we hope you'll
 try it in your own work and help shape its direction by sharing feedback on
 your experience.

Theres a great [youtube video](https://youtu.be/XyUFPHRQtyw) and accompanying
[gist](https://gist.github.com/nihalpasham/7c6b8b2c9e19790c218416b6ba4d9767#file-gemm-sol-rust-showcase-script-md)
from Nihal Pasham, one of the main engineers from Nvidia working on cuda-oxide,
that has both a simple explanation of what cuda-oxide is and what the goals are,
as well as an explanation of the different techniques and ideas that come
together to write a super fast GEMM kernel. 

One of the biggest ideas behind cuda-oxide is for it to feel like writing rust,
not CUDA C++, not a DSL, but idiomatic rust. With that in mind, I have been
trying to take full advantage of the type safety and other idioms that are so
important to rust. Heres the example from the README:
```rust
#[kernel]
pub fn map<T: Copy, F: Fn(T) -> T + Copy>(f: F, input: &[T], mut out: DisjointSlice<T>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = f(input[i]);
    }
}
```
One really nice 'rusty' thing going on here is the output type,
`DisjointSlice<T>`. When you go to write your output to global memory
at the end of computation you have to bounds check, as kernels are dispatched in
blocks and you may have not have perfectly block sized inputs. `.get_mut()` however
will just return `None` instead of `Some(T)`, a familar pattern in rust. You
might have also noticed that there is no unsafe code, which is great as CUDA C++
is **full** of pointer foot guns.

## Barriers

There are some rough edges still, as to be expected because cuda-oxide is still
in alpha (very usable with that in mind). Not every kernel is going to be able
to only have safe code, there are lots of capabilities that are _currently_ locked
behind unsafe, but I'm sure the plan is to add safe APIs for everything. One
thing that has recieved a (mostly) safe API are barriers. The barrier API
originally looked like:
```rust 
static mut BAR: Barrier = Barrier::UNINIT;

if thread::threadIdx_x() == 0 {
    unsafe {
        mbarrier_init(&mut BAR, 32);
        fence_proxy_async_shared_cta();
    }
}
thread::sync_threads()

// All threads arrive and wait
let token = unsafe { mbarrier_arrive(&BAR) };
unsafe { mbarrier_wait(&BAR, token); }

// all work done, invalidate barrier.
thread::sync_threads();
if thread::threadIdx_x() == 0 {
    let _bar = unsafe { mbarrier_inval(&mut BAR) }
}
thread::sync_threads();
```
Which, yeah, is a lot of unsafe code just to sync some barriers, and barriers
and incredibly common and important in the async tma/tcgen05 workloads that
actually dominate high performance kernels. Luckily, theres a better API now!
Its not totoally safe still, but only the initialization and invalidation are
unsafe.
```rust
static mut BAR: Barrier = Barrier::UNINIT;
let bar = ManagedBarrier<Uninit, Barrier>::from_static(&raw mut BAR);
// init is now thread safe, can be called on all threads
// the `fence_proxy_async_shared_cta()` also happens inside init.
let bar: ManagedBarrer<Ready, Barrier> = unsafe { bar.init(32) }; // type annotation unneeded
thread::sync_threads()

// All threads arrive and wait
let token = bar.arrive();
bar.wait(token);

// all work done, invalidate barrier
let bar: ManagedBarrer<Invalidated, Barrier> = unsafe { bar.inval() };
// trying to call bar again will result in a compile time error.
let token = bar.arrive()
```


## Usage in the Real World
Alright, in a real kernel, we need more than one barrier, the important ones are
a barrier to signal that the async memory transfer from global (GMEM) to shared
memory (SMEM), with the Tensor Memory Accelerator (TMA), has completed, and then
a barrier to signal that the shared memory block is no longer in use, which in a
GEMM kernel might be called the matrix multiply accumulate, or MMA barrier. Now
in fast kernels, you will usually see pipelining, where we have multiple
stages/waves of work happening, and each wave stage is going to need its own set
of barriers (and shared memory blocks).

I'm going to loosely outline our GEMM kernel, we have two warps (blocks of 32
threads that run in "lockstep"[add footnote about how this is not realy true
anymore]) running, one warp loads data with TMA, and the other warp dispatches
the tensor core operations on that data. Now, here's a simple way you could
**roughly** implement that.
```rust
if warp_id == TMA_WARP {
    let mut k = 0u32;
    while k < K / TILE_K {
        let stage = K % NUM_STAGES;
        if stage == 0 {
            mma_bar_0.wait(); 
            smem_load(TILE_A_0, tma_bar_0); // simplified greatly for example.
            smem_load(TILE_B_0, tma_bar_0);
            tma_bar_0.arrive();
        } else {
            mma_bar_1.wait(); 
            smem_load(TILE_A_1, tma_bar_1);
            smem_load(TILE_B_1, tma_bar_1);
            tma_bar_1.arrive();
        }
        k+=1;
    }
} 

if warp_id == MMA_WARP {
    let mut k = 0u32;
    while k < K / TILE_K {
        let stage = K % NUM_STAGES;
        if stage == 0 {
            tma_bar_0.wait(); 
            compute(TILE_A_0, TILE_B_0); // simplified greatly for example.
            mma_bar_0.arrive();
        } else {
            tma_bar_1.wait(); 
            compute(TILE_A_1, TILE_B_1); 
            mma_bar_1.arrive();
        }
        k+=1;
    }
} 
```
Okay, thats fine, but with pipelines that are usually four stages deep and
branches that in reality contain way more code, is there a better way? Of course
there is, taking advantage of one of my favorite rust features, pattern
matching! Lets look at the same code:
```rust
if warp_id == TMA_WARP {
    let mut k = 0u32;
    while k < K / TILE_K {
        let stage = K % NUM_STAGES;

        let (tile_a, tile_b, mma_bar, tma_bar) = match stage {
            0 => (&TILE_A_0, &TILE_B_0, &mma_bar_0, &tma_bar_0),
            _ => (&TILE_A_1, &TILE_B_1, &mma_bar_1, &tma_bar_1),
        };
        
        mma_bar.wait(); 
        smem_load(tile_a, tma_bar);
        smem_load(tile_b, tma_bar); 
        tma_bar.arrive();
        k+=1;
    }
} 

if warp_id == MMA_WARP {
    let mut k = 0u32;
    while k < K / TILE_K {
        let stage = K % NUM_STAGES;
        let (tile_a, tile_b, mma_bar, tma_bar) = match stage {
            0 => (&TILE_A_0, &TILE_B_0, &mma_bar_0, &tma_bar_0),
            _ => (&TILE_A_1, &TILE_B_1, &mma_bar_1, &tma_bar_1),
        };
        tma_bar_0.wait(); 
        compute(tile_a, tile_b);
        mma_bar_0.arrive();
        k+=1;
    }
} 
```
And thats just lovely! I atleast way prefer the look of that code and its very
familar to any rust programmers looking at it.


## THE CATCH
There is a catch though (in todays version of cuda-oxide), and that is
performance. The real version of those two kernels don't have the same
performance, at $M\times N\times K=8192^3$:

```
┌────────────────────────────────────┬───────────────────────┐
│              variant               │    throughput         │
├────────────────────────────────────┼───────────────────────┤
│ match yielding values (gemm_match) │ 679.3 TFLOP/s         │
├────────────────────────────────────┼───────────────────────┤
│ branch per stage (gemm_branch)     │ 729.5 TFLOP/s         │
└────────────────────────────────────┴───────────────────────┘
```

This is pretty reasonable/to be expected, cuda-oxide is very new and still in the alpha stage, but it does make me wonder if theres anything we can do about it. First we probably should look further into the cause though.

After looking into the PTX, the issue is pretty clear. Here's the
top of gemm_match and its TMA loop, trimmed to the interesting parts:
```
 .visible .entry gemm_match(
     .local .align 8 .b8  __local_depot1[64];   // 64 bytes of per-thread "stack"
     ...
     mov.b64  %SPL, __local_depot1;
     add.u64  %rd15, %SPL, 32;                  // address of a barrier's stack slot
     ...
     brx.idx  %r127, $L_brx_1;                  // jump table on `stage`
 $L__BB1_40:                                    // one match arm
     mov.b64  %rd211, __shared_mem_35;          // tile address: a constant. fine!
     mov.b64  %rd212, %rd15;                    // barrier: a pointer INTO THE DEPOT
     bra.uni  $L__BB1_41;
     ...
 $L__BB1_42:                                    // the mbarrier spin loop
     ld.local.b64  %rd148, [%rd212];            // reload barrier ptr from local mem
     { ... mbarrier.try_wait.parity.shared::cta.b64 p, [%rd148], %r129; ... }
     @%p35 bra  $L__BB1_42;                     // ...on EVERY poll
     ...
     ld.local.b64  %rd150, [%rd213];            // and again before the TMA copies
     cp.async.bulk.tensor.2d... [%rd33], ..., [%rd150];

```

The pick-1-of-N match statement has to become braches, loading from a table in
memory, or a chain of selects, and here we get the worst case. Instead of the
simple branched case, where each branch just has its own set of values that can
be easily assigned to registers by the compiler, the match case leaves the
barrier handles on the stack, and then indexes into that stack based on the
branch. This is absolutely brutal on performance, because getting data from the
local memory based stack is way slower than it might sound. This happens because
the barrier address gets locked behind a phi node (the compilers way of saying
"this value depends on which branch we arrived from"), preventing it from
promoting the value from the stack to a register, which it was able to do easily
in the if statement kernel.

Local memory is a bit of a weird term, its really referring to a per-thread
stack, which depending on whats going on with the cache at runtime, could be as
far away as GMEM, so paying the full HBM costs to access it. "Luckily", ours is
right in the middle of the hotloop and so should be kept in L1 pretty easily,
but thats still upwards of 30-40 cycles added to every iteration of the busy
waiting loop.

<details><summary>Claude's Compiler Optimization Adventure</summary>
<p>

## A Small Adventure in Compiler Land

This left me wondering, can we (Claude and I), make simple target improvements
to the code gen in cuda-oxide in order to fix this. The answer, pretty
unsuprisingly in this day and age, is yes, of course we can. Our new performance
looks like this:

```
┌────────────────────────────────────┬───────────────────────┐
│              variant               │    throughput         │
├────────────────────────────────────┼───────────────────────┤
│ match yielding values (gemm_match) │ 755.7 TFLOP/s         │
├────────────────────────────────────┼───────────────────────┤
│ branch per stage (gemm_branch)     │ 727.4 TFLOP/s         │
└────────────────────────────────────┴───────────────────────┘
```
Awesome, we're now beating gemm_branch pretty handily, and I'll let claude
explain why:

> The fix is two small passes that run on cuda-oxide's own IR, after Rust is
> lowered to the LLVM dialect but *before* the module is handed to LLVM. That
> placement is the whole trick. Nothing in LLVM was misbehaving — the shape was
> just outside what its passes are allowed to touch. SROA will only promote a
> stack slot to registers if it can account for every read and write, and an
> address escaping into a phi node ends that account. And LLVM's own
> if-conversion is cost-modeled for CPUs, where branch predictors make branches
> nearly free, so it refuses to speculate more than a couple of instructions to
> remove one. Rather than fight the middle-end, we hand it IR that no longer
> contains the trap.
> 
> **Pass 1: flatten the diamonds.** By the time a `match` reaches the IR, it's a
> cascade of compare-and-branch *diamonds*: a block ends in a conditional branch,
> each arm materializes its stage's addresses, and both jump to a merge block.
> The pass checks that every instruction in both arms is safe to execute
> unconditionally — address materialization, casts, comparisons; **no loads,
> stores, or calls** — and if so, hoists both arms above the branch and replaces
> the merge with `select` instructions: "compute both, keep one." A folded
> diamond becomes a pure straight-line block, which makes it a foldable arm of
> the *next* diamond out, so the whole 4-way cascade collapses on its own. This
> is a terrible trade on a CPU (you're doing 4× the work to avoid a nearly-free
> branch) and a great one on a GPU, where `selp` is a one-cycle ALU op and
> branches are anything but free.
> 
> **Pass 2: forward the slots.** The depot existed because the *address* of each
> barrier's stack slot flowed through the merge. But every one of those slots is
> written exactly once, at kernel entry, before any selecting happens — so the
> value you'd load is the same on every path. The pass proves that (single
> store, dominates every use, address never escapes anywhere it can't see), then
> forwards the *value* along the same control-flow edges the *pointer* was
> taking, mirroring every pointer-select with a value-select, and deletes the
> loads. The stores and stack slots are now dead code, LLVM's cleanup sweeps
> them away, and the depot simply never gets allocated.
> 
> Here's the same TMA loop as before, compiled with both passes:
 
```
$L__BB1_37:                               // top of the k-loop
    setp.eq.b32  %p40, %r10, 0;           // stage == 0?
    setp.eq.b32  %p41, %r10, 1;           // stage == 1?
    setp.eq.b32  %p42, %r10, 2;           // stage == 2?
    mov.b64   %rd92, __shared_mem_19;
    mov.b64   %rd93, __shared_mem_18;
    selp.b64  %rd94, %rd93, %rd92, %p42;  // 3-deep select chain picks the
    mov.b64   %rd95, __shared_mem_17;     // barrier address in registers —
    selp.b64  %rd96, %rd95, %rd94, %p41;  // no branch, no memory
    mov.b64   %rd97, __shared_mem_16;
    selp.b64  %rd98, %rd97, %rd96, %p40;
$L__BB1_38:                               // the mbarrier spin loop
    { ... mbarrier.try_wait.parity.shared::cta.b64 p, [%rd98], %r128; ... }
    @%p44 bra  $L__BB1_38;                // polls a register-held address
```
 
> No `__local_depot`, no `ld.local` — and look closer: no `brx.idx` either. The
> match compiled to the *third* selection mechanism, a chain of selects, and
> that's why it now beats `gemm_branch` rather than merely catching up to it.
> 
> `gemm_branch` still pays for its control flow. Every iteration starts with
> `brx.idx`, an indirect branch through a jump table — and a GPU has no branch
> predictor, so the warp's instruction fetch simply stalls until the target
> resolves. The four duplicated loop bodies also cost instruction-cache
> footprint, and since the pipeline rotates through stages 0→1→2→3, all four
> copies stay hot. The flattened match replaces all of that with about fifteen
> independent single-cycle ALU instructions that compute all four candidate
> addresses and throw three away. That sounds wasteful, and it is —
> deliberately. Redundant data flow is cheaper than control flow here: the
> selects issue back-to-back with no fetch stalls, right before a spin loop that
> was going to soak up any leftover latency anyway.
> 
> One thing worth noticing: `gemm_branch`'s numbers didn't move (727.4 vs 729.5
> is run-to-run noise), and that's the passes working as intended. Its match
> arms contain the actual TMA copies and barrier arrives — side effects — and
> the flatten pass refuses to speculate those, because "execute all four stages'
> copies every iteration" is not an optimization, it's a miscompile. Both passes
> are triggered by provable structure and do nothing otherwise: their failure
> mode is *didn't fire*, never *fired wrong*.
> 
> (There's also a third, one-line change: telling LLVM's optimizer not to
> convert switches into memory lookup tables — `--switch-to-lookup=false`.
> Without it, LLVM sometimes "helpfully" replaces a switch over shared-memory
> addresses with a table in global memory, which is both the slowest of the
> three selection mechanisms and, due to an address-space quirk in the NVPTX
> backend, generates PTX the driver rejects outright.)
> 
> So the rule of thumb the whole adventure boils down to: on a GPU, a
> pick-1-of-N wants to be *data flow*, not control flow or memory — and if the
> source language's nicest idiom doesn't naturally compile to that, the fix can
> live in a couple of surgical IR passes rather than in contorting your kernel
> code.

While its totally possible this optimization could produce unsound results if
enabled for everyone, its easy enough with the rust build system and modern
coding agents to maintain specific forks of the code-gen for a specific kernel,
where you can ensure numerical stability. I wouldn't recommend just ripping
coding agent PRs into your favorite open source projects though.

</p>
</details>


















```
┌────────────────────────────────────┬───────────────────────┬───────────────────────┐
│              variant               │    stock upstream     │         fork          │
├────────────────────────────────────┼───────────────────────┼───────────────────────┤
│ stage-indexed arrays (gemm_arith)  │ 780.2 TFLOP/s (1.00x) │ 780.1 TFLOP/s (1.00x) │
├────────────────────────────────────┼───────────────────────┼───────────────────────┤
│ match yielding values (gemm_match) │ 679.3 TFLOP/s (1.15x) │ 755.7 TFLOP/s (1.03x) │
├────────────────────────────────────┼───────────────────────┼───────────────────────┤
│ branch per stage (gemm_branch)     │ 729.5 TFLOP/s (1.07x) │ 727.4 TFLOP/s (1.07x) │
└────────────────────────────────────┴───────────────────────┴───────────────────────┘
```
