```@meta
CurrentModule = SimilaritySearch
```

# Memory Placement: NUMA Nodes and Huge Pages

A machine with two or more processor sockets has two or more **NUMA nodes**. Each node is a group
of cores with the memory attached to it. Every core can read every byte, but memory attached to
another node costs more to reach: the request crosses the link between sockets, and that link has
less bandwidth than the local memory controller. A search reads the database and the adjacency at
random, from every thread at once. So where those arrays sit decides how much of that traffic
crosses the link.

Two rules of the operating system decide where they sit:

- **First touch.** Linux and Windows put a new page on the node of the thread that writes it first.
  An array written by one thread therefore lies entirely on that thread's node, however many
  threads read it later. macOS on Apple silicon has one memory pool (no NUMA), so the rule does not
  matter there.
- **Huge pages.** Linux can back a 2 MB-aligned stretch of a large allocation with one 2 MB page
  instead of 512 pages of 4 KB (transparent huge pages). Fewer pages means fewer address
  translations missing in the TLB, which a random-access search pays often. Many small allocations
  rarely get huge pages.

Since 1.6.5 the package writes its large stores **from all threads** (balanced mode, the default),
each thread a contiguous stretch. The first touch then spreads their pages evenly over the nodes.
The blocks of a [`BlockMatrixDatabase`](@ref) are also large enough (at least 32 MB) to sit on huge
pages. The previous behavior (normal mode) is kept, and either mode only changes **where** the
memory lies, never what it holds or what a search returns.

| | normal mode (`set_page_spread!(false)`, up to 1.6.4) | balanced mode (`set_page_spread!(true)`, default) |
| :--- | :--- | :--- |
| bulk `append_items!` into a `BlockMatrixDatabase` or a quantized database (also `SearchGraph`'s and `AsymmetricSearchGraph`'s `append_items!`, and `sqcodes(enc, X)`) | written item by item by the calling thread | room reserved first, then filled by all threads |
| `StaticAdjList(adj)` | written by the calling thread | filled by all threads |
| default block of a new `BlockMatrixDatabase` | 256 columns | at least 32 MB (the last block holds only what it needs) |
| where the pages end up | the calling thread's node | spread over the nodes the threads run on |

A bulk append of fewer than 16 384 items, a single `push_item!`, or a session with one thread
always takes the serial path: there is nothing worth spreading.

On a two-socket Xeon Silver 4216 (64 threads), five indexes of 600K vectors built one after
another in one process, with 8-bit codes at 384 dimensions, answered 1.14-1.30× the queries per
second in balanced mode, at the same recall and the same distance evaluations. The gain matched
copying the codes into one matrix from all threads. Small datasets, like the ones below, fit in
the caches and show no difference.

---

## Choosing the mode

The mode is global and applies to the structures built **after** the call. A structure keeps the
layout it was built with.

```julia
# SimilaritySearch v1.6
using Random

page_spread()                      # true: balanced mode is the default

X = rand(Xoshiro(1), Float32, 16, 40_000)

balanced = BlockMatrixDatabase(16, Float32)
append_items!(balanced, MatrixDatabase(X))   # filled by all threads

set_page_spread!(false)            # normal mode, as up to 1.6.4
normal = try
    db = BlockMatrixDatabase(16, Float32)    # 256-column blocks
    append_items!(db, MatrixDatabase(X))     # filled by this thread
    db
finally
    set_page_spread!(true)         # restore the default for the rest of the session
end

# same items either way; only the block size and the placement differ
@test all(i -> balanced[i] == normal[i], 1:size(X, 2))
@test size(normal.blocks[1], 2) == 256 && length(normal.blocks) == 157
@test length(balanced.blocks) == 1   # blocks of up to 2^19 columns of 64 bytes: one is enough
```

A graph built in either mode answers the same queries with the same results:

```julia
# SimilaritySearch v1.6
using Random
const Dist = SimilaritySearch.Dist

X = rand(Xoshiro(2), Float32, 8, 20_000)
Q = MatrixDatabase(rand(Xoshiro(3), Float32, 8, 100))
ctx = SearchGraphContext(; reporters=[])

function build(X)
    G = SearchGraph(Dist.SqL2(), BlockMatrixDatabase(8, Float32))
    append_items!(G, ctx, MatrixDatabase(X))
    G
end

G1 = build(X)
set_page_spread!(false)
G2 = try build(X) finally set_page_spread!(true) end

# the graphs themselves may differ (parallel insertion is not deterministic), but both stores hold
# the same vectors, and an exhaustive search over either gives the same answer
@test all(i -> G1.db[i] == G2.db[i], 1:size(X, 2))
E1 = searchbatch(ExhaustiveSearch(Dist.SqL2(), G1.db), GenericContext(), Q, 5)
E2 = searchbatch(ExhaustiveSearch(Dist.SqL2(), G2.db), GenericContext(), Q, 5)
@test E1 == E2
```

---

## Balancing an existing structure: `spreadcopy`

A structure that was written by one thread stays where it is. Examples are an index read back
from disk (the reader rebuilds every array from one thread), a store filled with `push_item!` one
item at a time, or anything built in normal mode. [`spreadcopy`](@ref) returns a copy written by all
threads, whatever the mode says:

- arrays of plain values, [`MatrixDatabase`](@ref);
- [`BlockMatrixDatabase`](@ref), rebuilt with huge-page blocks (a database from 1.6.4 has 256-column
  blocks);
- quantized databases (codes, code sums and per-vector parameters);
- [`AdjList`](@ref) and [`StaticAdjList`](@ref);
- [`SearchGraph`](@ref) and [`AsymmetricSearchGraph`](@ref): database and adjacency, with hints,
  search parameters and members copied as they are.

```julia
# SimilaritySearch v1.6
using Random
const Dist = SimilaritySearch.Dist

X = rand(Xoshiro(4), Float32, 8, 20_000)
Q = MatrixDatabase(rand(Xoshiro(5), Float32, 8, 100))
ctx = SearchGraphContext(; reporters=[])

G = SearchGraph(Dist.SqL2(), MatrixDatabase(X))
index!(G, ctx)
# freeze the adjacency into one compact array pair (filled by all threads in balanced mode)
S = SearchGraph(G.dist, G.db, StaticAdjList(G.adj), G.hints, G.algo, G.len, G.members)

S2 = spreadcopy(S)                 # same graph, its arrays written by all threads
@test searchbatch(S2, ctx, Q, 10) == searchbatch(S, ctx, Q, 10)
@test S2.db.matrix == S.db.matrix && S2.db.matrix !== S.db.matrix
```

For an index stored with JLD2 (see [Index Persistence](persistence.md)), copy it once after
loading:

```julia
using JLD2
G = spreadcopy(JLD2.load("index.jld2", "G"))
```

The copy needs the memory of the structure twice while it runs; the original can be released
afterwards.

---

## Where to use it, and where not

**Balanced mode (the default) and `spreadcopy` pay off when all of these hold:**

- the machine has two or more NUMA nodes (`lscpu` reports `NUMA node(s): 2` or more; `numactl -H`
  lists them);
- the index is much larger than the last-level cache (hundreds of MB or more): only then do the
  reads go to memory at all;
- many threads search at once, as `searchbatch` does, so the traffic of all of them adds up on the
  memory controllers.

Typical cases: a server with two or more sockets answering batches of queries; an index loaded from
disk (`spreadcopy` it); a process that builds or loads several indexes one after another. In a
long process the allocator reuses memory the first structures freed, and that memory carries the
node of whoever touched it first. Balanced writes do not depend on that history.

**It brings nothing, or works against you, in these cases:**

- **One NUMA node**: a laptop, a single-socket server, Apple silicon. There is nothing to spread.
  The larger blocks still help a little with huge pages on Linux, and nothing is lost.
- **Small data** that fits in the caches (the examples on this page): the placement of memory that
  is never read from memory does not matter. Below 16 384 items the stores are filled serially
  anyway.
- **One searching thread**, or a process pinned to one node (`numactl --cpunodebind=0
  --membind=0`, or a scheduler that gives a job one socket). Local memory is the fastest for a
  thread that stays on its node. Spreading puts half of the pages on the other node, so here normal
  mode plus pinning the process is the better choice.
- **Memory-mapped storage** ([`MMapMatrixDatabase`](@ref)): its pages belong to the operating
  system's file cache, which the first-touch rule does not govern. `spreadcopy` has no method for
  it. Copy it into a `MatrixDatabase` if it fits in memory.
- **Growing one item at a time** (`push_item!`): those items are not spread. If an index grows that
  way for a long time and then serves many queries, `spreadcopy` it once it settles.
- **Comparing with numbers measured before 1.6.5**: build with `set_page_spread!(false)` so that
  the memory layout matches.
- **Memory tightly bounded**: `spreadcopy` holds two copies while it runs.

Linux's automatic NUMA balancing (`numa_balancing=1`) does not fix this by itself. It moves pages
toward the threads that use them, and every thread reads every part of a shared index.
