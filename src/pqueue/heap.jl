# This file is a part of SimilaritySearch.jl
# a simple heap for KnnHeap

"Index of the parent of the (1-based) binary-heap position `i`."
heapparent(i) = i>>1

"Index of the left child of the (1-based) binary-heap position `i`."
heapleft(i) = 2i

"""
    heapswap!(ids, dists, i, j)

Swaps the elements at positions `i` and `j` in both `ids` and `dists` in place,
keeping the two arrays in sync.
"""
@inline function heapswap!(ids, dists, i::Integer, j::Integer)
    @inbounds ids[i],   ids[j]   = ids[j],   ids[i]
    @inbounds dists[i], dists[j] = dists[j], dists[i]
end

"""
    heapfix_up!(order, ids, dists, i)

Restores the heap property by moving the item at position `i` upwards (towards the root)
while it violates `order` with respect to its parent. `ids` and `dists` are the parallel
backing arrays; `order` is a `Base.Order.Ordering` used to compare distances (e.g.,
[`DistOrder`](@ref)).
"""
function heapfix_up!(order, ids, dists, i)
    @inbounds while (p = heapparent(i)) > 0
        if lt(order, dists[p], dists[i])
            heapswap!(ids, dists, i, p)
            i = p
        else
            break
        end
    end
    i
end

"""
    heapfix_down!(order, ids, dists, n)

Restores the heap property by moving the item at the root (position 1) downwards while
it violates `order` with respect to its children, considering only the first `n` elements
of `ids`/`dists`. `order` is a `Base.Order.Ordering` used to compare distances.
"""
function heapfix_down!(order, ids, dists, n)
    i = 1
    @inbounds while (l = heapleft(i)) <= n
        r = l + 1
        if r > n || lt(order, dists[r], dists[l])
            lt(order, dists[l], dists[i]) && break
            heapswap!(ids, dists, i, l)
            i = l
        else
            lt(order, dists[r], dists[i]) && break
            heapswap!(ids, dists, i, r)
            i = r
        end
    end
    i
end

"""
    heapify!(order, ids, dists)

Rearranges `ids`/`dists` in place so that they satisfy the binary-heap property with
respect to `order`.
"""
function heapify!(order, ids, dists)
    for i in 2:length(ids)
        heapfix_up!(order, ids, dists, i)
    end
end

"""
    heapsort!(order, ids, dists)

Sorts `ids`/`dists` in place using the heap they already contain (built with `heapify!`),
repeatedly moving the root to the end and restoring the heap on the remaining prefix. The
result is sorted in the reverse of `order`.
"""
function heapsort!(order, ids, dists)
    for n in length(ids):-1:2
        heapswap!(ids, dists, 1, n)
        heapfix_down!(order, ids, dists, n - 1)
    end
end

"""
    isheap(order, ids, dists, i)

Checks whether the subtree rooted at position `i` satisfies the binary-heap property with
respect to `order` (only the immediate children of `i` are checked).
"""
function isheap(order, ids, dists, i)
    l = heapleft(i)
    r = l + 1
    n = length(ids)
    (l > n || !lt(order, dists[i], dists[l])) && (r > n || !lt(order, dists[i], dists[r]))
end

"""
    isheap(order, ids, dists)

Checks whether `ids`/`dists` fully satisfy the binary-heap property with respect to `order`.
"""
function isheap(order, ids, dists)
    n = length(ids)
    all(i -> isheap(order, ids, dists, i), 1:ceil(Int, n / 2))
end