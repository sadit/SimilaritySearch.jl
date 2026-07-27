# This file is a part of SimilaritySearch.jl
# a simple heap for KnnHeap

"Index of the parent of the (1-based) binary-heap position `i`."
heapparent(i) = i>>1

"Index of the left child of the (1-based) binary-heap position `i`."
heapleft(i) = 2i

"""
    heapfix_up!(order, A, i)

Restores the heap property by moving the item at position `i` upwards (towards the root)
while it violates `order` with respect to its parent. `A` is the backing array and `order`
is a `Base.Order.Ordering` used to compare items (e.g., [`DistOrder`](@ref)).
"""
function heapfix_up!(order, A, i)
    @inbounds while (p = heapparent(i)) > 0
        if lt(order, A[p], A[i])
            heapswap!(A, i, p)
            i = p
        else
            break
        end
    end
    i
end

"""
    heapfix_down!(order, A, n)

Restores the heap property by moving the item at the root (position 1) downwards while
it violates `order` with respect to its children, considering only the first `n` elements
of `A`. `order` is a `Base.Order.Ordering` used to compare items.
"""
function heapfix_down!(order, A, n)
    i = 1
    @inbounds while (l = heapleft(i)) <= n
        r = l + 1
        if r > n || lt(order, A[r], A[l])
            lt(order, A[l], A[i]) && break
            heapswap!(A, i, l)
            i = l
        else # weight(A[l]) < weight(A[r])
            lt(order, A[r], A[i]) && break
            heapswap!(A, i, r)
            # @show "RIGHT ", (i, l, r, n), (A[i], A[l], A[r])
            i = r
        end

    end

    i
end

"""
    heapify!(order, A)

Rearranges `A` in place so that it satisfies the binary-heap property with respect to
`order`.
"""
function heapify!(order, A)
    for i in 2:length(A)
        heapfix_up!(order, A, i)
    end
end

"""
    heapsort!(order, A)

Sorts `A` in place using the heap it already contains (built with `heapify!`), repeatedly
moving the root to the end and restoring the heap on the remaining prefix. The result is
sorted in the reverse of `order` (i.e., the same array can be seen as a heap again by
calling `reverse!` on it).
"""
function heapsort!(order, A)
    for n in length(A):-1:2
        heapswap!(A, 1, n)
        heapfix_down!(order, A, n-1)
    end
end


"Swaps the elements of `A` at positions `i` and `j` in place."
function heapswap!(A, i::Integer, j::Integer)
    @inbounds A[i], A[j] = A[j], A[i]
end

"""
    isheap(order, A, i)

Checks whether the subtree rooted at position `i` of `A` satisfies the binary-heap
property with respect to `order` (only the immediate children of `i` are checked).
"""
function isheap(order, A, i)
    l = heapleft(i)
    r = l + 1
    n = length(A)
    (l > n || !lt(order, A[i], A[l])) && (r > n || !lt(order, A[i], A[r]))
end

"""
    isheap(order, A)

Checks whether `A` fully satisfies the binary-heap property with respect to `order`.
"""
function isheap(order, A)
    n = length(A)
    all(i -> isheap(order, A, i), 1:ceil(Int, n / 2))
end