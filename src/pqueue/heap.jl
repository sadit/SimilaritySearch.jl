# This file is a part of SimilaritySearch.jl
# a generic simple heap 

"Index of the parent of the (1-based) binary-heap position `i`."
heapparent(i) = i>>1

"Index of the left child of the (1-based) binary-heap position `i`."
heapleft(i) = 2i

"""
    heapfix_up!(lt::Function, swap::Function, X, i)

Restores the heap property by moving the item at position `i` upwards (towards the root)
while it violates `lt` with respect to its parent. `X` is the backing array (or data structure),
`lt` checks the heap property, and `swap` swaps two items.
"""
function heapfix_up!(lt::Function, swap::Function, X, i)
    @inbounds while (p = heapparent(i)) > 0
        if lt(X, p, i)
            swap(X, i, p)
            i = p
        else
            break
        end
    end
    i
end

"""
    heapfix_down!(lt::Function, swap::Function, X, n)

Restores the heap property by moving the item at the root (position 1) downwards while
it violates `lt` with respect to its children, considering only the first `n` elements
of `X`.
"""
function heapfix_down!(lt::Function, swap::Function, X, n)
    i = 1
    @inbounds while (l = heapleft(i)) <= n
        r = l + 1
        if r > n || lt(X, r, l)
            lt(X, l, i) && break
            swap(X, i, l)
            i = l
        else
            lt(X, r, i) && break
            swap(X, i, r)
            i = r
        end
    end
    i
end

"""
    heapify!(lt::Function, swap::Function, X, n)

Rearranges `X[1:n]` in place so that it satisfies the binary-heap property with
respect to `lt`.
"""
function heapify!(lt::Function, swap::Function, X, n)
    for i in 2:n
        heapfix_up!(lt, swap, X, i)
    end
end

"""
    heapsort!(lt::Function, swap::Function, X, n)

Sorts `X[1:n]` in place using the heap it already contains (built with `heapify!`),
repeatedly moving the root to the end and restoring the heap on the remaining prefix. 
"""
function heapsort!(lt::Function, swap::Function, X, n)
    for i in n:-1:2
        swap(X, 1, i)
        heapfix_down!(lt, swap, X, i - 1)
    end
end

"""
    isheap(lt::Function, X, i, n)

Checks whether the subtree rooted at position `i` satisfies the binary-heap property with
respect to `lt` up to `n` items (only the immediate children of `i` are checked).
"""
function isheap(lt::Function, X, i, n)
    l = heapleft(i)
    r = l + 1
    (l > n || !lt(X, i, l)) && (r > n || !lt(X, i, r))
end

"""
    isheap(lt::Function, X, n)

Checks whether `X[1:n]` fully satisfy the binary-heap property with respect to `lt`.
"""
function isheap(lt::Function, X, n)
    all(i -> isheap(lt, X, i, n), 1:ceil(Int, n / 2))
end