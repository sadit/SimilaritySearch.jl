heapparent(i) = i>>1
heapleft(i) = 2i
#heapright(i) = heapleft(i)+1

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

function heapify!(order, A)
    for i in 2:length(A)
        heapfix_up!(order, A, i)
    end
end

function heapsort!(order, A)
    for n in length(A):-1:2
        heapswap!(A, 1, n)
        heapfix_down!(order, A, n-1)
    end
end


function heapswap!(A, i::Integer, j::Integer)
    @inbounds A[i], A[j] = A[j], A[i]
end

function isheap(order, A, i)
    l = heapleft(i)
    r = l + 1
    n = length(A)
    (l > n || !lt(order, A[i], A[l])) && (r > n || !lt(order, A[i], A[r]))
end

function isheap(order, A)
    n = length(A)
    all(i -> isheap(order, A, i), 1:ceil(Int, n / 2))
end