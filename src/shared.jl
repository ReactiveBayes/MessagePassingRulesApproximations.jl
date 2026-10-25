# This function computes cumulative start indices in the tuple `sizes`, note that `()` acts as size 1`
function __starts_at(sizes::Tuple)
    return last(
        reduce(sizes; init = (0, ())) do result, size
            current, accumulated = result
            next = current + prod(size)
            return (next, (accumulated..., current + 1))
        end,
    )
end

Base.@propagate_inbounds __as_vec_copyto!(container, start, input::Real) = container[start] = input
Base.@propagate_inbounds __as_vec_copyto!(container, start, input::AbstractVecOrMat) = copyto!(container, start, input, 1, length(input))

# This function linearizes the `inputs` argument into one (potentially big) vector
# For example (1, 2) becomes `[ 1, 2 ]`
#             ([2, 3], 2) becomes `[ 2, 3, 2 ]`
#             ([2, 3], [ 1.0 0.0; 0.0 1.0 ]) becomes `[ 2.0, 3.0, 1.0, 0.0, 0.0, 1.0 ]`
#             and so on
function __as_vec(inputs::Tuple)
    sizes = prod.(size.(inputs))
    starts_at = __starts_at(sizes)
    total = last(starts_at) + prod(last(sizes)) - 1

    T = promote_type(eltype.(inputs)...)
    x = Vector{T}(undef, total)

    foreach(inputs, starts_at) do input, start
        @inbounds __as_vec_copyto!(x, start, input)
    end

    return x
end

# This function extracts elements from the linearized version of inputs from the `__as_vec` function
# In case of `size = ()` we extract a single element a treat it as it would be a Univariate variable
function __splitjoinelement(x::AbstractVector, start, size::Tuple{})
    return x[start]
end

# This function extracts elements from the linearized version of inputs from the `__as_vec` function
# In case of `size = (Int, )` we extract the specified amount of elements as a view and treat it as a Multivariate variable
function __splitjoinelement(x::AbstractVector, start, size::Tuple{Int})
    return view(x, start:(start + prod(size) - 1))
end

# This function extracts elements from the linearized version of inputs from the `__as_vec` function
# In case of `size = (Int, Int)` we extract the specified amount of elements, reshape it into a matrix and treat it as a Matrixvariate variable
function __splitjoinelement(x::AbstractVector, start, size::Tuple{Int, Int})
    return reshape(
        view(x, start:(start + prod(size) - 1)), first(size), last(size)
    )
end

# This function extracts elements from the linearized version of inputs from the `__as_vec` function all at once
# Essentially this is the `undo` operation for the `__as_vec`
# See also `__splitjoinelement`
function __splitjoin(x::AbstractVector, sizes::Tuple)
    return map(sizes, __starts_at(sizes)) do size, start
        return __splitjoinelement(x, start, size)
    end
end

# The joint mean and covariance of independent inputs: their means concatenated, their
# covariances on the block diagonal. `sizes` are the inputs' own shapes: `()` for a scalar.
function joint_mean_cov(means::Tuple, covs::Tuple)
    sizes = map(size, means)
    m = __as_vec(means)
    T = promote_type(eltype(m), map(eltype, covs)...)
    V = zeros(T, length(m), length(m))
    foreach(covs, __starts_at(sizes), sizes) do cov, start, size
        block = start:(start + prod(size) - 1)
        V[block, block] .= cov
    end
    return m, V, sizes
end

# The joint mean and covariance of independent inputs, into `m` and `V`: every entry written.
function joint_mean_cov!(m::AbstractVector, V::AbstractMatrix, means::Tuple, covs::Tuple)
    sizes = map(size, means)
    starts = __starts_at(sizes)
    foreach(means, starts) do mean, start
        @inbounds __as_vec_copyto!(m, start, mean)
    end
    fill!(V, zero(eltype(V)))
    foreach(covs, starts, sizes) do cov, start, size
        block = start:(start + prod(size) - 1)
        copy_block!(V, block, cov)
    end
    return m, V
end

copy_block!(V, block, cov::Real) = (V[first(block), first(block)] = cov; V)
function copy_block!(V, block, cov::AbstractMatrix)
    for (j, J) in enumerate(block), (i, I) in enumerate(block)
        V[I, J] = cov[i, j]
    end
    return V
end

## Square roots of covariances

# The lower Cholesky factor of `V`, into `F`: `fastcholesky!` of a copy, in place, so the same
# factor as `cholsqrt(V)`, with no allocation when `V` is positive definite. Returns how the
# factor is stored: 'L', the lower triangle of `F`, or 'U', its transpose in the upper triangle;
# the other triangle is not zeroed.
function cholesky_factor!(F::AbstractMatrix, V::AbstractMatrix)
    copyto!(F, V)
    C = fastcholesky!(F)
    C.factors === F || copyto!(F, C.factors)
    return C.uplo
end

# The entry `(r, k)` of the lower factor stored in `F`, zero above the diagonal.
@inline lower_entry(F, uplo, r, k) = r < k ? zero(eltype(F)) : (uplo == 'L' ? F[r, k] : F[k, r])

# `x = m + s L u`, with `L` the lower factor stored in `F`.
function lower_mul_add!(x::AbstractVector, m::AbstractVector, s, F::AbstractMatrix, uplo::Char, u::AbstractVector)
    @inbounds for r in eachindex(x)
        acc = zero(eltype(x))
        if uplo == 'L'
            for k in 1:r
                acc += F[r, k] * u[k]
            end
        else
            for k in 1:r
                acc += F[k, r] * u[k]
            end
        end
        x[r] = m[r] + s * acc
    end
    return x
end
