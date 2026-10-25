"""
    GaussHermiteCubature(p::Int)

Gauss–Hermite cubature with `p` points per dimension, for expectations under a normal given by
its mean and variance, or mean vector and covariance. It is exact for polynomials of degree up to
`2p - 1` in each coordinate. In `d` dimensions it evaluates `p^d` points, a tensor product, so
it suits a handful of dimensions only.

It provides [`getpoints`](@ref) and [`getweights`](@ref), and through them
[`approximate_meancov`](@ref). [`ghcubature`](@ref)`(p)` builds the same.

# Examples

```jldoctest; setup = :(using MessagePassingRulesApproximations)
julia> gh = GaussHermiteCubature(3);

julia> sum(w * x^2 for (w, x) in zip(getweights(gh, 1.0, 4.0), getpoints(gh, 1.0, 4.0))) ≈ 5.0  # E[x²] = m² + v
true
```
"""
struct GaussHermiteCubature{PI, WI} <: AbstractApproximationMethod
    piter::PI
    witer::WI
end

GaussHermiteCubature(p::Int) = ghcubature(p)

Base.show(io::IO, method::GaussHermiteCubature) = print(io, "GaussHermiteCubature(", length(method.piter), ")")

"""
    ghcubature(p::Int) -> GaussHermiteCubature

A [`GaussHermiteCubature`](@ref) with `p` points per dimension, its nodes and weights computed by
FastGaussQuadrature.
"""
function ghcubature(p::Int)
    points, weights = FastGaussQuadrature.gausshermite(p)
    return GaussHermiteCubature(points, weights)
end

approximation_name(gh::GaussHermiteCubature) = "GaussHermite($(length(gh.piter)))"
approximation_short_name(gh::GaussHermiteCubature) = "GH$(length(gh.piter))"

"""
    getweights(method, mean::Real, variance::Real)
    getweights(method, mean::AbstractVector, covariance::AbstractMatrix)

The cubature weights of `method` for a normal with these moments: a lazy generator of numbers
summing to one, in the order of [`getpoints`](@ref), so that `E[f(x)] ≈ Σ wᵢ f(xᵢ)`.
"""
function getweights end

"""
    getpoints(method, mean::Real, variance::Real)
    getpoints(method, mean::AbstractVector, covariance::AbstractMatrix)

The cubature points of `method` for a normal with these moments: a lazy generator of numbers, or
of vectors in the multivariate case, in the order of [`getweights`](@ref).

!!! warning
    The multivariate generator yields **one buffer**, rewritten on every iteration, so that a
    sum over the points allocates nothing. `collect` therefore returns copies of the last point;
    use `map(copy, getpoints(…))` to keep them. A consumer may overwrite the point it is given,
    as [`approximate_meancov`](@ref) does.
"""
function getpoints end

## The point sets

# A scalar normal: the nodes mapped to `m + √(2v) ξᵢ`, the weights divided by `√π`.
struct HermitePoints{N, T, S}
    nodes::N
    m::T
    scale::S
end

# A normal in `d` dimensions: the tensor-product grid of `p^d` points, `m + √2 L ξ`, with `L` the
# lower Cholesky factor of the covariance. A point's index is decoded into one node per dimension,
# the first dimension varying fastest.
mutable struct HermiteGridPoints{N, T, W}
    nodes::N
    m::Vector{T}
    F::Matrix{T}
    uplo::Char
    unit::Vector{T}
    x::Vector{T}
    # `π^(-d/2)`, which normalises a product of `d` weights.
    normalisation::W
end

# A value: built prepared.
pointset(gh::GaussHermiteCubature, m::Real, v::Real) = HermitePoints(gh, m, sqrt(2 * v))

function pointset(gh::GaussHermiteCubature, m::AbstractVector, V::AbstractMatrix)
    T = float(promote_type(eltype(m), eltype(V)))
    d = length(m)
    return HermiteGridPoints(gh, Vector{T}(undef, d), Matrix{T}(undef, d, d), 'L', Vector{T}(undef, d), Vector{T}(undef, d), inv(π^(d / 2)))
end

check_nodes(points, gh) = length(points.nodes.piter) == length(gh.piter) ||
    throw(DimensionMismatch(lazy"the point set is for $(length(points.nodes.piter)) nodes, not $(length(gh.piter))"))

# A scalar point set is a plain value: `prepare!` returns a new one.
function prepare!(points::HermitePoints{N, T, S}, gh::GaussHermiteCubature, m::Real, v::Real) where {N, T, S}
    check_nodes(points, gh)
    return HermitePoints{N, T, S}(gh, m, sqrt(2 * v))
end

function prepare!(points::HermiteGridPoints, gh::GaussHermiteCubature, m::AbstractVector, V::AbstractMatrix)
    length(m) == length(points.m) || throw(DimensionMismatch(lazy"the point set is for dimension $(length(points.m)), not $(length(m))"))
    check_nodes(points, gh)
    points.nodes = gh
    copyto!(points.m, m)
    points.uplo = cholesky_factor!(points.F, V)
    points.normalisation = inv(π^(length(m) / 2))
    return points
end

npoints(points::HermitePoints) = length(points.nodes.piter)
is_value(::HermitePoints) = true
npoints(points::HermiteGridPoints) = length(points.nodes.piter)^length(points.m)

point!(points::HermitePoints, i) = points.m + points.scale * points.nodes.piter[i]
weight(points::HermitePoints, i) = points.nodes.witer[i] / sqrt(π)
logweight(points::HermitePoints, i) = log(points.nodes.witer[i]) - log(π) / 2
center(points::HermitePoints) = points.m

function point!(points::HermiteGridPoints, i)
    p, index = length(points.nodes.piter), i - 1
    for k in eachindex(points.unit)
        points.unit[k] = points.nodes.piter[index % p + 1]
        index ÷= p
    end
    return lower_mul_add!(points.x, points.m, sqrt(2), points.F, points.uplo, points.unit)
end

function weight(points::HermiteGridPoints, i)
    p, index = length(points.nodes.piter), i - 1
    w = one(eltype(points.nodes.witer))
    for _ in eachindex(points.unit)
        w *= points.nodes.witer[index % p + 1]
        index ÷= p
    end
    return w * points.normalisation
end

function logweight(points::HermiteGridPoints, i)
    p, index = length(points.nodes.piter), i - 1
    lw = zero(float(eltype(points.nodes.witer)))
    for _ in eachindex(points.unit)
        lw += log(points.nodes.witer[index % p + 1])
        index ÷= p
    end
    return lw + log(points.normalisation)
end

center(points::HermiteGridPoints) = points.m

weightsign(::Union{HermitePoints, HermiteGridPoints}, i) = 1

## The lazy points and weights

getweights(gh::GaussHermiteCubature, mean::T, variance::T) where {T <: Real} =
    Base.Generator(weight -> weight / sqrt(π), gh.witer)

function getpoints(gh::GaussHermiteCubature, mean::T, variance::T) where {T <: Real}
    scale = sqrt(2 * variance)
    return Base.Generator(point -> mean + scale * point, gh.piter)
end

function getweights(gh::GaussHermiteCubature, mean::AbstractVector{T}, covariance::AbstractMatrix{T}) where {T <: Real}
    points = prepared_pointset(gh, mean, covariance)
    return Base.Generator(i -> weight(points, i), 1:npoints(points))
end

function getpoints(gh::GaussHermiteCubature, mean::AbstractVector{T}, covariance::AbstractMatrix{T}) where {T <: Real}
    points = prepared_pointset(gh, mean, covariance)
    return Base.Generator(i -> point!(points, i), 1:npoints(points))
end
