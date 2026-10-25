const default_alpha = 1.0e-3 # Default value for the spread parameter
const default_beta = 2.0
const default_kappa = 0.0

struct UnscentedExtra{T, R, M, C}
    L::T
    λ::R
    Wm::M
    Wc::C
end

"""
    Unscented(; alpha = 1e-3, beta = 2.0, kappa = 0.0)
    Unscented(dim::Integer; alpha = 1e-3, beta = 2.0, kappa = 0.0)

The unscented transform: the mean and covariance of `f(x)` for a normal `x`, estimated from
`2d + 1` deterministically placed sigma points, `d` being the dimension of `x`. It needs no
derivatives and captures the mean and covariance of `f(x)` to second order. [`UT`](@ref) and
[`UnscentedTransform`](@ref) are aliases.

# Keywords

- `alpha`: the spread of the sigma points around the mean, a small positive number. Default
  `1e-3`;
- `beta`: prior knowledge of the input's distribution, `2` being optimal for a normal. Default
  `2.0`;
- `kappa`: the secondary scaling. Default `0.0`.

With `dim`, the method is for inputs of that dimension, and [`sigma_point_parameters`](@ref)
returns the weights it stores; an input of another dimension, or several inputs whose joint has
another dimension, is a `DimensionMismatch`.
Without it, the method takes inputs of any dimension. The sigma points and their weights are
computed on each call either way.

It is used through [`approximate`](@ref), for the output's moments, and
[`unscented_statistics`](@ref), which adds the cross-covariance [`smoothRTS`](@ref) needs.
[`Linearization`](@ref) is the alternative that expands the function instead.

# Examples

```jldoctest; setup = :(using MessagePassingRulesApproximations)
julia> m, V = approximate(Unscented(), sin, (0.0,), (1e-4,));

julia> isapprox(m, 0.0; atol = 1e-12) && isapprox(V, 1e-4; rtol = 1e-3)
true
```
"""
struct Unscented{A, B, K, E} <: AbstractApproximationMethod
    α::A
    β::B
    κ::K
    e::E
end

# Structure constructor
function Unscented(;
        alpha::A = default_alpha, beta::B = default_beta, kappa::K = default_kappa
    ) where {A <: Real, B <: Real, K <: Real}
    return Unscented{A, B, K, Nothing}(alpha, beta, kappa, nothing)
end

function Unscented(
        dim::Integer;
        alpha::Real = default_alpha,
        beta::Real = default_beta,
        kappa::Real = default_kappa,
    )
    α = alpha
    β = beta
    κ = kappa
    λ = α^2 * (dim + κ) - dim
    Wm = ones(2 * dim + 1)
    Wc = ones(2 * dim + 1)
    Wm ./= (2 * (dim + λ))
    Wc ./= (2 * (dim + λ))
    Wm[1] = λ / (dim + λ)
    Wc[1] = λ / (dim + λ) + (1 - α^2 + β)
    return Unscented(α, β, κ, UnscentedExtra(dim, λ, Wm, Wc))
end

"""
    UT

An alias for [`Unscented`](@ref), the unscented transform.
"""
const UT = Unscented

"""
    UnscentedTransform

An alias for [`Unscented`](@ref), the unscented transform.
"""
const UnscentedTransform = Unscented

approximation_name(::Unscented) = "Unscented"
approximation_short_name(::Unscented) = "UT"

# get-functions for the Unscented structure

getα(approximation::Unscented) = approximation.α
getβ(approximation::Unscented) = approximation.β
getκ(approximation::Unscented) = approximation.κ

getextra(approximation::Unscented) = approximation.e

getL(approximation::Unscented) = getL(getextra(approximation))
getλ(approximation::Unscented) = getλ(getextra(approximation))
getWm(approximation::Unscented) = getWm(getextra(approximation))
getWc(approximation::Unscented) = getWc(getextra(approximation))

getL(extra::UnscentedExtra) = extra.L
getλ(extra::UnscentedExtra) = extra.λ
getWm(extra::UnscentedExtra) = extra.Wm
getWc(extra::UnscentedExtra) = extra.Wc

"""
    sigma_point_parameters(method::Unscented, dim::Integer) -> (; λ, Wm, Wc)

The scaling `λ` and the weights of the mean, `Wm`, and of the covariance, `Wc`, of the `2 dim + 1`
sigma points `method` places for an input of dimension `dim`: those `method` carries when it was
built for `dim`, `Unscented(dim; …)`, and computed from its `alpha`, `beta` and `kappa` when it was
built without a dimension, `Unscented(; …)`. For a caller that places the sigma points itself.

# Throws

- `DimensionMismatch` when `method` was built for a dimension other than `dim`.

# Examples

```jldoctest
julia> p = MessagePassingRulesApproximations.sigma_point_parameters(Unscented(; alpha = 1.0), 2);

julia> p.λ, length(p.Wm), sum(p.Wm) ≈ 1
(0.0, 5, true)
```
"""
function sigma_point_parameters(method::Unscented, dim::Integer)
    extra = getextra(method) === nothing ? getextra(Unscented(dim; alpha = getα(method), beta = getβ(method), kappa = getκ(method))) : getextra(method)
    check_unscented_dimension(extra, dim)
    return (λ = getλ(extra), Wm = getWm(extra), Wc = getWc(extra))
end

# Copied and refactored from ForneyLab.jl

"""
    approximate(method::Unscented, f, means::Tuple, covs::Tuple) -> (m, V)
    approximate!(ws, method::Unscented, f, means, covs) -> (m, V)
    approximate!(m, V, ws, method::Unscented, f, means, covs) -> (m, V)

The mean and covariance of `f(x₁, x₂, …)` for independent normal inputs, by the unscented
transform. The in-place forms work in a [`workspace`](@ref) and allocate nothing of their own; see
[`approximate`](@ref)`(method, f, params...)`, which every point-set method shares.

# Arguments

- `f`: a function of one argument per input, returning a number or a vector;
- `means`, `covs`: one entry per input, a number and a variance, or a vector and a covariance
  matrix. Several inputs are treated jointly, as one normal whose covariance is block diagonal.

# Returns

`(m, V)`: the output's mean and variance, or mean vector and covariance matrix.

# Throws

A `DomainError` when a covariance is infinite. A zero covariance is not an error: the input is a
point, and the result is `f` at it with zero covariance, after a warning.
"""
approximate(method::Unscented, f::F, means::Tuple, covs::Tuple) where {F} = approximate_pointset(method, f, means, covs)

"""
    unscented_statistics(method::Unscented, g, means::Tuple, covs::Tuple) -> (m, V, C)

The output's mean `m` and covariance `V`, as [`approximate`](@ref)`(::Unscented, …)` computes
them, and the cross-covariance `C` between the inputs, concatenated into one vector, and the
output: the forward statistics [`smoothRTS`](@ref) takes. It is
[`forward_statistics`](@ref)`(method, g, means, covs)`, whose in-place form
[`forward_statistics!`](@ref) allocates nothing. Arguments and errors are those of `approximate`;
for a point input, `C` is zero.
"""
unscented_statistics(method::Unscented, g::G, means::Tuple, covs::Tuple) where {G} = forward_statistics(method, g, means, covs)

# `Val(false)` skips the cross-covariance, which is then `nothing`.
unscented_statistics(method::Unscented, ::Val{true}, g::G, means::Tuple, covs::Tuple) where {G} = forward_statistics(method, g, means, covs)
unscented_statistics(method::Unscented, ::Val{false}, g::G, means::Tuple, covs::Tuple) where {G} = (approximate(method, g, means, covs)..., nothing)

# A single input with an infinite covariance is an error; with a zero covariance it is a point:
# the transformed output is a point too, with zero covariance and zero cross-covariance.
degenerate_input(method::Unscented, means::Tuple{Real}, covs::Tuple{Real}) = degenerate_input(method, first(means), first(covs))
degenerate_input(method::Unscented, means::Tuple{AbstractVector}, covs::Tuple{AbstractMatrix}) = degenerate_input(method, first(means), first(covs))

function degenerate_input(::Unscented, m::Real, V::Real)
    isinf(V) && throw(DomainError(V, "the unscented transform cannot be computed with an infinite variance"))
    if V == 0.0
        @warn "Unscented transform called with zero covariance input" maxlog = 1
        return m
    end
    return nothing
end

function degenerate_input(::Unscented, m::AbstractVector, V::AbstractMatrix)
    any(isinf, V) && throw(DomainError(V, "the unscented transform cannot be computed with an infinite covariance"))
    if all(iszero, V)
        @warn "Unscented transform called with zero covariance input"
        return m
    end
    return nothing
end

# A method built for a dimension takes inputs of that dimension only.
check_unscented_dimension(method::Unscented, d) = check_unscented_dimension(getextra(method), d)
check_unscented_dimension(::Nothing, d) = nothing
check_unscented_dimension(extra::UnscentedExtra, d) =
    getL(extra) == d || throw(DimensionMismatch("`Unscented($(getL(extra)))` was built for dimension $(getL(extra)), not $d"))

## The point sets

# The `2d + 1` sigma points: the mean, then `m ± Lₖ` for the columns `Lₖ` of the lower Cholesky
# factor of `(d + λ) V`, alternating. `wm` and `wc` are the centre's weights in the mean and in
# the covariance, `wi` every other point's.
mutable struct SigmaPoints{T, L, W}
    m::T
    l::L
    wm::W
    wc::W
    wi::W
end

mutable struct SigmaPointsMv{T, W}
    m::Vector{T}
    F::Matrix{T}
    uplo::Char
    x::Vector{T}
    wm::W
    wc::W
    wi::W
end

# The weights for dimension `d`, warning when `d + λ` is negative.
function sigma_weights(method::Unscented, d)
    alpha = getα(method)
    beta = getβ(method)
    kappa = getκ(method)
    lambda = d == 1 ? (1 + kappa) * alpha^2 - 1 : (d + kappa) * alpha^2 - d
    if (d + lambda) < 0
        @warn "`(d + lambda)` in the sigma points computation routine is negative. This may lead to the incorrect results. Adjust the `alpha`, `kappa` and `beta` parameters."
    end
    wm = lambda / (d + lambda)
    wc = wm + (1 - alpha^2 + beta)
    wi = 1 / (2 * (d + lambda))
    return lambda, wm, wc, wi
end

function pointset(method::Unscented, m::Real, V::Real)
    check_unscented_dimension(method, 1)
    lambda, wm, wc, wi = sigma_weights(method, 1)
    return SigmaPoints(m, sqrt((1 + lambda) * V), promote(wm, wc, wi)...)
end

function pointset(method::Unscented, m::AbstractVector, V::AbstractMatrix)
    d = length(m)
    check_unscented_dimension(method, d)
    T = float(promote_type(eltype(m), eltype(V)))
    _, wm, wc, wi = sigma_weights(method, d)
    return SigmaPointsMv(Vector{T}(undef, d), Matrix{T}(undef, d, d), 'L', Vector{T}(undef, d), promote(wm, wc, wi)...)
end

function prepare!(points::SigmaPoints, method::Unscented, m::Real, V::Real)
    check_unscented_dimension(method, 1)
    lambda, points.wm, points.wc, points.wi = sigma_weights(method, 1)
    points.m = m
    points.l = sqrt((1 + lambda) * V)
    return points
end

function prepare!(points::SigmaPointsMv, method::Unscented, m::AbstractVector, V::AbstractMatrix)
    d = length(m)
    check_unscented_dimension(method, d)
    length(points.m) == d || throw(DimensionMismatch(lazy"the sigma points are for dimension $(length(points.m)), not $d"))
    lambda, points.wm, points.wc, points.wi = sigma_weights(method, d)
    copyto!(points.m, m)
    points.F .= (d + lambda) .* V
    points.uplo = cholesky_factor!(points.F, points.F)
    return points
end

npoints(::SigmaPoints) = 3
npoints(points::SigmaPointsMv) = 2 * length(points.m) + 1

point!(points::SigmaPoints, i) = i == 1 ? points.m : (iseven(i) ? points.m + points.l : points.m - points.l)

function point!(points::SigmaPointsMv, i)
    x, m = points.x, points.m
    if i == 1
        copyto!(x, m)
    else
        k, sign = i ÷ 2, iseven(i) ? 1 : -1
        @inbounds for r in eachindex(x)
            x[r] = r < k ? m[r] : (sign > 0 ? m[r] + lower_entry(points.F, points.uplo, r, k) : m[r] - lower_entry(points.F, points.uplo, r, k))
        end
    end
    return x
end

weight(points::Union{SigmaPoints, SigmaPointsMv}, i) = i == 1 ? points.wm : points.wi
covweight(points::Union{SigmaPoints, SigmaPointsMv}, i) = i == 1 ? points.wc : points.wi
center(points::Union{SigmaPoints, SigmaPointsMv}) = points.m

"""
    sigma_points_weights(method::Unscented, m::Real, V::Real) -> (points, weights_m, weights_c)
    sigma_points_weights(method::Unscented, m::AbstractVector, V::AbstractMatrix) -> (points, weights_m, weights_c)

The `2d + 1` sigma points of `N(m, V)` under `method`'s parameters, and their weights for the mean
and for the covariance, in the same order: tuples for a scalar normal, vectors otherwise. The
first point is the mean. It warns when the parameters make `d + λ` negative, which gives
unreliable estimates.

The point set of [`workspace`](@ref)`(method, f, m, V)` holds the same points without allocating
them, for the in-place operations.
"""
function sigma_points_weights(method::Unscented, m::Real, V::Real)
    points = prepare!(pointset(method, m, V), method, m, V)
    sigma_points = (point!(points, 1), point!(points, 2), point!(points, 3))
    weights_m = (weight(points, 1), weight(points, 2), weight(points, 3))
    weights_c = (covweight(points, 1), covweight(points, 2), covweight(points, 3))
    return (sigma_points, weights_m, weights_c)
end

function sigma_points_weights(method::Unscented, m::AbstractVector, V::AbstractMatrix)
    points = prepare!(pointset(method, m, V), method, m, V)
    n = npoints(points)
    sigma_points = [copy(point!(points, i)) for i in 1:n]
    weights_m = [weight(points, i) for i in 1:n]
    weights_c = [covweight(points, i) for i in 1:n]
    return (sigma_points, weights_m, weights_c)
end
