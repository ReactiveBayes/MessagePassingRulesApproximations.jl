"""
    GenUnscented()

The generalized unscented transform (Ebeigbe et al., 2021): `2d + 1` sigma points whose weights
match the input's mean and covariance and the third and fourth central moments of each of its
components, so it is exact for polynomials of degree three, and of degree four in each
component. [`GenUT`](@ref) and [`GenUnscentedTransform`](@ref) are aliases.

Its measure is given by `params = (m, V, S, K)`: the mean, the covariance, and the vectors of the
components' third and fourth central moments, `S = E[(x - m)³]` and `K = E[(x - m)⁴]`, numbers for
a scalar input. Several independent inputs take tuples, `(means, covs, skews, kurts)`. Without
`S` and `K`, `params = (m, V)` or `(means, covs)`, the input is normal: `S = 0` and `K = 3σ⁴`.
A `K` below the smallest one the skewness allows is raised to it.

It is used through [`approximate`](@ref), [`forward_statistics`](@ref), [`expectation`](@ref) and
[`approximate_meancov`](@ref), with their in-place forms; [`sigma_points_weights`](@ref) returns
its points. [`GenConstrainedUnscented`](@ref) keeps the points within bounds.

Ebeigbe, Berry, Norton and Simon (2021), *A generalized unscented transformation for
probability distributions*, arXiv:2104.01958; *Physical Review E* 111 (2025).

# Examples

```jldoctest; setup = :(using MessagePassingRulesApproximations)
julia> import MessagePassingRulesApproximations: GenUnscented

julia> m, V = approximate(GenUnscented(), x -> x^2, 1.0, 2.0);   # x ~ N(1, 2): E[x²] = 3

julia> m ≈ 3.0 && V ≈ 4 * 1.0^2 * 2.0 + 2 * 2.0^2   # Var[x²] = 4m²v + 2v²
true
```
"""
struct GenUnscented <: AbstractApproximationMethod end

"""
    GenUT

An alias for [`GenUnscented`](@ref), the generalized unscented transform.
"""
const GenUT = GenUnscented

"""
    GenUnscentedTransform

An alias for [`GenUnscented`](@ref), the generalized unscented transform.
"""
const GenUnscentedTransform = GenUnscented

"""
    GenConstrainedUnscented(lower, upper; shrink = 0.5)

The generalized unscented transform, [`GenUnscented`](@ref), with its sigma points kept within
the bounds `lower` and `upper`: numbers for a scalar input, vectors for a vector one, and tuples
of those, one entry per input, for several; `-Inf` and `Inf` leave a component unbounded. A point
that would fall outside moves towards the mean, to `shrink` times its distance from the mean to
the bound, and the point opposite moves so that the skewness stays matched when it can. The mean
and covariance are matched whatever the bounds; the mean must lie within them.

[`GenCUT`](@ref) is an alias. Its parameters are those of [`GenUnscented`](@ref).

# Examples

```jldoctest; setup = :(using MessagePassingRulesApproximations)
julia> import MessagePassingRulesApproximations: GenConstrainedUnscented, sigma_points_weights

julia> points, weights = sigma_points_weights(GenConstrainedUnscented(0.0, Inf), 1.0, 4.0);

julia> all(>=(0), points) && sum(weights) ≈ 1 && sum(weights .* points) ≈ 1.0
true
```
"""
struct GenConstrainedUnscented{L, U, R} <: AbstractApproximationMethod
    lower::L
    upper::U
    shrink::R

    function GenConstrainedUnscented(lower::L, upper::U; shrink::R = 0.5) where {L, U, R <: Real}
        0 < shrink < 1 || throw(ArgumentError(lazy"`shrink` must be between 0 and 1, not $shrink"))
        all(map((l, u) -> all(l .< u), __as_vec(lowertuple(lower)), __as_vec(lowertuple(upper)))) ||
            throw(ArgumentError("every lower bound must be below its upper bound"))
        return new{L, U, R}(lower, upper, shrink)
    end
end

lowertuple(bound::Tuple) = bound
lowertuple(bound) = (bound,)

"""
    GenCUT

An alias for [`GenConstrainedUnscented`](@ref).
"""
const GenCUT = GenConstrainedUnscented

const AnyGenUnscented = Union{GenUnscented, GenConstrainedUnscented}

approximation_name(::GenUnscented) = "GenUnscented"
approximation_short_name(::GenUnscented) = "GenUT"
approximation_name(::GenConstrainedUnscented) = "GenConstrainedUnscented"
approximation_short_name(::GenConstrainedUnscented) = "GenCUT"

## The point sets

# A scalar input: the mean, `m - u σ` and `m + v σ`, with weights `w₀`, `w⁻` and `w⁺`.
mutable struct GenSigmaPoints{T}
    m::T
    σ::T
    u::T
    v::T
    w0::T
    wminus::T
    wplus::T
end

# A vector input: the mean, then `m - uₖ Lₖ` for every column `Lₖ` of the lower Cholesky factor
# `L` of the covariance, then `m + vₖ Lₖ`. `a = (L∘³)⁻¹ S` is the standardised skewness and
# `c = (L∘⁴)⁻¹ K` the standardised kurtosis; `S` and `K` hold a normal's moments when none are
# given.
mutable struct GenSigmaPointsMv{T}
    m::Vector{T}
    L::Matrix{T}
    L3::Matrix{T}
    L4::Matrix{T}
    S::Vector{T}
    K::Vector{T}
    a::Vector{T}
    c::Vector{T}
    u::Vector{T}
    v::Vector{T}
    wminus::Vector{T}
    wplus::Vector{T}
    w0::T
    x::Vector{T}
    lower::Vector{T}
    upper::Vector{T}
end

pointset(method::AnyGenUnscented, m::Real, V::Real) = pointset(method, m, V, zero(V), 3 * V^2)

function pointset(::AnyGenUnscented, m::Real, V::Real, S::Real, K::Real)
    T = float(promote_type(typeof(m), typeof(V), typeof(S), typeof(K)))
    z = zero(T)
    return GenSigmaPoints(T(m), z, z, z, z, z, z)
end

pointset(method::AnyGenUnscented, m::AbstractVector, V::AbstractMatrix) = pointset(method, m, V, zeros(eltype(V), length(m)), zeros(eltype(V), length(m)))

function pointset(method::AnyGenUnscented, m::AbstractVector, V::AbstractMatrix, S::AbstractVector, K::AbstractVector)
    T = float(promote_type(eltype(m), eltype(V), eltype(S), eltype(K)))
    d = length(m)
    vector() = Vector{T}(undef, d)
    matrix() = Matrix{T}(undef, d, d)
    lower, upper = bound_vectors(method, T, d)
    return GenSigmaPointsMv(vector(), matrix(), matrix(), matrix(), vector(), vector(), vector(), vector(), vector(), vector(), vector(), vector(), zero(T), vector(), lower, upper)
end

# The bounds' buffers, which `constrain!` writes on every call.
bound_vectors(::AnyGenUnscented, ::Type{T}, d) where {T} = (Vector{T}(undef, d), Vector{T}(undef, d))

# A normal's third and fourth central moments.
prepare!(points::GenSigmaPoints, method::AnyGenUnscented, m::Real, V::Real) = prepare!(points, method, m, V, zero(V), 3 * V^2)

function prepare!(points::GenSigmaPointsMv, method::AnyGenUnscented, m::AbstractVector, V::AbstractMatrix)
    fill!(points.S, zero(eltype(points.S)))
    for k in eachindex(points.K)
        points.K[k] = 3 * V[k, k]^2
    end
    return prepare_moments!(points, method, m, V)
end

function prepare!(points::GenSigmaPoints, method::AnyGenUnscented, m::Real, V::Real, S::Real, K::Real)
    σ = sqrt(V)
    s = S / σ^3
    # The smallest fourth moment the skewness allows, raised slightly so that `u` stays positive.
    smallest = σ^4 * s^2
    K = K <= smallest ? 1.001 * smallest : K
    u = (-s + sqrt(4 * K / V^2 - 3 * s^2)) / 2
    v = u + s
    points.m, points.σ = m, σ
    points.u, points.v = constrain_scalar(method, m, σ, u, v, s)
    set_scalar_weights!(points)
    return points
end

function set_scalar_weights!(points::GenSigmaPoints)
    u, v = points.u, points.v
    points.wplus = inv(v * (u + v))
    points.wminus = (v / u) * points.wplus
    points.w0 = 1 - points.wplus - points.wminus
    return points
end

function prepare!(points::GenSigmaPointsMv, method::AnyGenUnscented, m::AbstractVector, V::AbstractMatrix, S::AbstractVector, K::AbstractVector)
    copyto!(points.S, S)
    copyto!(points.K, K)
    return prepare_moments!(points, method, m, V)
end

# The sigma points for the moments in `points.S` and `points.K`.
function prepare_moments!(points::GenSigmaPointsMv, method::AnyGenUnscented, m::AbstractVector, V::AbstractMatrix)
    d = length(points.m)
    length(m) == d || throw(DimensionMismatch(lazy"the sigma points are for dimension $d, not $(length(m))"))
    copyto!(points.m, m)
    L, L3, L4, a, c, u, v = points.L, points.L3, points.L4, points.a, points.c, points.u, points.v
    lower_factor!(L, V)
    L3 .= L .^ 3
    L4 .= L .^ 4
    copyto!(a, points.S)
    ldiv!(LowerTriangular(L3), a)
    # The smallest fourth moments the skewness allows, `L∘⁴ a²`, raised slightly.
    c .= a .^ 2
    mul!(u, L4, c)
    for k in eachindex(c)
        c[k] = points.K[k] < u[k] ? 1.001 * (u[k] + 1.0e-8) : points.K[k]
    end
    ldiv!(LowerTriangular(L4), c)
    for k in eachindex(u)
        u[k] = (-a[k] + sqrt(abs(4 * c[k] - 3 * a[k]^2))) / 2
        v[k] = u[k] + a[k]
    end
    constrain!(method, points)
    total = zero(eltype(u))
    for k in eachindex(u)
        points.wplus[k] = inv(v[k] * (u[k] + v[k]))
        points.wminus[k] = (v[k] / u[k]) * points.wplus[k]
        total += points.wminus[k]
    end
    for k in eachindex(u)
        total += points.wplus[k]
    end
    points.w0 = 1 - total
    return points
end

# The lower Cholesky factor of `V`, with zeros above the diagonal.
function lower_factor!(L::AbstractMatrix, V::AbstractMatrix)
    uplo = cholesky_factor!(L, V)
    d = size(L, 1)
    for k in 1:d, r in (k + 1):d
        uplo == 'U' && (L[r, k] = L[k, r])
        L[k, r] = zero(eltype(L))
    end
    return L
end

## The constraints

constrain_scalar(::GenUnscented, m, σ, u, v, s) = (u, v)

function constrain_scalar(method::GenConstrainedUnscented, m, σ, u, v, s)
    lower, upper = only(__as_vec(lowertuple(method.lower))), only(__as_vec(lowertuple(method.upper)))
    lower <= m <= upper || throw(DomainError(m, "the mean must lie within the bounds [$lower, $upper]"))
    return constrain_pair(u, v, s, method.shrink * (m - lower) / σ, method.shrink * (upper - m) / σ)
end

constrain!(::GenUnscented, points::GenSigmaPointsMv) = points

function constrain!(method::GenConstrainedUnscented, points::GenSigmaPointsMv)
    m, L, lower, upper = points.m, points.L, points.lower, points.upper
    copy_bounds!(lower, lowertuple(method.lower))
    copy_bounds!(upper, lowertuple(method.upper))
    for k in eachindex(m)
        lower[k] <= m[k] <= upper[k] || throw(DomainError(m, "the mean must lie within the bounds"))
    end
    for k in eachindex(points.u)
        # How far the mean may move along `-Lₖ` and along `Lₖ` before a component reaches a bound.
        reach_minus, reach_plus = oftype(points.u[k], Inf), oftype(points.u[k], Inf)
        for r in k:length(m)
            l = L[r, k]
            if l > 0
                reach_minus = min(reach_minus, (m[r] - lower[r]) / l)
                reach_plus = min(reach_plus, (upper[r] - m[r]) / l)
            elseif l < 0
                reach_minus = min(reach_minus, (upper[r] - m[r]) / -l)
                reach_plus = min(reach_plus, (m[r] - lower[r]) / -l)
            end
        end
        points.u[k], points.v[k] = constrain_pair(points.u[k], points.v[k], points.a[k], method.shrink * reach_minus, method.shrink * reach_plus)
    end
    return points
end

# The bounds of several inputs, concatenated into `buffer`.
function copy_bounds!(buffer::AbstractVector, bounds::Tuple)
    starts = __starts_at(map(size, bounds))
    length(buffer) == last(starts) + length(last(bounds)) - 1 ||
        throw(DimensionMismatch(lazy"the bounds have $(last(starts) + length(last(bounds)) - 1) components, the input $(length(buffer))"))
    foreach(bounds, starts) do bound, start
        @inbounds __as_vec_copyto!(buffer, start, bound)
    end
    return buffer
end

# `u` and `v` within their reaches. A side that moves takes the other with it, `v - u = s`, so
# that the skewness stays matched, unless that would take the other side out of its own reach.
function constrain_pair(u, v, s, reach_minus, reach_plus)
    moved_minus, moved_plus = u > reach_minus, v > reach_plus
    u, v = min(u, reach_minus), min(v, reach_plus)
    if moved_minus && !moved_plus && u + s > 0
        v = min(u + s, reach_plus)
    elseif moved_plus && !moved_minus && v - s > 0
        u = min(v - s, reach_minus)
    end
    return u, v
end

## The points

npoints(::GenSigmaPoints) = 3
npoints(points::GenSigmaPointsMv) = 2 * length(points.m) + 1

point!(points::GenSigmaPoints, i) = i == 1 ? points.m : (i == 2 ? points.m - points.u * points.σ : points.m + points.v * points.σ)
weight(points::GenSigmaPoints, i) = i == 1 ? points.w0 : (i == 2 ? points.wminus : points.wplus)
center(points::GenSigmaPoints) = points.m

function point!(points::GenSigmaPointsMv, i)
    x, m, L, d = points.x, points.m, points.L, length(points.m)
    if i == 1
        copyto!(x, m)
    elseif i <= d + 1
        k = i - 1
        for r in eachindex(x)
            x[r] = m[r] - points.u[k] * L[r, k]
        end
    else
        k = i - d - 1
        for r in eachindex(x)
            x[r] = m[r] + points.v[k] * L[r, k]
        end
    end
    return x
end

function weight(points::GenSigmaPointsMv, i)
    d = length(points.m)
    return i == 1 ? points.w0 : (i <= d + 1 ? points.wminus[i - 1] : points.wplus[i - d - 1])
end

center(points::GenSigmaPointsMv) = points.m

## Several independent inputs with their higher moments

input_layout(m, V, S, K) = input_layout(m, V)

struct JointGenPoints{P, T}
    inner::P
    m::Vector{T}
    V::Matrix{T}
    S::Vector{T}
    K::Vector{T}
end

pointset(method::AnyGenUnscented, means::Tuple, covs::Tuple, skews::Tuple, kurts::Tuple) = joint_gen_pointset(method, means, covs, skews, kurts)
prepare!(points, method::AnyGenUnscented, means::Tuple, covs::Tuple, skews::Tuple, kurts::Tuple) = prepare_gen_inputs!(points, method, means, covs, skews, kurts)

joint_gen_pointset(method, means::Tuple{Any}, covs::Tuple{Any}, skews::Tuple{Any}, kurts::Tuple{Any}) =
    pointset(method, only(means), only(covs), only(skews), only(kurts))

function joint_gen_pointset(method, means::Tuple, covs::Tuple, skews::Tuple, kurts::Tuple)
    m, V, _ = joint_mean_cov(means, covs)
    T = eltype(V)
    S, K = convert(Vector{T}, __as_vec(skews)), convert(Vector{T}, __as_vec(kurts))
    return JointGenPoints(pointset(method, m, V, S, K), convert(Vector{T}, m), V, S, K)
end

prepare_gen_inputs!(points, method, means, covs, skews, kurts) = prepare!(points, method, only(means), only(covs), only(skews), only(kurts))

function prepare_gen_inputs!(points::JointGenPoints, method, means, covs, skews, kurts)
    joint_mean_cov!(points.m, points.V, means, covs)
    starts = __starts_at(map(size, means))
    foreach(skews, kurts, starts) do skew, kurt, start
        @inbounds __as_vec_copyto!(points.S, start, skew)
        @inbounds __as_vec_copyto!(points.K, start, kurt)
    end
    prepare!(points.inner, method, points.m, points.V, points.S, points.K)
    return points
end

npoints(points::JointGenPoints) = npoints(points.inner)
point!(points::JointGenPoints, i) = point!(points.inner, i)
weight(points::JointGenPoints, i) = weight(points.inner, i)
center(points::JointGenPoints) = center(points.inner)

"""
    sigma_points_weights(method::GenUnscented, m, V[, S, K]) -> (points, weights)

The `2d + 1` sigma points of the generalized unscented transform for the mean `m`, covariance
`V` and, optionally, the components' third and fourth central moments `S` and `K` (a normal's
without), and their weights, which serve both the mean and the covariance: tuples for a scalar
input, vectors otherwise. The first point is the mean; then come the `d` points below it and the
`d` above. The same for [`GenConstrainedUnscented`](@ref).
"""
function sigma_points_weights(method::AnyGenUnscented, params::Vararg{Any, N}) where {N}
    points = prepare!(pointset(method, params...), method, params...)
    return materialise_points(points)
end

materialise_points(points::GenSigmaPoints) = ((point!(points, 1), point!(points, 2), point!(points, 3)), (weight(points, 1), weight(points, 2), weight(points, 3)))
materialise_points(points) = ([copy(point!(points, i)) for i in 1:npoints(points)], [weight(points, i) for i in 1:npoints(points)])
