"""
    GaussLaguerreQuadrature(p::Int)

Gauss–Laguerre quadrature with `p` points, for expectations under a gamma distribution given by
its shape `α` and scale `θ`: `params = (shape, scale)`, an exponential with rate `λ` being
`(1, 1 / λ)`. The points are the nodes of the generalised Laguerre rule for the weight
`t^(α - 1) e^(-t)`, mapped to `x = θ t`, so the rule is exact for polynomials of degree up to
`2p - 1` under the gamma. It is univariate.

The nodes depend on the shape: they are computed on every call, by Newton's method in place, so
the in-place operations allocate nothing for any shape. When Newton's method does not converge,
FastGaussQuadrature computes them, allocating.

It is used through [`expectation`](@ref), [`approximate`](@ref), [`forward_statistics`](@ref)
and [`approximate_meancov`](@ref). [`glquadrature`](@ref)`(p)` builds the same.

# Examples

```jldoctest; setup = :(using MessagePassingRulesApproximations)
julia> gl = GaussLaguerreQuadrature(5);

julia> expectation(gl, x -> x, 2.0, 3.0) ≈ 6.0   # the mean of a gamma, αθ
true

julia> expectation(gl, x -> x^2, 2.0, 3.0) ≈ 2.0 * 3.0 * 3.0^2   # E[x²] = α(α + 1)θ²
true
```
"""
struct GaussLaguerreQuadrature <: AbstractApproximationMethod
    p::Int

    function GaussLaguerreQuadrature(p::Int)
        p > 0 || throw(ArgumentError(lazy"Gauss–Laguerre quadrature needs at least one point, not $p"))
        return new(p)
    end
end

"""
    glquadrature(p::Int) -> GaussLaguerreQuadrature

A [`GaussLaguerreQuadrature`](@ref) with `p` points.
"""
glquadrature(p::Int) = GaussLaguerreQuadrature(p)

Base.show(io::IO, method::GaussLaguerreQuadrature) = print(io, "GaussLaguerreQuadrature(", method.p, ")")

approximation_name(gl::GaussLaguerreQuadrature) = "GaussLaguerre($(gl.p))"
approximation_short_name(gl::GaussLaguerreQuadrature) = "GL$(gl.p)"

## The point set

# The nodes `tᵢ` of the rule for the weight `t^(α - 1) e^(-t)`, the logarithms of the weights,
# normalised to sum to one, and the weights; the points are `θ tᵢ`.
mutable struct LaguerrePoints{T}
    nodes::Vector{T}
    logweights::Vector{T}
    weights::Vector{T}
    shape::T
    scale::T
end

function pointset(gl::GaussLaguerreQuadrature, shape::Real, scale::Real)
    T = float(promote_type(typeof(shape), typeof(scale)))
    return LaguerrePoints(Vector{T}(undef, gl.p), Vector{T}(undef, gl.p), Vector{T}(undef, gl.p), T(shape), T(scale))
end

function prepare!(points::LaguerrePoints, gl::GaussLaguerreQuadrature, shape::Real, scale::Real)
    length(points.nodes) == gl.p || throw(DimensionMismatch(lazy"the point set is for $(length(points.nodes)) points, not $(gl.p)"))
    shape > 0 || throw(DomainError(shape, "the shape of a gamma must be positive"))
    scale > 0 || throw(DomainError(scale, "the scale of a gamma must be positive"))
    points.shape = shape
    points.scale = scale
    laguerre_nodes!(points.nodes, points.logweights, shape - 1) || laguerre_nodes_fallback!(points.nodes, points.logweights, shape - 1)
    normalise_logweights!(points.logweights)
    map!(exp, points.weights, points.logweights)
    return points
end

npoints(points::LaguerrePoints) = length(points.nodes)
point!(points::LaguerrePoints, i) = points.scale * points.nodes[i]
weight(points::LaguerrePoints, i) = points.weights[i]
logweight(points::LaguerrePoints, i) = points.logweights[i]
weightsign(::LaguerrePoints, i) = 1
center(points::LaguerrePoints) = points.shape * points.scale


# The nodes of the generalised Gauss–Laguerre rule with parameter `a > -1`, and the logarithms of
# its weights up to a constant, by Newton's method on the three-term recurrence of the Laguerre
# polynomials from the asymptotic initial guesses of Numerical Recipes (`gaulag`). The weight of
# a node `z` is proportional to `-1 / (L'ₙ(z) Lₙ₋₁(z))`. Returns `false` when an iteration does not
# converge or the nodes are not increasing, leaving the fallback to FastGaussQuadrature.
function laguerre_nodes!(nodes::AbstractVector{T}, logweights::AbstractVector{T}, a::Real) where {T}
    n = length(nodes)
    z = zero(T)
    for i in 1:n
        if i == 1
            z = (1 + a) * (3 + 0.92 * a) / (1 + 2.4 * n + 1.8 * a)
        elseif i == 2
            z += (15 + 6.25 * a) / (1 + 0.9 * a + 2.5 * n)
        else
            k = i - 2
            z += ((1 + 2.55 * k) / (1.9 * k) + 1.26 * k * a / (1 + 3.5 * k)) * (z - nodes[i - 2]) / (1 + 0.3 * a)
        end
        converged = false
        derivative, previous = one(T), zero(T)
        for _ in 1:100
            p1, p2 = one(T), zero(T)
            for j in 1:n
                p3 = p2
                p2 = p1
                p1 = ((2j - 1 + a - z) * p2 - (j - 1 + a) * p3) / j
            end
            derivative, previous = (n * p1 - (n + a) * p2) / z, p2
            step = p1 / derivative
            z -= step
            if abs(step) <= 8 * eps(T) * abs(z)
                converged = true
                break
            end
        end
        (converged && isfinite(z) && z > (i == 1 ? 0 : nodes[i - 1])) || return false
        nodes[i] = z
        logweights[i] = -log(abs(derivative * previous))
    end
    return true
end

function laguerre_nodes_fallback!(nodes::AbstractVector, logweights::AbstractVector, a::Real)
    t, w = FastGaussQuadrature.gausslaguerre(length(nodes), a)
    copyto!(nodes, t)
    map!(log, logweights, w)
    return true
end

function normalise_logweights!(logweights::AbstractVector)
    largest = maximum(logweights)
    total = zero(largest)
    for lw in logweights
        total += exp(lw - largest)
    end
    logweights .-= largest + log(total)
    return logweights
end
