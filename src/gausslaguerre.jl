"""
    GaussLaguerreQuadrature(p::Int)

Gauss–Laguerre quadrature with `p` points, for expectations under a gamma distribution given by
its shape `α` and scale `θ`: `params = (shape, scale)`, an exponential with rate `λ` being
`(1, 1 / λ)`. The points are the nodes of the generalised Laguerre rule for the weight
`t^(α - 1) e^(-t)`, mapped to `x = θ t`, so the rule is exact for polynomials of degree up to
`2p - 1` under the gamma. It is univariate.

The nodes depend on the shape: they are computed on every call, in place, as the eigenvalues of
the Laguerre polynomials' Jacobi matrix polished by Newton's method, so the in-place operations
allocate nothing for any shape. In the unlikely case that the eigenvalue iteration does not
converge, FastGaussQuadrature computes them, allocating.

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
    laguerre_nodes!(points.nodes, points.logweights, points.weights, shape - 1) || laguerre_nodes_fallback!(points.nodes, points.logweights, shape - 1)
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
# its weights up to a constant, in place, with `work` as scratch. Golub–Welsch: the nodes are the
# eigenvalues of the Jacobi matrix of the Laguerre polynomials, diagonal `2i - 1 + a` and
# off-diagonal `√(i (i + a))`, and the weights the squares of the eigenvectors' first components,
# by the implicit QL method tracking those components only. Newton's method on the three-term
# recurrence then polishes each node, and gives its weight, `∝ -1 / (L'ₙ(z) Lₙ₋₁(z))`, to full
# relative accuracy however small. Returns `false` when the QL iteration does not converge.
function laguerre_nodes!(nodes::AbstractVector{T}, logweights::AbstractVector{T}, work::AbstractVector{T}, a::Real) where {T}
    n = length(nodes)
    d, e, z = nodes, work, logweights
    for i in 1:n
        d[i] = 2i - 1 + a
        e[i] = i < n ? sqrt(i * (i + a)) : zero(T)
        z[i] = i == 1 ? one(T) : zero(T)
    end
    symmetric_tridiagonal_ql!(d, e, z) || return false
    # Ascending, with their first components.
    for i in 2:n
        di, zi, j = d[i], z[i], i - 1
        while j >= 1 && d[j] > di
            d[j + 1], z[j + 1] = d[j], z[j]
            j -= 1
        end
        d[j + 1], z[j + 1] = di, zi
    end
    # Newton's polish of every node, and its weight; when one fails, the eigenvector weights.
    polished = true
    for i in 1:n
        node, logw, converged = laguerre_newton(d[i], n, a)
        if converged && abs(node - d[i]) <= sqrt(eps(T)) * max(abs(d[i]), one(T))
            work[i] = logw
            d[i] = node
        else
            polished = false
        end
    end
    for i in 1:n
        logweights[i] = polished ? work[i] : 2 * log(abs(z[i]))
    end
    return true
end

# Newton's method for a root of the Laguerre polynomial `Lₙ⁽ᵃ⁾` from `z`, and the logarithm of its
# weight up to a constant.
function laguerre_newton(z::T, n, a) where {T}
    derivative, previous = one(T), zero(T)
    for _ in 1:20
        p1, p2 = one(T), zero(T)
        for j in 1:n
            p3 = p2
            p2 = p1
            p1 = ((2j - 1 + a - z) * p2 - (j - 1 + a) * p3) / j
        end
        derivative, previous = (n * p1 - (n + a) * p2) / z, p2
        step = p1 / derivative
        z -= step
        abs(step) <= 4 * eps(T) * abs(z) && return z, -log(abs(derivative * previous)), true
    end
    return z, zero(T), false
end

# The eigenvalues of the symmetric tridiagonal matrix with diagonal `d` and off-diagonal `e[1:n-1]`,
# into `d`, and the first components of its eigenvectors into `z`, which starts as the first unit
# vector: the implicit QL method with Wilkinson shifts (`tqli`).
function symmetric_tridiagonal_ql!(d::AbstractVector{T}, e::AbstractVector{T}, z::AbstractVector{T}) where {T}
    n = length(d)
    for l in 1:n
        iterations = 0
        while true
            m = l
            while m < n
                abs(e[m]) <= eps(T) * (abs(d[m]) + abs(d[m + 1])) && break
                m += 1
            end
            m == l && break
            (iterations += 1) > 60 && return false
            g = (d[l + 1] - d[l]) / (2 * e[l])
            r = hypot(g, one(T))
            g = d[m] - d[l] + e[l] / (g + copysign(r, g))
            s, c, p = one(T), one(T), zero(T)
            underflow = false
            for i in (m - 1):-1:l
                f, b = s * e[i], c * e[i]
                r = hypot(f, g)
                e[i + 1] = r
                if iszero(r)
                    d[i + 1] -= p
                    e[m] = zero(T)
                    underflow = true
                    break
                end
                s, c = f / r, g / r
                g = d[i + 1] - p
                r = (d[i] - g) * s + 2 * c * b
                p = s * r
                d[i + 1] = g + p
                g = c * r - b
                f = z[i + 1]
                z[i + 1] = s * z[i] + c * f
                z[i] = c * z[i] - s * f
            end
            underflow && continue
            d[l] -= p
            e[l] = g
            e[m] = zero(T)
        end
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
