"""
    Linearization()

The local-linearization method: a function is replaced by its first-order Taylor expansion at
the inputs' means, computed by automatic differentiation (ForwardDiff), and moments pass
through the linear map exactly. It has no parameters.

It is exact for affine functions and accurate when the function is close to linear over the
inputs' spread; the function must be differentiable, and written generically enough for
ForwardDiff's dual numbers. [`Unscented`](@ref) is the alternative that needs no derivatives.

See [`approximate`](@ref)`(::Linearization, g, x̂)` for the map it computes.
"""
struct Linearization <: AbstractApproximationMethod end

approximation_name(::Linearization) = "Linearization"
approximation_short_name(::Linearization) = "LN"

"""
    approximate(::Linearization, g, x̂::Tuple) -> (A, b)

The local linear map of `g` at the expansion point `x̂`, so that `g(x) ≈ A * x + b` near `x̂`.

# Arguments

- `g`: a function of one or several arguments, each a number or a vector, returning a number or
  a vector;
- `x̂`: the expansion point, a tuple with one entry per argument of `g`, typically the inputs'
  means.

# Returns

`(A, b)`, where `x` stands for the arguments concatenated into one vector. `A` is a derivative
for a scalar function of one scalar, a gradient row for a scalar function of a vector, and a
Jacobian otherwise; `b = g(x̂) - A * x̂`. Propagating a normal `N(m, V)` of `x` gives
`N(A * m + b, A * V * A')`.

The in-place form, `approximate!(…, ws, Linearization(), g, x̂)` with
`ws = `[`workspace`](@ref)`(Linearization(), g, x̂)`, keeps ForwardDiff's configuration in the
workspace and allocates nothing of its own. It takes the array-valued parts of the result first:
`approximate!(ws, …)` for a scalar `g` of a scalar, `approximate!(a, ws, …)` for a scalar `g` of a
vector, writing the gradient into `a` and returning `(a', b)`, and `approximate!(A, b, ws, …)`
for a vector `g`.

# Examples

```jldoctest; setup = :(using MessagePassingRulesApproximations)
julia> approximate(Linearization(), (x, y) -> x .- y, ([1.0, 2.0], [0.5, 0.5]))
([1.0 0.0 -1.0 0.0; 0.0 1.0 0.0 -1.0], [0.0, 0.0])
```
"""
approximate(method::Linearization, g::G, x̂::Tuple) where {G} = local_linearization(g, x̂)

"""
    local_linearization(g, x̂::Tuple) -> (A, b)

The first-order expansion of `g` at `x̂`, the function behind
[`approximate`](@ref)`(::Linearization, g, x̂)`, which documents its arguments and result.
"""
function local_linearization(g::G, x̂::Tuple) where {G}
    # `g` at `x̂` sizes the workspace, and is the value the expansion needs.
    r = g(x̂...)
    ws = linearization_workspace(r, g, x̂, nothing, nothing)
    return linearize_into(ws, g, x̂, r)
end

# Several inputs are differentiated as one vector, split back into `g`'s arguments.
struct SplitArguments{G, S}
    g::G
    sizes::S
end

(split::SplitArguments)(x) = split.g(__splitjoin(x, split.sizes)...)

struct LinearizationWorkspace{F, X, A, B, C, M, S} <: AbstractWorkspace
    # The function ForwardDiff differentiates: `g` for one input, `SplitArguments` for several.
    f::F
    # The expansion point as one vector, for several inputs; `nothing` otherwise.
    x::X
    # The derivative, gradient or Jacobian, and the offset; `nothing` where they are scalars.
    A::A
    b::B
    cfg::C
    # For `forward_statistics`: the inputs' joint mean and covariance, for several inputs.
    μ::M
    Σ::S
end

Base.show(io::IO, ::LinearizationWorkspace) = print(io, "LinearizationWorkspace()")

workspace(method::Linearization, g::G, x̂::Tuple) where {G} = linearization_workspace(g(x̂...), g, x̂, nothing, nothing)

function workspace(method::Linearization, g::G, means::Tuple, covs::Tuple) where {G}
    μ, Σ = joint_buffers(means, covs)
    return linearization_workspace(g(means...), g, means, μ, Σ)
end

joint_buffers(::Tuple{Union{Real, AbstractVector}}, ::Tuple{Union{Real, AbstractMatrix}}) = (nothing, nothing)
function joint_buffers(means::Tuple, covs::Tuple)
    m, V, _ = joint_mean_cov(means, covs)
    return m, V
end

# One scalar input: a derivative, a number or a vector.
linearization_workspace(r::Real, g, x̂::Tuple{Real}, μ, Σ) = LinearizationWorkspace(g, nothing, nothing, nothing, nothing, μ, Σ)
function linearization_workspace(r::AbstractVector, g, x̂::Tuple{Real}, μ, Σ)
    T = promote_type(eltype(r), typeof(first(x̂)))
    return LinearizationWorkspace(g, nothing, similar(r, T), similar(r, T), nothing, μ, Σ)
end

# One vector input: a gradient or a Jacobian of `g` itself.
function linearization_workspace(r::Real, g, x̂::Tuple{AbstractVector}, μ, Σ)
    x = first(x̂)
    return LinearizationWorkspace(g, nothing, similar(x, promote_type(eltype(x), typeof(r))), nothing, ForwardDiff.GradientConfig(g, x), μ, Σ)
end

function linearization_workspace(r::AbstractVector, g, x̂::Tuple{AbstractVector}, μ, Σ)
    x = first(x̂)
    T = promote_type(eltype(x), eltype(r))
    return LinearizationWorkspace(g, nothing, Matrix{T}(undef, length(r), length(x)), similar(r, T), ForwardDiff.JacobianConfig(g, x), μ, Σ)
end

# Several inputs: of their concatenation.
function linearization_workspace(r::Real, g, x̂::Tuple, μ, Σ)
    x = __as_vec(x̂)
    split = SplitArguments(g, map(size, x̂))
    return LinearizationWorkspace(split, x, similar(x), nothing, ForwardDiff.GradientConfig(split, x), μ, Σ)
end

function linearization_workspace(r::AbstractVector, g, x̂::Tuple, μ, Σ)
    x = __as_vec(x̂)
    split = SplitArguments(g, map(size, x̂))
    T = promote_type(eltype(x), eltype(r))
    return LinearizationWorkspace(split, x, Matrix{T}(undef, length(r), length(x)), similar(r, T), ForwardDiff.JacobianConfig(split, x), μ, Σ)
end

# The allocating form: the workspace's buffers are fresh, so they are the result.
linearize_into(ws::LinearizationWorkspace{F, Nothing, Nothing}, g, x̂, r) where {F} = linearize!(ws, g, x̂, r)
linearize_into(ws::LinearizationWorkspace{F, X, <:AbstractVector, Nothing}, g, x̂, r) where {F, X} = linearize!(ws.A, ws, g, x̂, r)
linearize_into(ws::LinearizationWorkspace, g, x̂, r) = linearize!(ws.A, ws.b, ws, g, x̂, r)

# `g` at the expansion point: given, or computed.
value_of(g::G, x̂, ::Nothing) where {G} = g(x̂...)
value_of(g::G, x̂, r) where {G} = r

approximate!(ws::LinearizationWorkspace, ::Linearization, g::G, x̂::Tuple) where {G} = linearize!(ws, g, x̂, nothing)
approximate!(A::AbstractVector, ws::LinearizationWorkspace, ::Linearization, g::G, x̂::Tuple) where {G} = linearize!(A, ws, g, x̂, nothing)
approximate!(A::AbstractVecOrMat, b::AbstractVector, ws::LinearizationWorkspace, ::Linearization, g::G, x̂::Tuple) where {G} = linearize!(A, b, ws, g, x̂, nothing)

# The differentiated function's argument: the input itself, or the inputs concatenated.
differentiation_point(ws::LinearizationWorkspace{F, Nothing}, x̂) where {F} = first(x̂)
function differentiation_point(ws::LinearizationWorkspace, x̂)
    starts = __starts_at(map(size, x̂))
    foreach(x̂, starts) do input, start
        @inbounds __as_vec_copyto!(ws.x, start, input)
    end
    return ws.x
end

function linearize!(ws::LinearizationWorkspace{F, Nothing, Nothing}, g::G, x̂::Tuple{Real}, r) where {F, G}
    x = first(x̂)
    result = value_of(g, x̂, r)
    A = ForwardDiff.derivative(g, x)
    return (A, result - A * x)
end

function linearize!(A::AbstractVector, ws::LinearizationWorkspace{F, X, <:AbstractVector, Nothing}, g::G, x̂::Tuple, r) where {F, X, G}
    result = value_of(g, x̂, r)
    x = differentiation_point(ws, x̂)
    ForwardDiff.gradient!(A, ws.f, x, ws.cfg)
    return (A', result - A' * x)
end

function linearize!(A::AbstractVector, b::AbstractVector, ws::LinearizationWorkspace{F, Nothing, <:AbstractVector, <:AbstractVector, Nothing}, g::G, x̂::Tuple{Real}, r) where {F, G}
    x = first(x̂)
    result = value_of(g, x̂, r)
    ForwardDiff.derivative!(A, g, x)
    for j in eachindex(b)
        b[j] = result[j] - A[j] * x
    end
    return (A, b)
end

function linearize!(A::AbstractMatrix, b::AbstractVector, ws::LinearizationWorkspace, g::G, x̂::Tuple, r) where {G}
    result = value_of(g, x̂, r)
    x = differentiation_point(ws, x̂)
    ForwardDiff.jacobian!(A, ws.f, x, ws.cfg)
    for j in eachindex(b)
        acc = zero(eltype(b))
        for k in eachindex(x)
            acc += A[j, k] * x[k]
        end
        b[j] = result[j] - acc
    end
    return (A, b)
end

## Forward statistics

"""
    approximate(::Linearization, f, means::Tuple, covs::Tuple) -> (m, V)
    forward_statistics(::Linearization, f, means::Tuple, covs::Tuple) -> (m, V, C)

The inputs' joint normal pushed through the linear map `x ↦ A x + b` of `f` at the inputs'
means: `m = A x̄ + b`, `V = A Σ A'` and the cross-covariance `C = Σ A'`, for inputs with means
`means` and covariances `covs`, jointly `N(x̄, Σ)` with a block-diagonal `Σ`. The in-place
[`forward_statistics!`](@ref) takes a [`workspace`](@ref)`(Linearization(), f, means, covs)` and
the array-valued outputs first, as for the other methods.
"""
function forward_statistics(method::Linearization, f::F, means::Tuple, covs::Tuple) where {F}
    r = f(means...)
    μ, Σ = joint_buffers(means, covs)
    ws = linearization_workspace(r, f, means, μ, Σ)
    return linear_statistics_into(ws, f, means, covs, r)
end

approximate(method::Linearization, f::F, means::Tuple, covs::Tuple) where {F} = Base.front(forward_statistics(method, f, means, covs))

joint_moments(ws::LinearizationWorkspace{F, X, A, B, C, Nothing}, means, covs) where {F, X, A, B, C} = (first(means), first(covs))
joint_moments(ws::LinearizationWorkspace, means, covs) = joint_mean_cov!(ws.μ, ws.Σ, means, covs)

# The outputs in the type the map and the covariances promote to, as `A * Σ * A'` gives.
linear_statistics_into(ws::LinearizationWorkspace{F, Nothing, Nothing}, f, means, covs, r) where {F} = linear_statistics!(ws, f, means, covs, r)
linear_statistics_into(ws::LinearizationWorkspace{F, X, <:AbstractVector, Nothing}, f, means, covs, r) where {F, X} =
    linear_statistics!(similar(ws.A, promote_type(eltype(ws.A), map(eltype, covs)...)), ws, f, means, covs, r)
function linear_statistics_into(ws::LinearizationWorkspace{F, X, A}, f, means, covs, r) where {F, X, A}
    T = promote_type(eltype(ws.A), eltype(ws.b), map(eltype, covs)...)
    d_out = length(ws.b)
    d_in = A <: AbstractMatrix ? size(ws.A, 2) : 1
    return linear_statistics!(similar(ws.b, T), similar(ws.b, T, d_out, d_out), similar(ws.b, T, d_in, d_out), ws, f, means, covs, r)
end

forward_statistics!(ws::LinearizationWorkspace, ::Linearization, f::G, means::Tuple, covs::Tuple) where {G} = linear_statistics!(ws, f, means, covs, nothing)
forward_statistics!(C::AbstractVector, ws::LinearizationWorkspace, ::Linearization, f::G, means::Tuple, covs::Tuple) where {G} = linear_statistics!(C, ws, f, means, covs, nothing)
forward_statistics!(m::AbstractVector, V::AbstractMatrix, C::AbstractMatrix, ws::LinearizationWorkspace, ::Linearization, f::G, means::Tuple, covs::Tuple) where {G} =
    linear_statistics!(m, V, C, ws, f, means, covs, nothing)

# A scalar function of a scalar.
function linear_statistics!(ws::LinearizationWorkspace{F, Nothing, Nothing}, f::G, means::Tuple, covs::Tuple, r) where {F, G}
    A, b = linearize!(ws, f, means, r)
    μ, Σ = joint_moments(ws, means, covs)
    return (A * μ + b, A * Σ * A, Σ * A)
end

# A scalar function of a vector: `C = Σ a`, a vector.
function linear_statistics!(C::AbstractVector, ws::LinearizationWorkspace{F, X, <:AbstractVector, Nothing}, f::G, means::Tuple, covs::Tuple, r) where {F, X, G}
    a, b = linearize!(ws.A, ws, f, means, r)
    μ, Σ = joint_moments(ws, means, covs)
    mul!(C, Σ, ws.A)
    return (a * μ + b, a * C, C)
end

# A vector function: `m = A x̄ + b`, `C = Σ A'` and `V = A C`.
function linear_statistics!(m::AbstractVector, V::AbstractMatrix, C::AbstractMatrix, ws::LinearizationWorkspace, f::G, means::Tuple, covs::Tuple, r) where {G}
    A, b = linearize!(ws.A, ws.b, ws, f, means, r)
    μ, Σ = joint_moments(ws, means, covs)
    linear_moments!(m, V, C, A, b, μ, Σ)
    return (m, V, C)
end

# A scalar input: `A` is the derivative, a vector, and `C` a row.
function linear_moments!(m, V, C, A::AbstractVector, b, μ::Real, Σ::Real)
    for j in eachindex(m)
        m[j] = A[j] * μ + b[j]
        C[1, j] = Σ * A[j]
    end
    for k in eachindex(A), j in eachindex(A)
        V[j, k] = A[j] * Σ * A[k]
    end
    return m, V, C
end

function linear_moments!(m, V, C, A::AbstractMatrix, b, μ::AbstractVector, Σ::AbstractMatrix)
    mul!(m, A, μ)
    m .+= b
    mul!(C, Σ, A')
    mul!(V, A, C)
    return m, V, C
end
