# Point sets, workspaces, and the operations every point-set method shares.
#
# A method places weighted points for a measure given by plain parameters: `(m, V)` for a
# normal, `(shape, scale)` for a gamma, tuples of means and covariances for several independent
# inputs. An expectation is then `Σᵢ wᵢ f(xᵢ)`, and every operation below — the expectation, the
# moments of `f(x)`, the cross-covariance, the moments of a reweighted measure — is that sum,
# written once over the point-set interface:
#
#     pointset(method, params...)          the point set's buffers, allocated once
#     prepare!(points, method, params...)  places the points for these parameters
#     npoints(points), point!(points, i), weight(points, i), covweight(points, i),
#     logweight(points, i), center(points)
#
# Every operation `op(method, f, params...)` has an in-place `op!(out..., ws, method, f,
# params...)` that allocates nothing of its own, where `ws = workspace(method, f, params...)`
# holds every buffer. The leading `out...` are the array-valued parts of the result, in order;
# the scalar parts are returned. The allocating `op` builds the outputs and the workspace and
# calls `op!`.

"""
    AbstractWorkspace

The supertype of the workspaces [`workspace`](@ref) builds: the memory an in-place operation,
such as [`approximate!`](@ref) or [`expectation!`](@ref), works in. A workspace is written before
it is read on every call, so one workspace serves any number of calls, with any parameters of
the shapes it was built for; it never holds a result.
"""
abstract type AbstractWorkspace end

"""
    workspace(method, f, params...) -> AbstractWorkspace

The memory the in-place operations of `method` need for the function `f` and parameters of the
shapes of `params`: its points, factors and the values of `f` at the points. It takes the
arguments of the operation it serves, and calls `f` once, at the first point, to learn the size
and type of its output.

Build it once and pass it to every call of an in-place operation, such as
[`approximate!`](@ref), [`forward_statistics!`](@ref), [`expectation!`](@ref) or
[`approximate_meancov!`](@ref): those calls then allocate nothing of their own. Parameters of
other values but the same shapes and types may be passed to each call. It can serve as a rule's
scratch: it is written before it is read on every call, and no result points into it.

# Examples

```jldoctest; setup = :(using MessagePassingRulesApproximations)
julia> import MessagePassingRulesApproximations: workspace

julia> ws = workspace(Unscented(), sin, (0.0,), (1.0,));

julia> m, V = approximate!(ws, Unscented(), sin, (0.5,), (0.1,));

julia> m ≈ first(approximate(Unscented(), sin, (0.5,), (0.1,)))
true
```
"""
function workspace end

"""
    pointset(method, params...)

The point set of `method` for parameters of the shapes of `params`, with its buffers allocated
and its points not yet placed: [`prepare!`](@ref) places them. A method that implements it,
[`prepare!`](@ref), [`npoints`](@ref), [`point!`](@ref), [`weight`](@ref) and [`center`](@ref)
gets every generic operation, [`expectation`](@ref), [`approximate`](@ref),
[`forward_statistics`](@ref) and [`approximate_meancov`](@ref), with their in-place variants.
"""
function pointset end

"""
    prepare!(points, method, params...) -> points

Places the points of `points` for the parameters `params` of `method`'s measure: factorises the
covariance, computes the weights. It writes every buffer it reads, so a point set prepared for
other parameters, or never prepared, gives the same result. It returns the prepared point set:
`points` itself when it is mutable, a new value when it is an immutable one, as a scalar point
set of a few numbers is, so that it lives on the stack. Callers use the result.
"""
function prepare! end

"""
    npoints(points) -> Int

The number of points in the point set `points`.
"""
function npoints end

"""
    point!(points, i)

The `i`-th point of the point set `points`: a number, or a vector that is one buffer of
`points`, rewritten by the next call. Use it before asking for the next point.
"""
function point! end

"""
    weight(points, i)

The weight of the `i`-th point of `points` in an expectation: the weights sum to one, and
`E[f(x)] ≈ Σᵢ weight(points, i) f(point!(points, i))`. Some may be negative, as the centre's is
in the unscented transform.
"""
function weight end

"""
    covweight(points, i)

The weight of the `i`-th point of `points` in a covariance. It is [`weight`](@ref) for every
method but the unscented transform, whose centre weighs differently in a covariance.
"""
covweight(points, i) = weight(points, i)

"""
    logweight(points, i)

The logarithm of the absolute value of the weight of the `i`-th point of `points`, for an
expectation computed in log space, `LogIntegrand`: `log(abs(weight(points, i)))` unless the
method has it more accurately, as Gauss–Hermite does for weights that underflow. The sign is
[`weightsign`](@ref)'s.
"""
logweight(points, i) = log(abs(weight(points, i)))

"""
    weightsign(points, i)

The sign of the weight of the `i`-th point of `points`, `1` or `-1`, for an expectation computed
in log space. The weights of a quadrature are positive; the unscented transform's centre may weigh
negatively.
"""
weightsign(points, i) = sign(weight(points, i))

"""
    center(points)

The centre of the point set `points`, the mean of its measure: a number or a vector, one buffer
of `points`. The cross-covariance of [`forward_statistics`](@ref) is taken around it.
"""
function center end

## How a point is passed to `f`

# One input: `f(x)`. Several: `f(x₁, x₂, …)`, the point split into the inputs' shapes.
struct SingleInput end

struct SplitInputs{S}
    sizes::S
end

evaluate(f::F, ::SingleInput, x) where {F} = f(x)
evaluate(f::F, layout::SplitInputs, x) where {F} = f(__splitjoin(x, layout.sizes)...)

input_layout(m::Real, V::Real) = SingleInput()
input_layout(m::AbstractVector, V::AbstractMatrix) = SingleInput()
input_layout(::Tuple{Real}, ::Tuple{Real}) = SingleInput()
input_layout(::Tuple{AbstractVector}, ::Tuple{AbstractMatrix}) = SingleInput()
input_layout(means::Tuple, covs::Tuple) = SplitInputs(map(size, means))

## The workspace of a point-set method

struct PointSetWorkspace{P, L, Y, DX, DY} <: AbstractWorkspace
    points::P
    layout::L
    # The values of `f` at the points: a vector for a scalar `f`, a matrix with one column per
    # point otherwise. A scalar `f` that serves as a weight keeps its weighted values here.
    ys::Y
    # A point minus the centre, or `nothing` for scalar points.
    dx::DX
    # A value minus the output's mean, or `nothing` for a scalar `f`.
    dy::DY
end

Base.show(io::IO, ws::PointSetWorkspace) = print(io, "PointSetWorkspace(", npoints(ws.points), " points)")

workspace(method::AbstractApproximationMethod, f::F, params::Vararg{Any, N}) where {F, N} = first(prepared_workspace(Val(false), method, f, params...))

# The workspace, prepared for `params`, and `f` at its first point. The allocating operations
# start from both, so that they evaluate `f` once per point, as the in-place ones do. With
# `Val(true)`, a point set of a fixed, small number of points keeps the values of a scalar `f` in
# a tuple, so that an allocating operation on scalars allocates as little as the arithmetic does.
function prepared_workspace(tuples::Val, method::AbstractApproximationMethod, f::F, params::Vararg{Any, N}) where {F, N}
    points = held(tuples, prepared_pointset(method, params...))
    layout = input_layout(params...)
    first_value = evaluate(primal(f), layout, point!(points, 1))
    ys, dy = value_buffers(tuples, fixed_npoints(points), first_value, typeof(weight(points, 1)), npoints(points))
    return PointSetWorkspace(points, layout, ys, point_buffer(center(points)), dy), first_value
end

# An immutable point set, in a workspace the in-place operations reuse, lives in a mutable holder
# that `prepare!` replaces it in. The allocating operations, which prepare it once, keep it bare.
mutable struct HeldPoints{P}
    points::P
end

held(::Val{true}, points) = points
held(::Val{false}, points) = is_value(points) ? HeldPoints(points) : points

"""
    is_value(points) -> Bool

Whether the point set `points` is a plain, immutable value, which [`prepare!`](@ref) returns
anew rather than writes into, as the scalar point sets of the unscented transform and of
Gauss–Hermite cubature are: they then live on the stack in the allocating operations, and a
workspace holds them in a mutable holder. `false` by default; a method whose point set is
immutable returns `true`, and its [`pointset`](@ref) returns the point set already prepared for
the parameters it is given.
"""
is_value(points) = false

# The point set, prepared for `params`: a value's `pointset` already is, as it is built from them.
function prepared_pointset(method, params::Vararg{Any, N}) where {N}
    points = pointset(method, params...)
    return is_value(points) ? points : prepare!(points, method, params...)
end

npoints(held::HeldPoints) = npoints(held.points)
point!(held::HeldPoints, i) = point!(held.points, i)
weight(held::HeldPoints, i) = weight(held.points, i)
covweight(held::HeldPoints, i) = covweight(held.points, i)
logweight(held::HeldPoints, i) = logweight(held.points, i)
weightsign(held::HeldPoints, i) = weightsign(held.points, i)
center(held::HeldPoints) = center(held.points)
fixed_npoints(held::HeldPoints) = fixed_npoints(held.points)

# `Val(n)` for a point set of `n` points whatever its parameters, such as the unscented
# transform's scalar sigma points; `nothing` otherwise.
fixed_npoints(points) = nothing

value_buffers(::Val{true}, ::Val{K}, y::Real, ::Type{W}, n) where {K, W} = (ntuple(_ -> zero(promote_type(typeof(y), W)), Val(K)), nothing)
value_buffers(tuples, fixed, y, ::Type{W}, n) where {W} = output_buffers(y, W, n)

# The function whose values the operations combine: a `LogIntegrand`'s logarithm.
primal(f) = f

output_buffers(y::Real, ::Type{W}, n) where {W} = (Vector{promote_type(typeof(y), W)}(undef, n), nothing)
function output_buffers(y::AbstractVector, ::Type{W}, n) where {W}
    T = promote_type(eltype(y), W)
    return (Matrix{T}(undef, length(y), n), Vector{T}(undef, length(y)))
end
output_buffers(y, ::Type, n) = throw(ArgumentError(lazy"the function must return a number or a vector, not a `$(typeof(y))`"))

point_buffer(::Real) = nothing
point_buffer(x::AbstractVector) = similar(x)

prepare_workspace!(ws::PointSetWorkspace, method, params::Vararg{Any, N}) where {N} = (prepare_points!(ws.points, method, params...); ws)

function prepare_points!(points, method, params::Vararg{Any, N}) where {N}
    prepared = prepare!(points, method, params...)
    prepared === points || throw(ArgumentError("`prepare!` returned a new point set, which a workspace cannot keep: an immutable point set declares `is_value`"))
    return points
end
prepare_points!(held::HeldPoints, method, params::Vararg{Any, N}) where {N} = (held.points = prepare!(held.points, method, params...); held)

# `f` at the `i`-th point, or `first_value` for the first point when the caller has it.
@inline value_at(f::F, ws, i, first_value) where {F} =
    (i == 1 && !(first_value isa Nothing)) ? first_value : evaluate(f, ws.layout, point!(ws.points, i))

# A method may recognise inputs it cannot place points for, such as a zero covariance for the
# unscented transform, and return their centre: the result is then `f` at it, with no spread.
degenerate_input(method, params::Vararg{Any, N}) where {N} = nothing

## Expectations

"""
    LogIntegrand(logf)

A function given by its logarithm, `logf(x) = log f(x)`, for computations in log space that a
direct `f` would underflow or overflow. Called, it returns `exp(logf(x))`.

- [`expectation`](@ref)`(method, LogIntegrand(logf), params...)` returns `log E[f(x)]`: log in,
  log out.
- [`approximate_meancov`](@ref)`(method, LogIntegrand(logg), params...)` returns the moments of
  the measure reweighted by `g`, accurately however small `g` is over it.

# Examples

```jldoctest; setup = :(using MessagePassingRulesApproximations)
julia> sharp = LogIntegrand(x -> -1e4 * (x - 3)^2);   # exp(-1e4 (x - 3)²) underflows away from 3

julia> m, v = approximate_meancov(ghcubature(40), sharp, 0.0, 1.0);

julia> isfinite(m) && isfinite(v)
true
```
"""
struct LogIntegrand{F}
    logf::F
end

(f::LogIntegrand)(args...) = exp(f.logf(args...))

Base.log(f::LogIntegrand) = f.logf

primal(f::LogIntegrand) = log(f)

"""
    expectation(method, f, params...)
    expectation!(ws, method, f, params...)
    expectation!(out, ws, method, f, params...)

The expectation `E[f(x)] ≈ Σᵢ wᵢ f(xᵢ)` of `f` under the measure of `params`, by the point set
`method` places for it.

# Arguments

- `method`: a point-set method, such as [`Unscented`](@ref), [`GaussHermiteCubature`](@ref) or
  [`GaussLaguerreQuadrature`](@ref);
- `f`: a function of one point, returning a number or a vector; with several inputs, a function
  of one argument per input. A [`LogIntegrand`](@ref) gives the logarithm of the expectation,
  computed in log space;
- `params`: the measure's parameters, as `method` takes them: `(m, V)` for a normal, a mean and a
  variance or a vector and a covariance matrix; `(means, covs)`, tuples with one entry per
  independent normal input; `(shape, scale)` for the gamma of Gauss–Laguerre quadrature.

The in-place forms work in a [`workspace`](@ref)`(method, f, params...)`, and allocate nothing of
their own: `expectation!(ws, …)` for a scalar `f`, `expectation!(out, ws, …)` for a vector one,
writing the result into `out`.

# Examples

```jldoctest; setup = :(using MessagePassingRulesApproximations)
julia> expectation(ghcubature(3), x -> x^2, 1.0, 4.0) ≈ 5.0  # E[x²] = m² + v
true

julia> exp(expectation(ghcubature(20), LogIntegrand(x -> x), 0.0, 1.0)) ≈ exp(0.5)  # E[eˣ]
true
```
"""
function expectation(method::AbstractApproximationMethod, f::F, params::Vararg{Any, N}) where {F, N}
    ws, first_value = prepared_workspace(Val(true), method, f, params...)
    return ws.dy === nothing ? expectation_kernel(ws, f, first_value) : expectation_kernel!(similar(ws.dy), ws, f, first_value)
end

function expectation!(ws::PointSetWorkspace{P, L, <:AbstractVector, DX, Nothing}, method::AbstractApproximationMethod, f::F, params::Vararg{Any, N}) where {P, L, DX, F, N}
    return expectation_kernel(prepare_workspace!(ws, method, params...), f, nothing)
end

function expectation!(out::AbstractVector, ws::PointSetWorkspace, method::AbstractApproximationMethod, f::F, params::Vararg{Any, N}) where {F, N}
    return expectation_kernel!(out, prepare_workspace!(ws, method, params...), f, nothing)
end

function expectation_kernel(ws::PointSetWorkspace, f::F, first_value) where {F}
    points = ws.points
    result = weight(points, 1) * value_at(f, ws, 1, first_value)
    for i in 2:npoints(points)
        result += weight(points, i) * value_at(f, ws, i, nothing)
    end
    return result
end

function expectation_kernel!(out::AbstractVector, ws::PointSetWorkspace, f::F, first_value) where {F}
    points = ws.points
    fill!(out, zero(eltype(out)))
    for i in 1:npoints(points)
        w = weight(points, i)
        out .+= w .* value_at(f, ws, i, first_value)
    end
    return out
end

# In log space: log Σᵢ sᵢ exp(log |wᵢ| + log f(xᵢ)), with `sᵢ` the weights' signs, shifted by the
# largest term. A negative sum, which negative weights can give, has no logarithm.
function expectation_kernel(ws::PointSetWorkspace, f::LogIntegrand, first_value)
    points, terms = ws.points, with_log_terms(ws, f, first_value).ys
    largest = maximum(terms)
    isfinite(largest) || return largest
    total = zero(largest)
    for i in eachindex(terms)
        total += weightsign(points, i) * exp(terms[i] - largest)
    end
    return largest + log(total)
end

# The values the operations combine, in `ws.ys`, or, when it is a tuple, in a new workspace with
# a new tuple: `f`'s values, the weights of a reweighted measure, the terms of a log-space sum.
with_values(ws::PointSetWorkspace, f::F, first_value) where {F} = (evaluate_points!(ws, f, first_value); ws)
with_values(ws::PointSetWorkspace{P, L, <:NTuple{K, Any}}, f::F, first_value) where {P, L, K, F} =
    with_tuple(ws, ntuple(i -> value_at(f, ws, i, first_value), Val(K)))

with_tilted_weights(ws::PointSetWorkspace, g::G, first_value) where {G} = (tilted_weights!(ws, g, first_value); ws)
with_tilted_weights(ws::PointSetWorkspace{P, L, <:NTuple{K, Any}}, g::G, first_value) where {P, L, K, G} =
    with_tuple(ws, ntuple(i -> weight(ws.points, i) * value_at(g, ws, i, first_value), Val(K)))
function with_tilted_weights(ws::PointSetWorkspace{P, L, <:NTuple{K, Any}}, g::LogIntegrand, first_value) where {P, L, K}
    terms = ntuple(i -> logweight(ws.points, i) + value_at(log(g), ws, i, first_value), Val(K))
    largest = maximum(terms)
    return with_tuple(ws, ntuple(i -> weightsign(ws.points, i) * exp(terms[i] - largest), Val(K)))
end

function with_log_terms(ws::PointSetWorkspace, f::LogIntegrand, first_value)
    for i in 1:npoints(ws.points)
        ws.ys[i] = logweight(ws.points, i) + value_at(log(f), ws, i, first_value)
    end
    return ws
end
with_log_terms(ws::PointSetWorkspace{P, L, <:NTuple{K, Any}}, f::LogIntegrand, first_value) where {P, L, K} =
    with_tuple(ws, ntuple(i -> logweight(ws.points, i) + value_at(log(f), ws, i, first_value), Val(K)))

with_tuple(ws::PointSetWorkspace, ys::Tuple) = PointSetWorkspace(ws.points, ws.layout, ys, ws.dx, ws.dy)

# The values of a scalar `f`: a buffer, or a tuple for a fixed point set.
const ScalarValues = Union{AbstractVector, Tuple}

## The moments of f(x): `approximate` and `forward_statistics`

# The values of `f` at the points, into `ws.ys`.
function evaluate_points!(ws::PointSetWorkspace, f::F, first_value) where {F}
    for i in 1:npoints(ws.points)
        store_value!(ws.ys, i, value_at(f, ws, i, first_value))
    end
    return ws.ys
end

store_value!(ys::AbstractVector, i, y::Real) = (ys[i] = y; nothing)
function store_value!(ys::AbstractMatrix, i, y::AbstractVector)
    length(y) == size(ys, 1) || throw(DimensionMismatch(lazy"the function returned $(length(y)) values at point $i, and $(size(ys, 1)) at the first"))
    copyto!(view(ys, :, i), y)
    return nothing
end

# The mean and covariance of the values in `ws.ys`, and, for `Val(true)`, their cross-covariance
# with the points, written into the given outputs; a `nothing` output is a scalar, returned.
# A scalar `f`:
function pushforward_moments!(::Val{C}, cross, ws::PointSetWorkspace{P, L, <:ScalarValues, DX, Nothing}) where {C, P, L, DX}
    points, ys = ws.points, ws.ys
    n = npoints(points)
    m_tilde = weight(points, 1) * ys[1]
    for i in 2:n
        m_tilde += weight(points, i) * ys[i]
    end
    V_tilde = covweight(points, 1) * (ys[1] - m_tilde)^2
    for i in 2:n
        V_tilde += covweight(points, i) * (ys[i] - m_tilde)^2
    end
    C_tilde = C ? scalar_cross_covariance!(cross, points, ys, m_tilde) : nothing
    return (m_tilde, V_tilde, C_tilde)
end

# Scalar points: a scalar cross-covariance.
function scalar_cross_covariance!(::Nothing, points, ys, m_tilde)
    m = center(points)
    C_tilde = covweight(points, 1) * (point!(points, 1) - m) * (ys[1] - m_tilde)
    for i in 2:npoints(points)
        C_tilde += covweight(points, i) * (point!(points, i) - m) * (ys[i] - m_tilde)
    end
    return C_tilde
end

# Vector points: the cross-covariance is a vector.
function scalar_cross_covariance!(cross::AbstractVector, points, ys, m_tilde)
    m = center(points)
    fill!(cross, zero(eltype(cross)))
    for i in 1:npoints(points)
        x = point!(points, i)
        wc = covweight(points, i)
        for k in eachindex(cross)
            cross[k] += (wc * (x[k] - m[k])) * (ys[i] - m_tilde)
        end
    end
    return cross
end

# A vector `f`: the mean and covariance into `m_tilde` and `V_tilde`.
function pushforward_moments!(::Val{C}, m_tilde::AbstractVector, V_tilde::AbstractMatrix, cross, ws::PointSetWorkspace{P, L, <:AbstractMatrix}) where {C, P, L}
    points, ys, dy = ws.points, ws.ys, ws.dy
    n = npoints(points)
    fill!(m_tilde, zero(eltype(m_tilde)))
    for i in 1:n
        w = weight(points, i)
        for j in eachindex(m_tilde)
            m_tilde[j] += w * ys[j, i]
        end
    end
    fill!(V_tilde, zero(eltype(V_tilde)))
    for i in 1:n
        wc = covweight(points, i)
        for j in eachindex(dy)
            dy[j] = ys[j, i] - m_tilde[j]
        end
        for k in eachindex(dy), j in eachindex(dy)
            V_tilde[j, k] += wc * (dy[j] * dy[k])
        end
    end
    C && vector_cross_covariance!(cross, points, ys, m_tilde, dy)
    return (m_tilde, V_tilde, C ? cross : nothing)
end

# Scalar points: the cross-covariance is a row, `1 × length(f(x))`; vector points: a matrix,
# `length(x) × length(f(x))`.
function vector_cross_covariance!(cross::AbstractMatrix, points, ys, m_tilde, dy)
    m = center(points)
    fill!(cross, zero(eltype(cross)))
    for i in 1:npoints(points)
        x = point!(points, i)
        wc = covweight(points, i)
        for j in eachindex(dy)
            dy[j] = ys[j, i] - m_tilde[j]
        end
        add_cross!(cross, wc, x, m, dy)
    end
    return cross
end

function add_cross!(cross::AbstractMatrix, wc, x::Real, m::Real, dy)
    for j in eachindex(dy)
        cross[1, j] += (wc * (x - m)) * dy[j]
    end
    return cross
end

function add_cross!(cross::AbstractMatrix, wc, x::AbstractVector, m::AbstractVector, dy)
    for j in eachindex(dy), k in eachindex(x)
        cross[k, j] += (wc * (x[k] - m[k])) * dy[j]
    end
    return cross
end

# The array-valued outputs of `approximate` and `forward_statistics` for a workspace.
moment_outputs(ws::PointSetWorkspace) = (similar(ws.dy), similar(ws.dy, length(ws.dy), length(ws.dy)))
cross_output(ws::PointSetWorkspace{P, L, <:ScalarValues, Nothing}) where {P, L} = nothing
# The cross-covariance multiplies points by values: in the type both promote to.
cross_output(ws::PointSetWorkspace{P, L, <:ScalarValues, <:AbstractVector}) where {P, L} = similar(ws.dx, promote_type(eltype(ws.ys), eltype(ws.dx)))
cross_output(ws::PointSetWorkspace{P, L, <:AbstractMatrix, Nothing}) where {P, L} =
    similar(ws.dy, promote_type(eltype(ws.dy), typeof(center(ws.points))), 1, length(ws.dy))
cross_output(ws::PointSetWorkspace{P, L, <:AbstractMatrix, <:AbstractVector}) where {P, L} =
    similar(ws.dy, promote_type(eltype(ws.dy), eltype(ws.dx)), length(ws.dx), length(ws.dy))

"""
    approximate(method, f, params...) -> (m, V)
    approximate!(ws, method, f, params...) -> (m, V)
    approximate!(m, V, ws, method, f, params...) -> (m, V)

The mean `m` and variance, or covariance, `V` of `f(x)` under the measure of `params`, by the
point set of `method`: `m = Σᵢ wᵢ f(xᵢ)` and `V = Σᵢ w⁽ᶜ⁾ᵢ (f(xᵢ) - m)(f(xᵢ) - m)'`. The arguments
are [`expectation`](@ref)'s.

The in-place forms work in a [`workspace`](@ref)`(method, f, params...)` and allocate nothing of
their own: `approximate!(ws, …)` for a scalar `f`, `approximate!(m, V, ws, …)` for a vector one,
writing into the vector `m` and the matrix `V`.

[`Linearization`](@ref) has an `approximate` of its own, which returns the local linear map.

# Examples

```jldoctest; setup = :(using MessagePassingRulesApproximations)
julia> m, V = approximate(ghcubature(10), x -> 2x + 1, 1.0, 0.5);

julia> m ≈ 3.0 && V ≈ 2.0
true
```
"""
approximate(method::AbstractApproximationMethod, f::F, params::Vararg{Any, N}) where {F, N} = approximate_pointset(method, f, params...)

function approximate_pointset(method::AbstractApproximationMethod, f::F, params::Vararg{Any, N}) where {F, N}
    point = degenerate_input(method, params...)
    point === nothing || return degenerate_result(Val(false), f, input_layout(params...), point)
    ws, first_value = prepared_workspace(Val(true), method, f, params...)
    ws = with_values(ws, f, first_value)
    if ws.dy === nothing
        m_tilde, V_tilde, _ = pushforward_moments!(Val(false), nothing, ws)
        return (m_tilde, V_tilde)
    else
        m, V = moment_outputs(ws)
        pushforward_moments!(Val(false), m, V, nothing, ws)
        return (m, V)
    end
end

function approximate!(ws::PointSetWorkspace{P, L, <:AbstractVector, DX, Nothing}, method::AbstractApproximationMethod, f::F, params::Vararg{Any, N}) where {P, L, DX, F, N}
    point = degenerate_input(method, params...)
    if point !== nothing
        y = convert(eltype(ws.ys), evaluate(f, ws.layout, point))
        return (y, zero(y))
    end
    evaluate_points!(prepare_workspace!(ws, method, params...), f, nothing)
    m_tilde, V_tilde, _ = pushforward_moments!(Val(false), nothing, ws)
    return (m_tilde, V_tilde)
end

function approximate!(m::AbstractVector, V::AbstractMatrix, ws::PointSetWorkspace{P, L, <:AbstractMatrix}, method::AbstractApproximationMethod, f::F, params::Vararg{Any, N}) where {P, L, F, N}
    point = degenerate_input(method, params...)
    if point !== nothing
        copyto!(m, evaluate(f, ws.layout, point))
        fill!(V, zero(eltype(V)))
        return (m, V)
    end
    evaluate_points!(prepare_workspace!(ws, method, params...), f, nothing)
    pushforward_moments!(Val(false), m, V, nothing, ws)
    return (m, V)
end

"""
    forward_statistics(method, f, params...) -> (m, V, C)
    forward_statistics!(…, ws, method, f, params...) -> (m, V, C)

The mean `m` and covariance `V` of `f(x)`, as [`approximate`](@ref) computes them, and the
cross-covariance `C = Σᵢ w⁽ᶜ⁾ᵢ (xᵢ - x̄)(f(xᵢ) - m)'` between the input `x`, its several inputs
concatenated into one vector, and `f(x)`: the forward statistics [`smoothRTS`](@ref) takes. `C` is
a number for a scalar input and scalar `f`, a vector for a vector input and scalar `f`, a row
`1 × length(f(x))` for a scalar input and vector `f`, and a matrix otherwise.

The in-place form works in a [`workspace`](@ref)`(method, f, params...)`, allocates nothing of its
own, and takes the array-valued parts of the result first, in order: `forward_statistics!(ws, …)`
when all three are scalars, `forward_statistics!(C, ws, …)` for a vector input and scalar `f`,
and `forward_statistics!(m, V, C, ws, …)` for a vector `f`.

# Examples

```jldoctest; setup = :(using MessagePassingRulesApproximations)
julia> m, V, C = forward_statistics(ghcubature(10), x -> 2x + 1, 1.0, 0.5);

julia> m ≈ 3.0 && V ≈ 2.0 && C ≈ 1.0   # C = Cov(x, 2x + 1) = 2 × 0.5
true
```
"""
function forward_statistics(method::AbstractApproximationMethod, f::F, params::Vararg{Any, N}) where {F, N}
    point = degenerate_input(method, params...)
    point === nothing || return degenerate_result(Val(true), f, input_layout(params...), point)
    ws, first_value = prepared_workspace(Val(true), method, f, params...)
    ws = with_values(ws, f, first_value)
    cross = cross_output(ws)
    if ws.dy === nothing
        return pushforward_moments!(Val(true), cross, ws)
    else
        m, V = moment_outputs(ws)
        return pushforward_moments!(Val(true), m, V, cross, ws)
    end
end

function forward_statistics!(ws::PointSetWorkspace{P, L, <:AbstractVector, Nothing, Nothing}, method::AbstractApproximationMethod, f::F, params::Vararg{Any, N}) where {P, L, F, N}
    point = degenerate_input(method, params...)
    if point !== nothing
        y = convert(eltype(ws.ys), evaluate(f, ws.layout, point))
        return (y, zero(y), zero(y))
    end
    evaluate_points!(prepare_workspace!(ws, method, params...), f, nothing)
    return pushforward_moments!(Val(true), nothing, ws)
end

function forward_statistics!(C::AbstractVector, ws::PointSetWorkspace{P, L, <:AbstractVector, <:AbstractVector, Nothing}, method::AbstractApproximationMethod, f::F, params::Vararg{Any, N}) where {P, L, F, N}
    point = degenerate_input(method, params...)
    if point !== nothing
        y = convert(eltype(ws.ys), evaluate(f, ws.layout, point))
        return (y, zero(y), fill!(C, zero(eltype(C))))
    end
    evaluate_points!(prepare_workspace!(ws, method, params...), f, nothing)
    return pushforward_moments!(Val(true), C, ws)
end

function forward_statistics!(m::AbstractVector, V::AbstractMatrix, C::AbstractMatrix, ws::PointSetWorkspace{P, L, <:AbstractMatrix}, method::AbstractApproximationMethod, f::F, params::Vararg{Any, N}) where {P, L, F, N}
    point = degenerate_input(method, params...)
    if point !== nothing
        copyto!(m, evaluate(f, ws.layout, point))
        fill!(V, zero(eltype(V)))
        fill!(C, zero(eltype(C)))
        return (m, V, C)
    end
    evaluate_points!(prepare_workspace!(ws, method, params...), f, nothing)
    return pushforward_moments!(Val(true), m, V, C, ws)
end

# `f` at a point input, with no spread: zero covariance, and zero cross-covariance, which is a
# genuine zero rather than "not computed", since callers do arithmetic with it.
function degenerate_result(::Val{C}, f::F, layout, point) where {C, F}
    y = evaluate(f, layout, point)
    V = y isa Real ? zero(y) : zeros(eltype(y), length(y), length(y))
    C || return (y, V)
    cross = if y isa Real
        point isa Real ? zero(y) : zeros(eltype(y), length(point))
    else
        zeros(eltype(y), point isa Real ? 1 : length(point), length(y))
    end
    return (y, V, cross)
end

## The moments of a reweighted measure

"""
    approximate_meancov(method, g, params...) -> (mean, covariance)
    approximate_meancov!(ws, method, g, m::Real, v::Real) -> (mean, variance)
    approximate_meancov!(μ, Σ, ws, method, g, params...) -> (μ, Σ)

The mean and variance (or covariance) of the density proportional to `g(x) p(x)`, where `p` is
the measure of `params`, computed over the points `method` places for `p`. For a normal `p` this
is the projection of a normal reweighted by a likelihood onto a normal, the core of an
expectation-propagation update.

# Arguments

- `method`: a point-set method, such as [`GaussHermiteCubature`](@ref)`(p)`, or anything with
  [`getpoints`](@ref) and [`getweights`](@ref) for the measure;
- `g`: a non-negative function of a point, a number for a scalar measure and a vector otherwise.
  It need not be normalised, but must not vanish at every point. A [`LogIntegrand`](@ref)
  `LogIntegrand(log ∘ g)` computes the same in log space, for a `g` that underflows;
- `params`: the measure's parameters, such as the mean and variance, or mean vector and
  covariance matrix, of a normal.

The moments are in the float type of the measure's mean. The in-place forms work in a
[`workspace`](@ref)`(method, g, params...)` and allocate nothing of their own when `method` is a
point-set method.

# Examples

```jldoctest; setup = :(using MessagePassingRulesApproximations)
julia> m, v = approximate_meancov(ghcubature(21), x -> exp(-x^2 / 2), 0.0, 1.0);  # N(0, 1) × N(0, 1)

julia> isapprox(m, 0.0; atol = 1e-12) && v ≈ 0.5
true
```
"""
function approximate_meancov(method::AbstractApproximationMethod, g::G, params::Vararg{Any, N}) where {G, N}
    ws, first_value = prepared_workspace(Val(true), method, g, params...)
    ws = with_tilted_weights(ws, g, first_value)
    ws.dx === nothing && return meancov_kernel(ws, meancov_eltype(params...))
    T = meancov_eltype(params...)
    d = length(ws.dx)
    return meancov_kernel!(zeros(T, d), zeros(T, d, d), ws)
end

function approximate_meancov!(ws::PointSetWorkspace{P, L, <:AbstractVector, Nothing}, method::AbstractApproximationMethod, g::G, params::Vararg{Any, N}) where {P, L, G, N}
    ws = with_tilted_weights(prepare_workspace!(ws, method, params...), g, nothing)
    return meancov_kernel(ws, meancov_eltype(params...))
end

function approximate_meancov!(μ::AbstractVector, Σ::AbstractMatrix, ws::PointSetWorkspace{P, L, <:AbstractVector, <:AbstractVector}, method::AbstractApproximationMethod, g::G, params::Vararg{Any, N}) where {P, L, G, N}
    ws = with_tilted_weights(prepare_workspace!(ws, method, params...), g, nothing)
    return meancov_kernel!(μ, Σ, ws)
end

meancov_eltype(m::AbstractVector, P::AbstractMatrix) = promote_type(eltype(m), eltype(P))
meancov_eltype(m::Real, rest...) = float(typeof(m))

# The weights of the reweighted measure, `wᵢ g(xᵢ)`, into `ws.ys`; in log space, shifted so that
# the largest is one.
function tilted_weights!(ws::PointSetWorkspace, g::G, first_value) where {G}
    points, ys = ws.points, ws.ys
    for i in 1:npoints(points)
        ys[i] = weight(points, i) * value_at(g, ws, i, first_value)
    end
    return ys
end

function tilted_weights!(ws::PointSetWorkspace, g::LogIntegrand, first_value)
    points, ys = ws.points, ws.ys
    for i in 1:npoints(points)
        ys[i] = logweight(points, i) + value_at(log(g), ws, i, first_value)
    end
    largest = maximum(ys)
    for i in eachindex(ys)
        ys[i] = weightsign(points, i) * exp(ys[i] - largest)
    end
    return ys
end

# The moments in the measure's float type `R`, whatever the points', the weights' and `g`'s.
function meancov_kernel(ws::PointSetWorkspace, ::Type{R}) where {R}
    points, cs = ws.points, ws.ys
    norm = zero(R)
    mean = zero(R)
    for i in 1:npoints(points)
        c = convert(R, cs[i])
        mean += convert(R, point!(points, i)) * c
        norm += c
    end
    mean /= norm
    var = zero(R)
    for i in 1:npoints(points)
        var += convert(R, cs[i]) * (convert(R, point!(points, i)) - mean)^2
    end
    var /= norm
    return mean, var
end

function meancov_kernel!(μ::AbstractVector, Σ::AbstractMatrix, ws::PointSetWorkspace)
    points, cs, dx = ws.points, ws.ys, ws.dx
    fill!(μ, zero(eltype(μ)))
    norm = zero(eltype(μ))
    for i in 1:npoints(points)
        x = point!(points, i)
        c = cs[i]
        for k in eachindex(μ)
            μ[k] += x[k] * c
        end
        norm += c
    end
    μ ./= norm
    fill!(Σ, zero(eltype(Σ)))
    for i in 1:npoints(points)
        x = point!(points, i)
        c = cs[i]
        for k in eachindex(dx)
            dx[k] = x[k] - μ[k]
        end
        for k in eachindex(dx), j in eachindex(dx)
            Σ[j, k] += c * (dx[j] * dx[k])
        end
    end
    Σ ./= norm
    return μ, Σ
end

"""
    expectation!(ws, method, f, params...)
    expectation!(out, ws, method, f, params...)

The in-place form of [`expectation`](@ref), in a [`workspace`](@ref): the result for a scalar
`f`, written into `out` for a vector one. It allocates nothing of its own.
"""
expectation!

"""
    approximate!(ws, method, f, params...) -> (m, V)
    approximate!(m, V, ws, method, f, params...) -> (m, V)

The in-place form of [`approximate`](@ref), in a [`workspace`](@ref): the mean and variance for a
scalar `f`, the mean and covariance written into `m` and `V` for a vector one. It allocates
nothing of its own. [`Linearization`](@ref)'s returns its linear map instead.
"""
approximate!

"""
    forward_statistics!(ws, method, f, params...) -> (m, V, C)
    forward_statistics!(C, ws, method, f, params...) -> (m, V, C)
    forward_statistics!(m, V, C, ws, method, f, params...) -> (m, V, C)

The in-place form of [`forward_statistics`](@ref), in a [`workspace`](@ref), with the
array-valued parts of the result first: none when all three are scalars, `C` for a vector input
and a scalar `f`, and `m`, `V` and `C` for a vector `f`. It allocates nothing of its own.
"""
forward_statistics!

"""
    approximate_meancov!(ws, method, g, m::Real, v::Real) -> (mean, variance)
    approximate_meancov!(μ, Σ, ws, method, g, params...) -> (μ, Σ)

The in-place form of [`approximate_meancov`](@ref), in a [`workspace`](@ref)`(method, g,
params...)`: the moments for a scalar measure, written into `μ` and `Σ` for a vector one. It
allocates nothing of its own for a point-set method.
"""
approximate_meancov!

## A method with `getpoints` and `getweights` only

# A method of another package that defines `getpoints` and `getweights` for a normal, and not the
# point-set interface, still works with every generic operation: its points are collected on
# each `prepare!`, which allocates.
mutable struct GeneratedPoints{X, W, C}
    points::Vector{X}
    weights::Vector{W}
    center::C
end

function pointset(method::AbstractApproximationMethod, m::Real, v::Real)
    return GeneratedPoints(collect(getpoints(method, m, v)), collect(getweights(method, m, v)), m)
end

function pointset(method::AbstractApproximationMethod, m::AbstractVector, P::AbstractMatrix)
    return GeneratedPoints([copy(x) for x in getpoints(method, m, P)], collect(getweights(method, m, P)), copy(m))
end

function prepare!(points::GeneratedPoints, method::AbstractApproximationMethod, m::Real, v::Real)
    points.points = collect(getpoints(method, m, v))
    points.weights = collect(getweights(method, m, v))
    points.center = m
    return points
end

function prepare!(points::GeneratedPoints, method::AbstractApproximationMethod, m::AbstractVector, P::AbstractMatrix)
    points.points = [copy(x) for x in getpoints(method, m, P)]
    points.weights = collect(getweights(method, m, P))
    copyto!(points.center, m)
    return points
end

npoints(points::GeneratedPoints) = length(points.points)
point!(points::GeneratedPoints, i) = points.points[i]
weight(points::GeneratedPoints, i) = points.weights[i]
center(points::GeneratedPoints) = points.center

## Several independent inputs, and one input given as a tuple

# One input in a tuple is that input; several are one joint input, with their means
# concatenated and their covariances on the block diagonal.
pointset(method::AbstractApproximationMethod, means::Tuple{Real}, covs::Tuple{Real}) = pointset(method, first(means), first(covs))
pointset(method::AbstractApproximationMethod, means::Tuple{AbstractVector}, covs::Tuple{AbstractMatrix}) = pointset(method, first(means), first(covs))
prepare!(points, method::AbstractApproximationMethod, means::Tuple, covs::Tuple) = prepare_inputs!(points, method, means, covs)
# A point set that is not joint was built for one input.
prepare_inputs!(points, method, means::Tuple, covs::Tuple) = prepare!(points, method, only(means), only(covs))

struct JointPoints{P, M, V}
    inner::P
    m::M
    V::V
end

function pointset(method::AbstractApproximationMethod, means::Tuple, covs::Tuple)
    m, V, _ = joint_mean_cov(means, covs)
    return JointPoints(pointset(method, m, V), m, V)
end

function prepare_inputs!(points::JointPoints, method, means::Tuple, covs::Tuple)
    joint_mean_cov!(points.m, points.V, means, covs)
    prepare!(points.inner, method, points.m, points.V)
    return points
end

npoints(points::JointPoints) = npoints(points.inner)
point!(points::JointPoints, i) = point!(points.inner, i)
weight(points::JointPoints, i) = weight(points.inner, i)
covweight(points::JointPoints, i) = covweight(points.inner, i)
weightsign(points::JointPoints, i) = weightsign(points.inner, i)
logweight(points::JointPoints, i) = logweight(points.inner, i)
center(points::JointPoints) = center(points.inner)
