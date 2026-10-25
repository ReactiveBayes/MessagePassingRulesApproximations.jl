# [Expectations, in place](@id approximations-api)

Every point-set method of this package, [`Unscented`](@ref), [`GaussHermiteCubature`](@ref),
[`GaussLaguerreQuadrature`](@ref) and the [generalized unscented
transform](@ref approximations-generalized-unscented), approximates an expectation the same way:
it places weighted points ``x_i`` for a measure, and

```math
\mathrm{E}[f(x)] ≈ \sum_i w_i f(x_i).
```

The measure is given by plain parameters, never by a distribution: `(m, V)` for a normal, a
number and a variance or a vector and a covariance matrix; `(means, covs)`, tuples with one entry
per independent normal input; `(shape, scale)` for a gamma; `(m, V, S, K)` for the generalized
unscented transform's higher moments. Everything else is that sum, written once for every method:

| function | returns |
|---|---|
| [`expectation`](@ref)`(method, f, params...)` | ``\sum_i w_i f(x_i)`` |
| [`approximate`](@ref)`(method, f, params...)` | the mean and covariance of ``f(x)`` |
| [`forward_statistics`](@ref)`(method, f, params...)` | those and the cross-covariance of ``x`` and ``f(x)``, which [`smoothRTS`](@ref) takes |
| [`approximate_meancov`](@ref)`(method, g, params...)` | the mean and covariance of the measure reweighted by ``g`` |

A moment, a cross-covariance, the moments of a reweighted measure: each is an expectation of a
transformed function, so no method needs its own.

```@example api
using MessagePassingRulesApproximations

f(x) = sin(x) + x^2
m, v = 0.5, 0.3

(
    unscented = approximate(Unscented(), f, m, v),
    cubature = approximate(ghcubature(20), f, m, v),
    E = expectation(ghcubature(20), f, m, v),
)
```

## Log space

A [`LogIntegrand`](@ref) gives a function by its logarithm. [`expectation`](@ref) then returns
the logarithm of the expectation, and [`approximate_meancov`](@ref) the moments of the measure
reweighted by a function that would underflow everywhere, as a sharp likelihood far from the
prior does:

```@example api
sharp(x) = -(x - 30)^2 / 2e-3   # the log of a likelihood far from N(0, 1)

(
    direct = approximate_meancov(ghcubature(20), x -> exp(sharp(x)), 0.0, 1.0),
    log_space = approximate_meancov(ghcubature(20), LogIntegrand(sharp), 0.0, 1.0),
)
```

## In place, with nothing allocated

Every operation `op(method, f, params...)` has an in-place form,

```julia
op!(out..., ws, method, f, params...)
```

with three rules:

1. **Outputs first.** The leading `out...` are the array-valued parts of the result, in order,
   written in place; the scalar parts are returned. A scalar result has no `out`:
   `approximate!(ws, method, f, m, v)` returns two numbers, `approximate!(m, V, ws, method, f,
   μ, Σ)` writes a vector and a matrix.
2. **One workspace,** [`workspace`](@ref MessagePassingRulesApproximations.workspace)`(method, f, params...)`, with the arguments of the
   operation it serves. It holds every buffer the operation needs, built once; it calls `f` once
   to learn the size of its output. Then any number of calls, with parameters of other values and
   the same shapes, allocate nothing of their own. What `f` itself allocates is its own business:
   a function that returns a static vector allocates nothing.
3. **The allocating form calls the in-place one.** `op(method, f, params...)` builds the
   workspace and the outputs and calls the same code, evaluating `f` once per point.

```@example api
import MessagePassingRulesApproximations: workspace

gh = ghcubature(10)
ws = workspace(gh, f, m, v)
approximate!(ws, gh, f, 0.1, 0.2)   # compile
@allocated approximate!(ws, gh, f, 0.1, 0.2)
```

The same in three dimensions, with a vector output written into `μ` and `Σ`:

```@example api
g(x) = [x[1] * x[2], sin(x[3])]
μ_in, Σ_in = [0.1, 0.2, 0.3], [1.0 0.1 0.0; 0.1 1.0 0.1; 0.0 0.1 1.0]
ut = Unscented()
ws = workspace(ut, g, μ_in, Σ_in)
μ, Σ = zeros(2), zeros(2, 2)
approximate!(μ, Σ, ws, ut, g, μ_in, Σ_in)   # compile
g(μ_in)
(call = @allocated(approximate!(μ, Σ, ws, ut, g, μ_in, Σ_in)), by_g = 7 * @allocated(g(μ_in)))
```

What the call allocates is `g`'s: a new vector at each of the seven sigma points. A function that
returns a static vector, from StaticArrays, or a number, allocates nothing, and the call then
allocates nothing at all.

[`Linearization`](@ref) keeps its ForwardDiff configuration in its workspace, and
[`smoothRTS!`](@ref) its factorisation.

### A workspace as a rule's scratch

A workspace suits the scratch of a MessagePassingRulesBase rule (see
[scratch](@extref MessagePassingRulesBase glossary-scratch)): it is written before it is read on
every call, so a workspace reused, or filled with garbage, gives the same result as a fresh one;
no result points into it; and its type follows from the types of the inputs, so an engine infers
it. A rule declares it from its inputs and calls the in-place form in its body:

```julia
@define_message_update_rule(
    node = DeltaFn, target = :out, algorithm = DeltaApproximation{<:Unscented},
    args = (m[:in...]::NormalDistributionsFamily,),
    scratch = (algo, ctx, args) -> workspace(algo.method, f, means(args), covs(args)),
    body = (scratch, algo, ctx, args) -> begin
        m, V = approximate!(scratch, algo.method, f, means(args), covs(args))
        NormalMeanVariance(m, V)
    end,
)
```

## Adding a method

A method becomes a point-set method, with every operation above and their in-place forms, by
implementing six functions for its point set, a type of its own:

- [`pointset`](@ref MessagePassingRulesApproximations.pointset)`(method, params...)`: the point set, its buffers allocated for parameters of
  these shapes;
- [`prepare!`](@ref MessagePassingRulesApproximations.prepare!)`(points, method, params...)`: places the points, writing every buffer it
  reads;
- [`npoints`](@ref MessagePassingRulesApproximations.npoints), [`point!`](@ref MessagePassingRulesApproximations.point!), [`weight`](@ref MessagePassingRulesApproximations.weight) and [`center`](@ref MessagePassingRulesApproximations.center), and optionally
  [`covweight`](@ref MessagePassingRulesApproximations.covweight), [`logweight`](@ref MessagePassingRulesApproximations.logweight) and [`weightsign`](@ref MessagePassingRulesApproximations.weightsign).

A method that defines only [`getpoints`](@ref) and [`getweights`](@ref) for a normal still works
with every operation, collecting its points on each call, which allocates.

## API

```@docs
expectation
expectation!
approximate(::AbstractApproximationMethod, ::F, ::Vararg{Any, N}) where {F, N}
approximate!
forward_statistics(::AbstractApproximationMethod, ::F, ::Vararg{Any, N}) where {F, N}
forward_statistics!
LogIntegrand
MessagePassingRulesApproximations.AbstractWorkspace
MessagePassingRulesApproximations.workspace
MessagePassingRulesApproximations.pointset
MessagePassingRulesApproximations.prepare!
MessagePassingRulesApproximations.npoints
MessagePassingRulesApproximations.point!
MessagePassingRulesApproximations.weight
MessagePassingRulesApproximations.covweight
MessagePassingRulesApproximations.logweight
MessagePassingRulesApproximations.weightsign
MessagePassingRulesApproximations.center
```
