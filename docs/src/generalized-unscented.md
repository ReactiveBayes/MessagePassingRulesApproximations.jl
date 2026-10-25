# [The generalized unscented transform](@id approximations-generalized-unscented)

The generalized unscented transform of Ebeigbe et al. places `2d + 1` sigma points, as many as
the [unscented transform](@ref "The unscented transform") does, but matches more of the input:
its mean and covariance, and the third and fourth central moments of each of its components. It
suits an input that is not normal, skewed or heavy-tailed, whose moments are known. For a normal
input it matches the fourth moment too, which the unscented transform with its default
parameters does not.

```@example genut
using MessagePassingRulesApproximations
import MessagePassingRulesApproximations: GenUnscented, GenConstrainedUnscented, sigma_points_weights
```

`GenUnscented` and `GenConstrainedUnscented` are public, not exported: RxGPs exports names of its
own for its copy of the transform, which would otherwise clash.

## The sigma points

For a scalar input with mean ``m``, variance ``σ^2`` and central moments ``S = \mathrm{E}[(x -
m)^3]`` and ``K = \mathrm{E}[(x - m)^4]``, the points are ``m``, ``m - uσ`` and ``m + vσ``, with

```math
u = \tfrac{1}{2}\left(-s + \sqrt{4k - 3s^2}\right), \quad v = u + s, \qquad
w^+ = \frac{1}{v (u + v)}, \quad w^- = \frac{v}{u} w^+, \quad w_0 = 1 - w^- - w^+,
```

where ``s = S/σ^3`` and ``k = K/σ^4`` are the standardised skewness and kurtosis. A vector input
takes them per component, along the columns of the Cholesky factor of its covariance. A gamma's
points reproduce its four moments:

```@example genut
α, θ = 2.0, 1.5
m, V, S, K = α * θ, α * θ^2, 2α * θ^3, 3α * (α + 2) * θ^4   # a gamma's central moments
points, weights = sigma_points_weights(GenUnscented(), m, V, S, K)
[sum(w * (x - m)^k for (w, x) in zip(weights, points)) for k in 2:4], (V, S, K)
```

Without `S` and `K` the input is normal, with no skewness and ``K = 3σ^4``. A quartic shows the
gain over the unscented transform, which is exact to degree three for a normal:

```@example genut
f(x) = x^4
m, v = 0.5, 0.3
(exact = m^4 + 6m^2 * v + 3v^2, generalized = expectation(GenUnscented(), f, m, v), unscented = expectation(Unscented(), f, m, v))
```

## Bounds

`GenConstrainedUnscented(lower, upper)` keeps the points within bounds, for an input that cannot
leave them, a positive one for instance. A point that would fall outside moves towards the mean;
the mean and covariance are still matched:

```@example genut
points, weights = sigma_points_weights(GenConstrainedUnscented(0.0, Inf), 1.0, 4.0)
(points = points, mean = sum(weights .* points), variance = sum(weights .* (points .- 1) .^ 2))
```

## API

```@docs
MessagePassingRulesApproximations.GenUnscented
MessagePassingRulesApproximations.GenUT
MessagePassingRulesApproximations.GenUnscentedTransform
MessagePassingRulesApproximations.GenConstrainedUnscented
MessagePassingRulesApproximations.GenCUT
sigma_points_weights(::MessagePassingRulesApproximations.AnyGenUnscented, ::Vararg{Any, N}) where {N}
```
