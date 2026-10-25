# Gauss–Laguerre quadrature

Gauss–Laguerre quadrature approximates an expectation under a gamma distribution by a weighted
sum of the function at fixed points. It suits a positive quantity, a rate or a precision, whose
belief is a gamma.

## The method

For a gamma with shape ``α`` and scale ``θ``, the substitution ``x = θ t`` turns the expectation
into an integral against ``t^{α - 1} e^{-t}``, which the generalised Gauss–Laguerre rule with
`p` nodes ``t_i`` and weights ``w_i`` evaluates:

```math
\mathrm{E}[f(x)] = \frac{1}{Γ(α)} \int_0^∞ f(θ t)\, t^{α - 1} e^{-t} \, \mathrm{d}t
≈ \sum_{i=1}^{p} \bar w_i f(θ t_i), \qquad \bar w_i = \frac{w_i}{\sum_j w_j}.
```

The nodes are the roots of the generalised Laguerre polynomial of degree `p`, so the sum is exact
when `f` is a polynomial of degree up to `2p - 1`. The rule depends on the shape, so its nodes
are computed on every call: they are the eigenvalues of the Jacobi matrix of the Laguerre
polynomials (Golub and Welsch), found in place and polished by Newton's method, so a call
allocates nothing, whatever the shape; sixteen nodes cost a few microseconds. An exponential with
rate ``λ`` is a gamma of shape `1` and scale `1 / λ`.

Five points reproduce the first nine raw moments of a gamma, ``\mathrm{E}[x^k] = θ^k α (α + 1)
⋯ (α + k - 1)``:

```@example gauss-laguerre
using MessagePassingRulesApproximations

gl = GaussLaguerreQuadrature(5)
α, θ = 2.5, 0.8
[(expectation(gl, x -> x^k, α, θ), θ^k * prod(α + j for j in 0:(k - 1))) for k in 1:4]
```

A function that is not smooth at zero, such as ``\log x``, converges slowly for a shape below
one, where the gamma's density is singular there; it converges fast for larger shapes.

## Through a function, and reweighted

The pushforward of a gamma through a function, as a mean and variance, is
[`approximate`](@ref); the moments of a gamma reweighted by a likelihood are
[`approximate_meancov`](@ref), in log space with a [`LogIntegrand`](@ref):

```@example gauss-laguerre
precision = 2.0, 0.5                      # a gamma belief on a precision
loglikelihood(τ) = 0.5 * log(τ) - τ * 1.7  # N(y | 0, 1 / τ) for an observation y with y² / 2 = 1.7
(
    pushforward = approximate(gl, τ -> 1 / sqrt(τ), precision...),
    posterior = approximate_meancov(glquadrature(20), LogIntegrand(loglikelihood), precision...),
)
```

The posterior is a gamma with shape ``α + 1/2`` and rate ``1/θ + 1.7``: mean ``2.5 / 3.7 ≈ 0.68``.

## API

```@docs
GaussLaguerreQuadrature
glquadrature
```
