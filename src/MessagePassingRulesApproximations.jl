"""
    MessagePassingRulesApproximations

Numerics for expectations under a measure, and for propagating moments through a function.
Everything here takes and returns means, covariances and other plain parameters, numbers and
arrays, never distributions, and knows nothing of message passing: node packages, such as the
Delta node's, build their rules on it.

- the point-set methods: the unscented transform, [`Unscented`](@ref), and its generalisation to
  higher moments, `GenUnscented`; Gauss–Hermite cubature under a normal,
  [`GaussHermiteCubature`](@ref); Gauss–Laguerre quadrature under a gamma,
  [`GaussLaguerreQuadrature`](@ref). Each places weighted points, and
  [`expectation`](@ref), [`approximate`](@ref), [`forward_statistics`](@ref) and
  [`approximate_meancov`](@ref) are the same weighted sums for all of them;
- [`Linearization`](@ref): the first-order expansion of `f` at the inputs' means, by automatic
  differentiation, through [`approximate`](@ref) and [`local_linearization`](@ref);
- [`smoothRTS`](@ref): the Rauch–Tung–Striebel correction of an input's marginal, from the
  forward statistics of any method and a backward message.

Every operation has an in-place form, such as [`approximate!`](@ref), which works in a
`workspace` built once and allocates nothing of its own.

# Examples

```jldoctest; setup = :(using MessagePassingRulesApproximations)
julia> m, V = approximate(Unscented(), x -> 2x + 1, (1.0,), (0.5,));

julia> m ≈ 3.0 && V ≈ 2.0
true

julia> approximate(Linearization(), x -> x^2, (3.0,))
(6.0, -9.0)
```
"""
module MessagePassingRulesApproximations

using LinearAlgebra
using FastCholesky: cholinv, cholsqrt, fastcholesky!
import ForwardDiff, FastGaussQuadrature
using Compat: @compat

export AbstractApproximationMethod, approximation_name, approximation_short_name
export Unscented, UT, UnscentedTransform
export approximate, approximate!, forward_statistics, forward_statistics!, expectation, expectation!, LogIntegrand
export unscented_statistics, sigma_points_weights, smoothRTS, smoothRTS!
export Linearization, local_linearization
export GaussHermiteCubature, ghcubature, getweights, getpoints, approximate_meancov, approximate_meancov!
export GaussLaguerreQuadrature, glquadrature

@compat public sigma_point_parameters
@compat public GenUnscented, GenUT, GenUnscentedTransform, GenConstrainedUnscented, GenCUT
@compat public AbstractWorkspace, workspace, pointset, prepare!, npoints, point!, weight, covweight, logweight, weightsign, center, is_value

include("approximations.jl")
include("shared.jl")
include("pointsets.jl")
include("unscented.jl")
include("linearization.jl")
include("gausshermite.jl")
include("gausslaguerre.jl")
include("genunscented.jl")
include("rts.jl")

end
