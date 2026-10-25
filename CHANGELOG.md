# Changelog

All notable changes to MessagePassingRulesApproximations.jl are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-10-25

### Added

- One interface for the point-set methods, `pointset`, `prepare!`, `npoints`, `point!`, `weight`
  and `center` (public), and the operations written once over it for every method:
  `expectation`, `approximate`, `forward_statistics` (the forward statistics `smoothRTS` takes)
  and `approximate_meancov`.
- An in-place form of every operation, `op!(out..., ws, method, f, params...)`, which works in a
  `workspace(method, f, params...)` built once and allocates nothing of its own: `approximate!`,
  `forward_statistics!`, `expectation!`, `approximate_meancov!` and `smoothRTS!`. Linearization
  keeps its ForwardDiff configuration in its workspace.
- `LogIntegrand`: expectations and reweighted moments in log space, for integrands that
  underflow.
- `GaussLaguerreQuadrature` (`glquadrature`): expectations under a gamma, with nodes computed in
  place for every shape.
- `GenUnscented` and `GenConstrainedUnscented` (public, not exported): the generalized unscented
  transform of Ebeigbe et al., which matches the third and fourth central moments too, and its
  variant with the sigma points within bounds.
- `forward_statistics(::Linearization, …)`, the statistics of a linearisation, which the Delta
  node computed itself.

### Fixed

- The multivariate `getpoints` and `getweights` of `GaussHermiteCubature` were not type-stable,
  and allocated at every point.
- A zero-covariance input to the unscented transform gives a cross-covariance of the shape a
  non-degenerate one has: a vector for a vector input and a scalar output, a `1 × n` row for a
  scalar input and an output of length `n`.

## [1.0.0] - 2026-10-06

The first release: the numerics the node packages of the ReactiveMP ecosystem share (the unscented transform, local
linearisation, Gauss–Hermite cubature and a Rauch–Tung–Striebel smoother), developed in the
[ReactiveMP](https://github.com/ReactiveBayes/ReactiveMP.jl) repository, whose history this
repository keeps.

[Unreleased]: https://github.com/ReactiveBayes/MessagePassingRulesApproximations.jl/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/ReactiveBayes/MessagePassingRulesApproximations.jl/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/ReactiveBayes/MessagePassingRulesApproximations.jl/releases/tag/v1.0.0
