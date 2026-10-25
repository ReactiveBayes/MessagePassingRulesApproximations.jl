# MessagePassingRulesApproximations

[![CI](https://github.com/ReactiveBayes/MessagePassingRulesApproximations.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ReactiveBayes/MessagePassingRulesApproximations.jl/actions/workflows/CI.yml)
[![Docs: stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://reactivebayes.github.io/MessagePassingRulesApproximations.jl/stable/)
[![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://reactivebayes.github.io/MessagePassingRulesApproximations.jl/dev/)
[![Coverage](https://codecov.io/gh/ReactiveBayes/MessagePassingRulesApproximations.jl/graph/badge.svg)](https://codecov.io/gh/ReactiveBayes/MessagePassingRulesApproximations.jl)
[![Code style: Runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)

Numerics for propagating moments through a function, and for expectations under a normal or a
gamma: the unscented transform and its generalisation to higher moments, local linearisation,
Gauss–Hermite cubature, Gauss–Laguerre quadrature and a Rauch–Tung–Striebel smoother. It works on
means, covariances and other plain parameters, not on distributions, and knows nothing of message
passing. Every operation has an in-place form that allocates nothing. The node packages of the
[ReactiveMP](https://github.com/ReactiveBayes/ReactiveMP.jl) ecosystem build their rules on it.

```julia
import Pkg; Pkg.add("MessagePassingRulesApproximations")
```

```julia
using MessagePassingRulesApproximations

m, V = approximate(Unscented(), x -> 2x + 1, (1.0,), (0.5,))   # ≈ (3.0, 2.0)
A, b = approximate(Linearization(), x -> x^2, (3.0,))          # (6.0, -9.0)
E = expectation(ghcubature(10), x -> x^2, 1.0, 4.0)            # ≈ 5.0, E[x²] = m² + v

# In place: a workspace built once, then no allocation per call.
import MessagePassingRulesApproximations: workspace
ws = workspace(ghcubature(10), sin, 0.0, 1.0)
m, V = approximate!(ws, ghcubature(10), sin, 0.5, 0.2)
```

- Documentation: <https://reactivebayes.github.io/MessagePassingRulesApproximations.jl/stable/>. `make docs` builds it
  locally, into `docs/build`.
- Benchmarks: `julia --project=benchmark benchmark/benchmarks.jl`.
- Tests: `make test` runs the suite as CI does; `make test test_args="name:Gauss"` selects items
  by tag, by name or by file. `make help` lists the other targets: `docs`, `docs-serve`, `format`,
  `check-format`, `test-fast`.
- Julia 1.10 or later. MIT licence.
