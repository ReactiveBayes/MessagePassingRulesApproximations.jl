# The cost of every method and operation, allocating and in place. Run it with
# `julia --project=benchmark benchmark/benchmarks.jl`, which prints the median time and the
# allocations of each; `SUITE` is a BenchmarkTools suite for other runners.
using BenchmarkTools, LinearAlgebra, StaticArrays
using MessagePassingRulesApproximations
import MessagePassingRulesApproximations: workspace, GenUnscented

const SUITE = BenchmarkGroup()

scalar(x) = sin(x) + x^2
vector(x) = SVector(x[1] * x[2], sin(x[3]), x[1]^2)
likelihood(x) = exp(-sum(abs2, x) / 2)

m, v = 0.5, 0.3
μ, Σ = [0.1, 0.2, 0.3], [1.0 0.1 0.0; 0.1 1.0 0.1; 0.0 0.1 1.0]

for method in (Unscented(), ghcubature(5), GenUnscented())
    name = approximation_short_name(method)
    group = SUITE[name] = BenchmarkGroup()
    ws = workspace(method, scalar, m, v)
    group["approximate, scalar"] = @benchmarkable approximate($method, scalar, $m, $v)
    group["approximate!, scalar"] = @benchmarkable approximate!($ws, $method, scalar, $m, $v)
    wsv = workspace(method, vector, μ, Σ)
    mo, Vo, Co = zeros(3), zeros(3, 3), zeros(3, 3)
    group["forward_statistics, 3 → 3"] = @benchmarkable forward_statistics($method, vector, $μ, $Σ)
    group["forward_statistics!, 3 → 3"] = @benchmarkable forward_statistics!($mo, $Vo, $Co, $wsv, $method, vector, $μ, $Σ)
    wsm = workspace(method, likelihood, μ, Σ)
    group["approximate_meancov!, 3"] = @benchmarkable approximate_meancov!($mo, $Vo, $wsm, $method, likelihood, $μ, $Σ)
end

gl = GaussLaguerreQuadrature(16)
SUITE["GL16"] = BenchmarkGroup()
wsg = workspace(gl, scalar, 2.0, 1.5)
SUITE["GL16"]["expectation"] = @benchmarkable expectation($gl, scalar, 2.0, 1.5)
SUITE["GL16"]["expectation!"] = @benchmarkable expectation!($wsg, $gl, scalar, 2.0, 1.5)

lin = Linearization()
SUITE["LN"] = BenchmarkGroup()
wsl = workspace(lin, vector, (μ,))
A, b = zeros(3, 3), zeros(3)
SUITE["LN"]["approximate, 3 → 3"] = @benchmarkable approximate($lin, vector, ($μ,))
SUITE["LN"]["approximate!, 3 → 3"] = @benchmarkable approximate!($A, $b, $wsl, $lin, vector, ($μ,))

if abspath(PROGRAM_FILE) == @__FILE__
    results = run(SUITE; verbose = false, seconds = 1)
    for (group, trials) in sort(collect(results); by = first), (name, trial) in sort(collect(trials); by = first)
        t = median(trial)
        println(rpad("$group  $name", 44), lpad(BenchmarkTools.prettytime(t.time), 12), lpad("$(t.allocs) allocs", 12), lpad(BenchmarkTools.prettymemory(t.memory), 12))
    end
end
