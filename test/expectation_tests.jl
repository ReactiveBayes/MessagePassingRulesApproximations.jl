# The operations every point-set method shares, against exact moments, and the contracts of the
# in-place forms: the same results as the allocating ones, on a workspace reused for other
# parameters or filled with garbage, with `f` evaluated once per point.

@testitem "expectation, approximate and forward_statistics: exact on low-degree polynomials" tags = [:approximations] begin
    using MessagePassingRulesApproximations, LinearAlgebra
    import MessagePassingRulesApproximations: GenUnscented

    # Under N(m, v): E[x²] = m² + v, Var[x²] = 4m²v + 2v², Cov(x, x²) = 2mv.
    m, v = 0.5, 2.0
    for method in (Unscented(; alpha = 1.0), ghcubature(5), GenUnscented())
        @test expectation(method, x -> x^2, m, v) ≈ m^2 + v
        @test expectation(method, x -> x^2, (m,), (v,)) ≈ m^2 + v
        mean, var = approximate(method, x -> x^2, m, v)
        @test mean ≈ m^2 + v
        method isa Unscented || @test var ≈ 4m^2 * v + 2v^2
        M, V, C = forward_statistics(method, x -> 2x + 1, m, v)
        @test M ≈ 2m + 1 && V ≈ 4v && C ≈ 2v
    end

    # A linear map of a vector: exact for every method, with the cross-covariance Σ A'.
    μ, Σ = [1.0, -2.0], [2.0 0.5; 0.5 1.0]
    A, b = [1.0 2.0; -1.0 0.5; 0.0 3.0], [0.1, 0.2, 0.3]
    for method in (Unscented(), ghcubature(3), GenUnscented())
        M, V, C = forward_statistics(method, x -> A * x + b, μ, Σ)
        @test M ≈ A * μ + b
        @test V ≈ A * Σ * A'
        @test C ≈ Σ * A'
        @test expectation(method, x -> A * x + b, μ, Σ) ≈ A * μ + b
    end

    # Several inputs are one joint input: the cross-covariance is with their concatenation.
    means, covs = (1.0, [0.0, 2.0]), (0.5, [1.0 0.2; 0.2 1.0])
    M, V, C = forward_statistics(ghcubature(4), (x, y) -> x + sum(y), means, covs)
    @test M ≈ 3.0 && V ≈ 0.5 + 2.4 && C ≈ [0.5, 1.2, 1.2]
    # Linearization gives the same statistics for a linear function.
    @test all(forward_statistics(Linearization(), (x, y) -> x + sum(y), means, covs) .≈ (M, V, C))
    @test all(approximate(Linearization(), (x, y) -> x + sum(y), means, covs) .≈ (M, V))
end

@testitem "LogIntegrand: log space where a direct integrand underflows" tags = [:approximations] begin
    using MessagePassingRulesApproximations

    # A benign integrand: the same as the direct sum.
    for method in (ghcubature(20), Unscented(; alpha = 1.0), GaussLaguerreQuadrature(10))
        params = method isa GaussLaguerreQuadrature ? (2.0, 0.5) : (0.3, 0.8)
        direct = expectation(method, x -> exp(-x^2), params...)
        @test exp(expectation(method, LogIntegrand(x -> -x^2), params...)) ≈ direct
        @test all(approximate_meancov(method, LogIntegrand(x -> -x^2), params...) .≈ approximate_meancov(method, x -> exp(-x^2), params...))
    end

    # A sharp likelihood far from the prior: the direct weights underflow to zero everywhere,
    # and 0 / 0 is NaN; in log space the moments are finite, the mass on the nodes nearest to it.
    m, v, a, s = 0.0, 1.0, 30.0, 1.0e-3
    sharp(x) = exp(-(x - a)^2 / 2s)
    @test all(isnan, approximate_meancov(ghcubature(20), sharp, m, v))
    mean, var = approximate_meancov(ghcubature(20), LogIntegrand(x -> -(x - a)^2 / 2s), m, v)
    @test isfinite(mean) && isfinite(var)
    @test mean ≈ maximum(getpoints(ghcubature(20), m, v)) rtol = 1.0e-6

    # log E[e^(x + 750)] = 750 + 1/2 under N(0, 1), though e^750 overflows.
    @test expectation(ghcubature(30), LogIntegrand(x -> x + 750), 0.0, 1.0) ≈ 750.5
    @test isinf(expectation(ghcubature(30), x -> exp(x + 750), 0.0, 1.0))
end

@testitem "in-place forms: the allocating results, on reused and poisoned workspaces" tags = [:approximations] begin
    using MessagePassingRulesApproximations, LinearAlgebra
    import MessagePassingRulesApproximations: workspace, GenUnscented, GenConstrainedUnscented, PointSetWorkspace

    # Every buffer of a workspace, filled with NaN: a workspace is written before it is read.
    function poison!(ws::PointSetWorkspace)
        points = ws.points
        foreach(name -> poison_field!(points, name), fieldnames(typeof(points)))
        fill!(ws.ys, NaN)
        ws.dx === nothing || fill!(ws.dx, NaN)
        ws.dy === nothing || fill!(ws.dy, NaN)
        return ws
    end
    poison_field!(points, name) = poison_value!(points, name, getfield(points, name))
    poison_value!(points, name, value::AbstractArray{<:AbstractFloat}) = fill!(value, NaN)
    poison_value!(points, name, value::AbstractFloat) = ismutable(points) && setfield!(points, name, oftype(value, NaN))
    poison_value!(points, name, value) = nothing

    f(x::Real) = [sin(x), x^2]
    f(x::AbstractVector) = [sin(x[1]) * x[2], x[1]^2, cos(x[2])]
    μ, Σ, μ2, Σ2 = [0.3, -0.2], [1.0 0.2; 0.2 0.5], [1.0, 1.0], [0.5 0.0; 0.0 2.0]
    for (method, scalar) in (
            (Unscented(), Unscented()), (ghcubature(4), ghcubature(4)), (GenUnscented(), GenUnscented()),
            (GenConstrainedUnscented([-5.0, -5.0], [5.0, 5.0]), GenConstrainedUnscented(-1.0, 2.0)),
        )
        ws = workspace(method, f, μ, Σ)
        expected = forward_statistics(method, f, μ2, Σ2)
        m, V, C = zeros(3), zeros(3, 3), zeros(2, 3)
        @test all(forward_statistics!(m, V, C, ws, method, f, μ2, Σ2) .≈ expected)
        @test all(forward_statistics!(m, V, C, poison!(ws), method, f, μ2, Σ2) .≈ expected)
        @test all(approximate!(m, V, poison!(ws), method, f, μ2, Σ2) .≈ expected[1:2])

        sws = workspace(scalar, sin, 0.1, 1.0)
        expected = approximate(scalar, sin, 0.4, 0.3)
        @test all(approximate!(poison!(sws), scalar, sin, 0.4, 0.3) .≈ expected)
        @test expectation!(poison!(sws), scalar, sin, 0.4, 0.3) ≈ expectation(scalar, sin, 0.4, 0.3)
    end

    gl = GaussLaguerreQuadrature(8)
    ws = workspace(gl, sqrt, 2.0, 1.0)
    @test expectation!(poison!(ws), gl, sqrt, 7.0, 0.3) ≈ expectation(gl, sqrt, 7.0, 0.3)
end

@testitem "the allocating forms evaluate f once per point" tags = [:approximations] begin
    using MessagePassingRulesApproximations

    function counting(f)
        calls = Ref(0)
        return (x -> (calls[] += 1; f(x))), calls
    end

    for (method, n) in ((Unscented(), 3), (ghcubature(7), 7))
        g, calls = counting(sin)
        approximate(method, g, (0.5,), (0.2,))
        @test calls[] == n
        g, calls = counting(sin)
        forward_statistics(method, g, 0.5, 0.2)
        @test calls[] == n
        g, calls = counting(x -> exp(-x^2))
        approximate_meancov(method, g, 0.5, 0.2)
        @test calls[] == n
    end
    # The value at x̂ and one evaluation with dual numbers, as before.
    g, calls = counting(sin)
    approximate(Linearization(), g, (0.5,))
    @test calls[] == 2
end

@testitem "a method with getpoints and getweights only" tags = [:approximations] begin
    using MessagePassingRulesApproximations, LinearAlgebra

    # A method of another package that defines only the lazy points and weights, as RxGPs' do:
    # the midpoint rule of two points, `m ± √v`, exact for degree three.
    struct TwoPoints <: AbstractApproximationMethod end
    MessagePassingRulesApproximations.getpoints(::TwoPoints, m::Real, v::Real) = (m - sqrt(v), m + sqrt(v))
    MessagePassingRulesApproximations.getweights(::TwoPoints, m::Real, v::Real) = (0.5, 0.5)
    MessagePassingRulesApproximations.getpoints(::TwoPoints, m::AbstractVector, P::AbstractMatrix) = (m - sqrt.(diag(P)), m + sqrt.(diag(P)))
    MessagePassingRulesApproximations.getweights(::TwoPoints, m::AbstractVector, P::AbstractMatrix) = (0.5, 0.5)

    @test all(approximate_meancov(TwoPoints(), x -> 1.0, 1.0, 4.0) .≈ (1.0, 4.0))
    @test expectation(TwoPoints(), x -> x^2, 1.0, 4.0) ≈ 5.0
    mean, cov = approximate_meancov(TwoPoints(), x -> 1.0, [1.0, 2.0], [4.0 0.0; 0.0 1.0])
    @test mean ≈ [1.0, 2.0] && cov ≈ [4.0 2.0; 2.0 1.0]
end
