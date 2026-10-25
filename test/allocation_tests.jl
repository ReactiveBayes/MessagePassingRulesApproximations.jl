# The in-place operations allocate nothing of their own: each is measured inside a function,
# after a warm-up, with integrands that allocate nothing themselves (scalar functions, and
# functions that return a static vector).

@testmodule Allocations begin
    using StaticArrays, LinearAlgebra

    # The bytes one call allocates, its arguments passed through a function barrier.
    function allocations(op::O, args...) where {O}
        op(args...)
        op(args...)
        return @allocated op(args...)
    end

    square(x) = x^2
    negsquare(x) = -x^2
    sumsquares(x) = sum(abs2, x)
    likelihood(x) = exp(-sum(abs2, x) / 2)
    two_values(x::Real) = SVector(x^2, sin(x))
    three_values(x::AbstractVector) = SVector(x[1]^2, x[1] * x[2], sin(x[2]))
    of_two(x, y) = x^2 + sum(y)
    two_of_two(x, y) = SVector(x * y[1], y[2] - x)

    # A covariance of dimension `d`.
    covariance(d) = Matrix(1.0I, d, d) + 0.1 * ones(d, d)
end

@testitem "allocations: point-set methods, scalar inputs" tags = [:performance] setup = [Allocations] begin
    using MessagePassingRulesApproximations
    import MessagePassingRulesApproximations: workspace, GenUnscented, GenConstrainedUnscented
    using .Allocations: allocations, square, negsquare, two_values

    methods = (Unscented(), ghcubature(7), GenUnscented(), GenConstrainedUnscented(-3.0, 3.0))
    for method in methods
        ws = workspace(method, square, 0.3, 0.7)
        @test allocations(approximate!, ws, method, square, 0.5, 0.2) == 0
        @test allocations(forward_statistics!, ws, method, square, 0.5, 0.2) == 0
        @test allocations(expectation!, ws, method, square, 0.5, 0.2) == 0
        @test allocations(approximate_meancov!, ws, method, square, 0.5, 0.2) == 0
        @test @inferred(approximate!(ws, method, square, 0.5, 0.2)) isa Tuple{Float64, Float64}

        logws = workspace(method, LogIntegrand(negsquare), 0.3, 0.7)
        @test allocations(expectation!, logws, method, LogIntegrand(negsquare), 0.5, 0.2) == 0
        @test allocations(approximate_meancov!, logws, method, LogIntegrand(negsquare), 0.5, 0.2) == 0

        vws = workspace(method, two_values, 0.3, 0.7)
        m, V, C = zeros(2), zeros(2, 2), zeros(1, 2)
        @test allocations(approximate!, m, V, vws, method, two_values, 0.5, 0.2) == 0
        @test allocations(forward_statistics!, m, V, C, vws, method, two_values, 0.5, 0.2) == 0
        @test allocations(expectation!, m, vws, method, two_values, 0.5, 0.2) == 0
    end

    # A gamma, by Gauss–Laguerre, for any shape: the nodes are computed in place.
    gl = GaussLaguerreQuadrature(12)
    ws = workspace(gl, square, 2.0, 1.5)
    @test allocations(approximate!, ws, gl, square, 0.7, 3.0) == 0
    @test allocations(expectation!, ws, gl, square, 40.0, 0.2) == 0
    @test allocations(approximate_meancov!, ws, gl, square, 5.0, 0.2) == 0
    logws = workspace(gl, LogIntegrand(negsquare), 2.0, 1.5)
    @test allocations(approximate_meancov!, logws, gl, LogIntegrand(negsquare), 5.0, 0.2) == 0
end

@testitem "allocations: point-set methods, vector and several inputs" tags = [:performance] setup = [Allocations] begin
    using MessagePassingRulesApproximations
    import MessagePassingRulesApproximations: workspace, GenUnscented, GenConstrainedUnscented
    using .Allocations: allocations, sumsquares, likelihood, three_values, of_two, two_of_two, covariance

    for d in (1, 3, 6)
        methods = (Unscented(), ghcubature(3), GenUnscented(), GenConstrainedUnscented(fill(-10.0, d), fill(10.0, d)))
        μ, Σ = [0.1 * k for k in 1:d], covariance(d)
        for method in methods
            ws = workspace(method, sumsquares, μ, Σ)
            C = zeros(d)
            @test allocations(approximate!, ws, method, sumsquares, μ, Σ) == 0
            @test allocations(forward_statistics!, C, ws, method, sumsquares, μ, Σ) == 0
            @test allocations(expectation!, ws, method, sumsquares, μ, Σ) == 0
            mws = workspace(method, likelihood, μ, Σ)
            m, P = zeros(d), zeros(d, d)
            @test allocations(approximate_meancov!, m, P, mws, method, likelihood, μ, Σ) == 0
            if d >= 2
                vws = workspace(method, three_values, μ, Σ)
                m3, V3, C3 = zeros(3), zeros(3, 3), zeros(d, 3)
                @test allocations(forward_statistics!, m3, V3, C3, vws, method, three_values, μ, Σ) == 0
            end
        end
    end

    # Two inputs, a scalar and a vector, as one joint input.
    means, covs = (0.5, [1.0, 2.0]), (0.3, [1.0 0.2; 0.2 2.0])
    for method in (Unscented(), ghcubature(3), GenUnscented())
        ws = workspace(method, of_two, means, covs)
        C = zeros(3)
        @test allocations(forward_statistics!, C, ws, method, of_two, means, covs) == 0
        vws = workspace(method, two_of_two, means, covs)
        m, V, Cv = zeros(2), zeros(2, 2), zeros(3, 2)
        @test allocations(forward_statistics!, m, V, Cv, vws, method, two_of_two, means, covs) == 0
    end

    # The higher moments of several inputs.
    skews, kurts = (0.1, [0.0, 0.2]), (0.5, [3.0, 13.0])
    ws = workspace(GenUnscented(), of_two, means, covs, skews, kurts)
    @test allocations(approximate!, ws, GenUnscented(), of_two, means, covs, skews, kurts) == 0
end

@testitem "allocations: Linearization and smoothRTS" tags = [:performance] setup = [Allocations] begin
    using MessagePassingRulesApproximations, LinearAlgebra
    import MessagePassingRulesApproximations: workspace
    using .Allocations: allocations, square, sumsquares, two_values, three_values, of_two, two_of_two

    lin = Linearization()
    ws = workspace(lin, square, (0.3,))
    @test allocations(approximate!, ws, lin, square, (0.5,)) == 0
    vws = workspace(lin, two_values, (0.3,))
    @test allocations(approximate!, zeros(2), zeros(2), vws, lin, two_values, (0.5,)) == 0
    for d in (1, 3, 6)
        x = [0.2 * k for k in 1:d]
        gws = workspace(lin, sumsquares, (x,))
        @test allocations(approximate!, zeros(d), gws, lin, sumsquares, (x,)) == 0
    end
    x = [0.1, 0.2, 0.3]
    jws = workspace(lin, three_values, (x,))
    A, b = zeros(3, 3), zeros(3)
    @test allocations(approximate!, A, b, jws, lin, three_values, (x,)) == 0

    means, covs = (0.5, [1.0, 2.0]), (0.3, [1.0 0.2; 0.2 2.0])
    mws = workspace(lin, of_two, means)
    @test allocations(approximate!, zeros(3), mws, lin, of_two, means) == 0
    fws = workspace(lin, two_of_two, means, covs)
    m, V, C = zeros(2), zeros(2, 2), zeros(3, 2)
    @test allocations(forward_statistics!, m, V, C, fws, lin, two_of_two, means, covs) == 0
    sws = workspace(lin, of_two, means, covs)
    @test allocations(forward_statistics!, zeros(3), sws, lin, of_two, means, covs) == 0
    @test @inferred(forward_statistics!(m, V, C, fws, lin, two_of_two, means, covs)) isa Tuple{Vector{Float64}, Matrix{Float64}, Matrix{Float64}}

    # The smoother, for a vector output and for a scalar one.
    m_tilde, V_tilde, C_tilde = [0.5, -0.5], [2.0 0.3; 0.3 1.0], [1.0 0.5; 0.2 0.1; 0.0 0.3]
    m_fw, V_fw, m_bw, V_bw = [0.1, 0.2, 0.3], Matrix(2.0I, 3, 3), [1.0, 0.0], Matrix(1.0I, 2, 2)
    rws = workspace(smoothRTS, m_tilde, V_tilde, C_tilde, m_fw, V_fw, m_bw, V_bw)
    m_in, V_in = zeros(3), zeros(3, 3)
    @test allocations(smoothRTS!, m_in, V_in, rws, m_tilde, V_tilde, C_tilde, m_fw, V_fw, m_bw, V_bw) == 0
    rws = workspace(smoothRTS, 0.5, 2.0, [1.0, 0.5, 0.2], m_fw, V_fw, 1.0, 1.0)
    @test allocations(smoothRTS!, m_in, V_in, rws, 0.5, 2.0, [1.0, 0.5, 0.2], m_fw, V_fw, 1.0, 1.0) == 0
end
