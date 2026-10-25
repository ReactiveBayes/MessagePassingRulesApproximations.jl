@testitem "Gauss–Laguerre: exact gamma moments, for small and large shapes" tags = [:approximations] begin
    using MessagePassingRulesApproximations

    # E[xᵏ] = θᵏ α (α + 1) ⋯ (α + k - 1) under a gamma with shape α and scale θ.
    raw(α, θ, k) = θ^k * prod(α + j for j in 0:(k - 1); init = 1.0)
    for α in (0.5, 1.0, 5.0, 50.0), θ in (0.7, 2.0)
        gl = GaussLaguerreQuadrature(6)
        for k in 0:11   # exact up to degree 2p - 1
            @test expectation(gl, x -> x^k, α, θ) ≈ raw(α, θ, k) rtol = 1.0e-10
        end
        m, V = approximate(gl, identity, α, θ)
        @test m ≈ α * θ && V ≈ α * θ^2
    end

    # An exponential is a gamma of shape one.
    @test expectation(glquadrature(4), x -> x, 1.0, 1 / 3) ≈ 1 / 3

    # The nodes are the eigenvalues of the Jacobi matrix, as LAPACK computes them, and the rule
    # is exact up to degree 2p - 1, for shapes from small to large.
    using LinearAlgebra
    import MessagePassingRulesApproximations: laguerre_nodes!, normalise_logweights!
    for p in (1, 3, 10, 32, 64), α in (0.05, 0.3, 1.0, 7.5, 60.0, 300.0)
        a = α - 1
        nodes, logweights, work = zeros(p), zeros(p), zeros(p)
        @test laguerre_nodes!(nodes, logweights, work, a)
        @test nodes ≈ eigvals(SymTridiagonal([2i - 1 + a for i in 1:p], [sqrt(i * (i + a)) for i in 1:(p - 1)])) rtol = 1.0e-11
        weights = exp.(normalise_logweights!(logweights))
        for k in 0:min(2p - 1, 10)   # E[(t / α)ᵏ] = ∏ⱼ (a + j) / α
            @test sum(weights .* (nodes ./ α) .^ k) ≈ prod((a + j) / α for j in 1:k; init = 1.0) rtol = 1.0e-10
        end
    end

    @test approximation_name(GaussLaguerreQuadrature(5)) == "GaussLaguerre(5)"
    @test approximation_short_name(glquadrature(5)) == "GL5"
    @test_throws DomainError expectation(glquadrature(5), identity, -1.0, 1.0)
    @test_throws DomainError expectation(glquadrature(5), identity, 1.0, 0.0)
    @test_throws ArgumentError GaussLaguerreQuadrature(0)
end

@testitem "Gauss–Laguerre: the moments of a reweighted gamma" tags = [:approximations] begin
    using MessagePassingRulesApproximations

    # A gamma reweighted by e^(-βx) is a gamma with rate 1/θ + β: shape α, scale 1/(1/θ + β).
    α, θ, β = 3.0, 2.0, 0.5
    scale = 1 / (1 / θ + β)
    m, v = approximate_meancov(glquadrature(20), x -> exp(-β * x), α, θ)
    @test m ≈ α * scale && v ≈ α * scale^2
    m, v = approximate_meancov(glquadrature(20), LogIntegrand(x -> -β * x), α, θ)
    @test m ≈ α * scale && v ≈ α * scale^2
end
