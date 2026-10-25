@testitem "GenUnscented: the sigma points reproduce four central moments" tags = [:approximations] begin
    using MessagePassingRulesApproximations, LinearAlgebra
    import MessagePassingRulesApproximations: GenUnscented, GenUT, GenUnscentedTransform, sigma_points_weights

    @test GenUT === GenUnscented === GenUnscentedTransform
    @test approximation_name(GenUnscented()) == "GenUnscented" && approximation_short_name(GenUnscented()) == "GenUT"

    # A gamma with shape α and scale θ: S = 2αθ³ and K = 3α(α + 2)θ⁴.
    moments(α, θ) = (α * θ, α * θ^2, 2α * θ^3, 3α * (α + 2) * θ^4)
    for (α, θ) in ((2.0, 1.5), (0.5, 1.0), (10.0, 0.1))
        m, V, S, K = moments(α, θ)
        points, weights = sigma_points_weights(GenUnscented(), m, V, S, K)
        central(k) = sum(w * (x - m)^k for (w, x) in zip(weights, points))
        @test sum(weights) ≈ 1
        @test sum(weights .* points) ≈ m
        @test central(2) ≈ V && central(3) ≈ S && central(4) ≈ K
    end

    # A normal, without higher moments: the points are m and m ± √3 σ.
    points, weights = sigma_points_weights(GenUnscented(), 1.0, 4.0)
    @test all(points .≈ (1.0, 1.0 - 2sqrt(3), 1.0 + 2sqrt(3)))
    @test all(weights .≈ (2 / 3, 1 / 6, 1 / 6))

    # A vector input: the mean and covariance, and each component's third and fourth moments.
    μ, D = [1.0, -2.0, 0.5], Diagonal([2.0, 1.0, 0.5])
    S, K = [0.5, -0.2, 0.1], 3 .* diag(D) .^ 2 .+ 0.5
    points, weights = sigma_points_weights(GenUnscented(), μ, Matrix(D), S, K)
    @test length(points) == 7 && sum(weights) ≈ 1
    @test sum(weights .* points) ≈ μ
    @test sum(w * (x - μ) * (x - μ)' for (w, x) in zip(weights, points)) ≈ D
    for k in 1:3
        @test sum(w * (x[k] - μ[k])^3 for (w, x) in zip(weights, points)) ≈ S[k] atol = 1.0e-12
        @test sum(w * (x[k] - μ[k])^4 for (w, x) in zip(weights, points)) ≈ K[k]
    end

    # A correlated normal: the mean and covariance.
    Σ = [2.0 0.3 0.1; 0.3 1.0 0.2; 0.1 0.2 0.5]
    points, weights = sigma_points_weights(GenUnscented(), μ, Σ)
    @test sum(weights .* points) ≈ μ
    @test sum(w * (x - μ) * (x - μ)' for (w, x) in zip(weights, points)) ≈ Σ

    # Exact for a cubic of a normal, where the unscented transform is not: E[x³] = m³ + 3mv.
    m, v = 0.5, 0.3
    @test expectation(GenUnscented(), x -> x^3, m, v) ≈ m^3 + 3m * v
    @test first(approximate(GenUnscented(), x -> x^2, m, v)) ≈ m^2 + v
end

@testitem "GenUnscented: several inputs with their higher moments" tags = [:approximations] begin
    using MessagePassingRulesApproximations
    import MessagePassingRulesApproximations: GenUnscented

    means, covs, skews, kurts = (1.0, [0.0, 1.0]), (2.0, [1.0 0.0; 0.0 1.0]), (0.5, [0.0, 0.0]), (13.0, [3.0, 3.0])
    f(a, b) = a + sum(b)
    m, V = approximate(GenUnscented(), f, means, covs, skews, kurts)
    @test m ≈ 2.0 && V ≈ 4.0
    # E[(a - 1)³] is the first input's third moment.
    @test expectation(GenUnscented(), (a, b) -> (a - 1)^3, means, covs, skews, kurts) ≈ 0.5
    @test expectation(GenUnscented(), (a, b) -> (a - 1)^4, means, covs, skews, kurts) ≈ 13.0
    # One input in a tuple is that input.
    @test all(approximate(GenUnscented(), sin, (1.0,), (2.0,), (0.5,), (13.0,)) .≈ approximate(GenUnscented(), sin, 1.0, 2.0, 0.5, 13.0))
end

@testitem "GenConstrainedUnscented: the points stay within the bounds" tags = [:approximations] begin
    using MessagePassingRulesApproximations
    import MessagePassingRulesApproximations: GenConstrainedUnscented, GenCUT, GenUnscented, sigma_points_weights

    @test GenCUT === GenConstrainedUnscented

    # Bounds the transform would violate: the points move in, the mean and variance stay.
    points, weights = sigma_points_weights(GenConstrainedUnscented(0.0, Inf), 1.0, 4.0)
    @test all(>=(0), points)
    @test sum(weights .* points) ≈ 1.0 && sum(weights .* (points .- 1) .^ 2) ≈ 4.0

    # Bounds it respects: the unconstrained points.
    @test all(sigma_points_weights(GenConstrainedUnscented(-100.0, 100.0), 1.0, 4.0)[1] .≈ sigma_points_weights(GenUnscented(), 1.0, 4.0)[1])

    # A vector input, bounded in its first component only.
    μ, Σ = [1.0, 0.0], [4.0 1.0; 1.0 2.0]
    points, weights = sigma_points_weights(GenConstrainedUnscented([0.0, -Inf], [Inf, Inf]), μ, Σ)
    @test all(x -> x[1] >= 0, points)
    @test sum(weights .* points) ≈ μ
    @test sum(w * (x - μ) * (x - μ)' for (w, x) in zip(weights, points)) ≈ Σ

    @test_throws DomainError sigma_points_weights(GenConstrainedUnscented(0.0, 1.0), 2.0, 1.0)
    @test_throws ArgumentError GenConstrainedUnscented(1.0, 0.0)
    @test_throws ArgumentError GenConstrainedUnscented(0.0, 1.0; shrink = 1.5)
end
