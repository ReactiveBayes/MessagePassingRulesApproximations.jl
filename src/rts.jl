# A vector input, with a vector or a scalar output: the in-place form, into fresh outputs.
const RTSVectorArguments = Union{
    Tuple{AbstractVector, AbstractMatrix, AbstractMatrix, AbstractVector, AbstractMatrix, AbstractVector, AbstractMatrix},
    Tuple{Real, Real, AbstractVector, AbstractVector, AbstractMatrix, Real, Real},
}

"""
    smoothRTS(m_tilde, V_tilde, C_tilde, m_fw_in, V_fw_in, m_bw_out, V_bw_out) -> (m_in, V_in)

The Rauch–Tung–Striebel correction of an input's marginal through `out = g(in)`: given the
forward statistics of `g` and a backward message on `out`, the mean and covariance of the
marginal on `in`. It pairs with [`unscented_statistics`](@ref) or a linearisation, which provide
the forward statistics; a Delta node's joint marginal over its inputs is built on it.

# Arguments

- `m_tilde`, `V_tilde`: the mean and covariance of `g(in)` under the forward message on `in`;
- `C_tilde`: the cross-covariance between `in` and `g(in)` under that message;
- `m_fw_in`, `V_fw_in`: the mean and covariance of the forward message on `in`;
- `m_bw_out`, `V_bw_out`: the mean and covariance of the backward message on `out`.

Scalars or vectors and matrices, consistently.

# Returns

`(m_in, V_in)`, the mean and covariance of the marginal on `in`. Only `V_tilde + V_bw_out` is
inverted, so `V_tilde` may be singular: a linearisation of a `g` with more outputs than inputs makes
it so. When `V_tilde` is not finite, the result is the forward message's `(m_fw_in, V_fw_in)`
unchanged.

The in-place `smoothRTS!(m_in, V_in, ws, m_tilde, …)` writes the vector `m_in` and the matrix
`V_in`, for a vector input, and allocates nothing of its own; `ws` is
[`workspace`](@ref)`(smoothRTS, m_tilde, V_tilde, C_tilde, m_fw_in, V_fw_in, m_bw_out, V_bw_out)`.
For a scalar input, `smoothRTS` itself allocates nothing.

Petersen, Hoffmann and Rostalski (2018), *On approximate nonlinear Gaussian message passing on
factor graphs*, IEEE Statistical Signal Processing Workshop.
"""
function smoothRTS(
        m_tilde, V_tilde, C_tilde, m_fw_in, V_fw_in, m_bw_out, V_bw_out
    )
    arguments = (m_tilde, V_tilde, C_tilde, m_fw_in, V_fw_in, m_bw_out, V_bw_out)
    arguments isa RTSVectorArguments || return smoothRTS_generic(arguments...)
    ws = workspace(smoothRTS, arguments...)
    T = promote_type(eltype(m_fw_in), eltype(V_fw_in), eltype(C_tilde))
    return smoothRTS!(similar(m_fw_in, T), similar(V_fw_in, T), ws, arguments...)
end

# Any other shapes, scalars among them: the equations as they are.
function smoothRTS_generic(
        m_tilde, V_tilde, C_tilde, m_fw_in, V_fw_in, m_bw_out, V_bw_out
    )
    # A forward output covariance that is not finite leaves nothing to correct with.
    all(isfinite, V_tilde) || return (m_fw_in, V_fw_in)
    # The backward message conditions the joint normal of `in` and `out`, a Kalman update that
    # inverts only `V_tilde + V_bw_out`. `V_tilde` itself may be singular, as a linearisation
    # from fewer inputs than outputs makes it, and a zero cross-covariance, as a zero-covariance
    # input gives, returns the forward statistics.
    K = C_tilde * cholinv(V_tilde + V_bw_out)
    return (m_fw_in + K * (m_bw_out - m_tilde), V_fw_in - K * C_tilde')
end

struct RTSWorkspace{S, K, R} <: AbstractWorkspace
    # `V_tilde + V_bw_out`, factorised in place; `K' = (V_tilde + V_bw_out)⁻¹ C'`; and the
    # innovation `m_bw_out - m_tilde`. All `nothing` for a scalar output.
    S::S
    Kt::K
    r::R
end

Base.show(io::IO, ::RTSWorkspace) = print(io, "RTSWorkspace()")

workspace(::typeof(smoothRTS), m_tilde::Real, V_tilde::Real, C_tilde::AbstractVector, m_fw_in, V_fw_in, m_bw_out, V_bw_out) =
    RTSWorkspace(nothing, nothing, nothing)

function workspace(::typeof(smoothRTS), m_tilde::AbstractVector, V_tilde::AbstractMatrix, C_tilde::AbstractMatrix, m_fw_in, V_fw_in, m_bw_out, V_bw_out)
    T = float(promote_type(eltype(V_tilde), eltype(V_bw_out), eltype(C_tilde)))
    d_out, d_in = length(m_tilde), size(C_tilde, 1)
    return RTSWorkspace(Matrix{T}(undef, d_out, d_out), Matrix{T}(undef, d_out, d_in), Vector{T}(undef, d_out))
end

# A scalar output: `K = C / (V_tilde + V_bw_out)`, a vector.
function smoothRTS!(m_in::AbstractVector, V_in::AbstractMatrix, ::RTSWorkspace{Nothing}, m_tilde::Real, V_tilde::Real, C_tilde::AbstractVector, m_fw_in::AbstractVector, V_fw_in::AbstractMatrix, m_bw_out::Real, V_bw_out::Real)
    copyto!(m_in, m_fw_in)
    copyto!(V_in, V_fw_in)
    isfinite(V_tilde) || return (m_in, V_in)
    precision = cholinv(V_tilde + V_bw_out)
    innovation = m_bw_out - m_tilde
    for k in eachindex(m_in)
        m_in[k] += (C_tilde[k] * precision) * innovation
    end
    for j in eachindex(C_tilde), i in eachindex(C_tilde)
        V_in[i, j] -= (C_tilde[i] * precision) * C_tilde[j]
    end
    return (m_in, V_in)
end

function smoothRTS!(m_in::AbstractVector, V_in::AbstractMatrix, ws::RTSWorkspace, m_tilde::AbstractVector, V_tilde::AbstractMatrix, C_tilde::AbstractMatrix, m_fw_in::AbstractVector, V_fw_in::AbstractMatrix, m_bw_out::AbstractVector, V_bw_out::AbstractMatrix)
    copyto!(m_in, m_fw_in)
    copyto!(V_in, V_fw_in)
    all(isfinite, V_tilde) || return (m_in, V_in)
    S, Kt, r = ws.S, ws.Kt, ws.r
    S .= V_tilde .+ V_bw_out
    factor = fastcholesky!(S)
    for j in axes(C_tilde, 1), i in axes(C_tilde, 2)
        Kt[i, j] = C_tilde[j, i]
    end
    ldiv!(factor, Kt)
    r .= m_bw_out .- m_tilde
    mul!(m_in, Kt', r, true, true)
    mul!(V_in, Kt', C_tilde', -1, true)
    return (m_in, V_in)
end

"""
    smoothRTS!(m_in, V_in, ws, m_tilde, V_tilde, C_tilde, m_fw_in, V_fw_in, m_bw_out, V_bw_out) -> (m_in, V_in)

The in-place form of [`smoothRTS`](@ref), for a vector input: the marginal's mean and covariance
written into `m_in` and `V_in`, with `ws` from
[`workspace`](@ref)`(smoothRTS, m_tilde, V_tilde, C_tilde, m_fw_in, V_fw_in, m_bw_out, V_bw_out)`.
It allocates nothing of its own.
"""
smoothRTS!
