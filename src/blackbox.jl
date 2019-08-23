using UniqueVectors: UniqueVector

export Blackbox
export impedance_matrices, admittance_matrices, scattering_matrices

"""
    Blackbox{T, U}(ω::Vector{Float64}, Y::Vector{U}, P::U, Q::U,
        ports::UniqueVector{T}) where {T, U}
    Blackbox(ω::Vector{<:Real}, Y::Vector{U}, P::U, Q::U,
        ports::AbstractVector{T}) where {T, U}

An admittance model with constant `P` and `Q` but with varying `Y`, indexed by a real
variable `ω`.
"""
struct Blackbox{T, U<:AbstractMatrix{<:Number}} <: AdmittanceModel{T}
    ω::Vector{Float64}
    Y::Vector{U}
    P::U
    Q::U
    ports::UniqueVector{T}
    function Blackbox{T, U}(ω::Vector{Float64}, Y::Vector{U}, P::U, Q::U,
        ports::UniqueVector{T}) where {T, U}
        if length(Y) > 0
            l = size(Y[1], 1)
            @assert size(P) == (l, length(ports))
            @assert size(Q) == (l, length(ports))
            @assert all([size(y) == (l, l) for y in Y])
        end
        @assert length(ω) == length(Y)
        return new{T, U}(ω, Y, P, Q, ports)
    end
end

Blackbox(ω::Vector{<:Real}, Y::Vector{U}, P::U, Q::U,
    ports::AbstractVector{T}) where {T, U} = Blackbox{T, U}(ω, Y, P, Q, UniqueVector(ports))

get_Y(bbox::Blackbox) = bbox.Y

get_P(bbox::Blackbox) = bbox.P

get_Q(bbox::Blackbox) = bbox.Q

get_ports(bbox::Blackbox) = bbox.ports

function partial_copy(bbox::Blackbox{T, U};
    Y::Union{Vector{V}, Nothing}=nothing,
    P::Union{V, Nothing}=nothing,
    Q::Union{V, Nothing}=nothing,
    ports::Union{AbstractVector{W}, Nothing}=nothing) where {T, U, V, W}
    Y = Y == nothing ? get_Y(bbox) : Y
    P = P == nothing ? get_P(bbox) : P
    Q = Q == nothing ? get_Q(bbox) : Q
    ports = ports == nothing ? get_ports(bbox) : ports
    return Blackbox(bbox.ω, Y, P, Q, ports)
end

function compatible(bboxes::AbstractVector{Blackbox{T, U}}) where {T, U}
    if length(bboxes) == 0
        return true
    end
    ω = bboxes[1].ω
    return all([ω == bbox.ω for bbox in bboxes[2:end]])
end

# TODO for sparse matrices use sparse linear solver instead of \ and /

function impedance_to_scattering(Z::AbstractMatrix{<:Number}, Z0::AbstractVector{<:Real})
    U = diagm(0 => Z0)
    return (Z + U) \ (Z - U)
end

function scattering_to_impedance(S::AbstractMatrix{<:Number}, Z0::AbstractVector{<:Real})
    U = diagm(0 => Z0)
    return U * (I + S) / (I - S)
end

function admittance_to_scattering(Y::AbstractMatrix{<:Number}, Z0::AbstractVector{<:Real})
    U = diagm(0 => Z0)
    return (I + Y * U) \ (I - Y * U)
end

function scattering_to_admittance(S::AbstractMatrix{<:Number}, Z0::AbstractVector{<:Real})
    V = diagm(0 => 1 ./ Z0)
    return (I - S) * ((I + S) \ V)
end

"""
    impedance_matrices(bbox::Blackbox)

Find the impedance matrices `Z` with `y = Zx` for each `ω`.
"""
impedance_matrices(bbox::Blackbox) = [transpose(bbox.Q) * (Y \ collect(bbox.P))
    for Y in bbox.Y]

"""
    admittance_matrices(bbox::Blackbox)

Find the admittance matrices `Y` with `Yy = x` for each `ω`.
"""
function admittance_matrices(bbox::Blackbox)
    if bbox.P == I # avoid unnecessary matrix inversion
        return map(collect, bbox.Y)
    else
        return [inv(Z) for Z in impedance_matrices(bbox)]
    end
end

"""
    scattering_matrices(bbox::Blackbox, Z0::AbstractVector{<:Real})

Find the scattering matrices that map incoming waves to outgoing waves on transmission
lines with characteristic impedance `Z0` for each `ω`.
"""
function scattering_matrices(bbox::Blackbox, Z0::AbstractVector{<:Real})
    if bbox.P == I # avoid unnecessary matrix inversion
        return [admittance_to_scattering(Y, Z0) for Y in admittance_matrices(bbox)]
    else
        return [impedance_to_scattering(Z, Z0) for Z in impedance_matrices(bbox)]
    end
end

function canonical_gauge(bbox::Blackbox)
    n = length(bbox.ports)
    Y = admittance_matrices(bbox)
    P = Matrix{eltype(eltype(Y))}(I, n, n)
    Q = Matrix{eltype(eltype(Y))}(I, n, n)
    return Blackbox(bbox.ω, Y, P, Q, bbox.ports)
end

"""
    Blackbox(ω::Vector{<:Real}, pso::PSOModel)

Create a Blackbox for a PSOModel where `ω` represents angular frequency.
"""
function Blackbox(ω::Vector{<:Real}, pso::PSOModel)
    K, G, C = get_Y(pso)
    Y = [K ./ s + G + C .* s for s in 1im * ω]
    P = get_P(pso) * (1.0 + 0im)
    Q = get_Q(pso) * (1.0 + 0im)
    return Blackbox(ω, Y, P, Q, get_ports(pso))
end
