using UniqueVectors: UniqueVector

export Blackbox
export impedance_matrices, admittance_matrices, scattering_matrices

struct Blackbox{T, U<:AbstractMatrix{<:Number}} <: AdmittanceModel{T}
    ω::Vector{Float64}
    Y::Vector{U}
    P::U
    Q::U
    ports::UniqueVector{T}
    function Blackbox{T, U}(ω::Vector{Float64}, Y::Vector{U}, P::U, Q::U,
            ports::UniqueVector{T}) where {T, U}
        if !isempty(Y)
            l = size(first(Y), 1)
            @assert size(P) == (l, length(ports))
            @assert size(Q) == (l, length(ports))
            @assert all(size(y) == (l, l) for y in Y)
        end
        @assert length(ω) == length(Y)
        return new(ω, Y, P, Q, ports)
    end
end

"""
    Blackbox(ω::Vector{<:Number}, Y::Vector{U}, P::U, Q::U,
        ports::AbstractVector{T}) where {T, U}
Create a `Blackbox`, which has an admittance matrix `Y(ω′)` sampled over a
discrete set of frequencies `ω` such that `Y[i] == Y(ω[i])`, for valid `i`.
The port names are of type `T`.
"""
function Blackbox(ω::Vector{<:Number}, Y::Vector{U}, P::U, Q::U,
        ports::AbstractVector{T}) where {T, U}
    @assert isreal(ω)
    return Blackbox{T, U}(ω, Y, P, Q, UniqueVector(ports))
end

get_Y(bbox::Blackbox) = bbox.Y

get_P(bbox::Blackbox) = bbox.P

get_Q(bbox::Blackbox) = bbox.Q

get_ports(bbox::Blackbox) = bbox.ports

function partial_copy(bbox::Blackbox{T, U};
        Y::Union{Vector{V}, Nothing} = nothing,
        P::Union{V, Nothing} = nothing,
        Q::Union{V, Nothing} = nothing,
        ports::Union{AbstractVector{W}, Nothing} = nothing) where {T, U, V, W}
    Y = isnothing(Y) ? get_Y(bbox) : Y
    P = isnothing(P) ? get_P(bbox) : P
    Q = isnothing(Q) ? get_Q(bbox) : Q
    ports = isnothing(ports) ? get_ports(bbox) : ports
    return Blackbox(bbox.ω, Y, P, Q, ports)
end

function compatible(bboxes::AbstractVector{<:Blackbox})
    if isempty(bboxes)
        return true
    end
    ω = bboxes[1].ω
    return all(ω == bbox.ω for bbox in bboxes[2:end])
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
        return inv.(impedance_matrices(bbox))
    end
end

"""
    scattering_matrices(bbox::Blackbox, Z0::AbstractVector{<:Real})

Find the scattering matrices that map incoming waves to outgoing waves on transmission
lines with characteristic impedance `Z0` for each `ω`.
"""
function scattering_matrices(bbox::Blackbox, Z0::AbstractVector{<:Number})
    @assert isreal(Z0)
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
    Blackbox(ω::AbstractVector{<:Number}, pso::PSOModel)

Create a Blackbox for a PSOModel where `ω` represents angular frequency.
"""
function Blackbox(ω::AbstractVector{<:Number}, pso::PSOModel)
    @assert isreal(ω)
    K, G, C = get_Y(pso)
    Y = [K ./ s + G + C .* s for s in im * ω]
    P = complex(float(get_P(pso)))
    Q = complex(float(get_Q(pso)))
    return Blackbox(ω, Y, P, Q, get_ports(pso))
end
