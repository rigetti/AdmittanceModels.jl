using UniqueVectors: UniqueVector
using SparseArrays: spzeros, spdiagm
using LinearAlgebra: I, eigen, svd, diagm
using IterativeSolvers: lobpcg

export PSOModel
export lossless_modes_dense, lossy_modes_dense, lossless_modes_sparse

"""
    PSOModel{T, U}(K::U, G::U, C::U, P::U, Q::U, ports::UniqueVector{T}) where {T, U}
    PSOModel(K::U, G::U, C::U, P::U, Q::U, ports::AbstractVector{T}) where {T, U}

A Positive Second Order model, representing the equations `(K + G∂ₜ + C∂²ₜ)Φ = Px`,
`y = Qᵀ∂ₜΦ`.
"""
struct PSOModel{T, U<:AbstractMatrix{Float64}} <: AdmittanceModel{T}
    K::U
    G::U
    C::U
    P::U
    Q::U
    ports::UniqueVector{T}
    function PSOModel{T, U}(K::U, G::U, C::U, P::U, Q::U,
        ports::UniqueVector{T}) where {T, U}
        l = size(K, 1)
        @assert size(P) == (l, length(ports))
        @assert size(Q) == (l, length(ports))
        @assert all([size(y) == (l, l) for y in [K, G, C]])
        return new{T, U}(K, G, C, P, Q, ports)
    end
end

PSOModel(K::U, G::U, C::U, P::U, Q::U, ports::AbstractVector{T}) where {T, U} =
    PSOModel{T, U}(K, G, C, P, Q, UniqueVector(ports))

get_Y(pso::PSOModel) = [pso.K, pso.G, pso.C]

get_P(pso::PSOModel) = pso.P

get_Q(pso::PSOModel) = pso.Q

get_ports(pso::PSOModel) = pso.ports

function partial_copy(pso::PSOModel{T, U};
    Y::Union{Vector{V}, Nothing}=nothing,
    P::Union{V, Nothing}=nothing,
    Q::Union{V, Nothing}=nothing,
    ports::Union{AbstractVector{W}, Nothing}=nothing) where {T, U, V, W}
    Y = Y == nothing ? get_Y(pso) : Y
    P = P == nothing ? get_P(pso) : P
    Q = Q == nothing ? get_Q(pso) : Q
    ports = ports == nothing ? get_ports(pso) : ports
    return PSOModel(Y..., P, Q, ports)
end

compatible(psos::AbstractVector{PSOModel{T, U}}) where {T, U} = true

function canonical_gauge(pso::PSOModel)
    fullP = collect(pso.P)
    F = qr(fullP)
    (m, n) = size(fullP)
    transform = (F.Q * Matrix(I,m,m)) * [transpose(inv(F.R)) zeros(n, m-n);
        zeros(m-n, n) Matrix(I, m-n, m-n)]
    return apply_transform(pso, transform)
end

# NOTE we cannot write lossy_modes_sparse because KrylovKit.geneigsolve
# solves Av = λBv but requires that B is positive semidefinite and A is
# symmetric or Hermitian. For lossy modes, we can make B positive semidefinite
# and A nonsymmetric or A symmetric and B not positive semidefinite. If
# KrylovKit gains the ability to solve either type of eigenproblem we can
# write lossy_modes_sparse.

"""
    lossy_modes_dense(pso::PSOModel; min_freq::Real=0,
        max_freq::Union{Real, Nothing}=nothing, real_tol::Real=Inf, imag_tol::Real=Inf)

Use a dense eigenvalue solver to find the decaying modes of the PSOModel. Each mode consists
of an eigenvalue `λ` and a spatial distribution vector `v`. The frequency of the mode is
`imag(λ)/(2π)` and the decay rate is `-2*real(λ)/(2π)`. `min_freq` and `max_freq` give the
frequency range in which to find eigenvalues. Passivity is enforced using an inverse power
method solver, and `real_tol` and `imag_tol` are the tolerances used in that method to
determine convergence of the real and imaginary parts of the eigenvalue.
"""
function lossy_modes_dense(pso::PSOModel; min_freq::Real=0,
    max_freq::Union{Real, Nothing}=nothing, real_tol::Real=Inf, imag_tol::Real=Inf)
    K, G, C = get_Y(pso)
    Z = zero(K)
    M = collect([C Z; Z I]) \ collect([-G -K; I Z])
    M_scale = maximum(abs.(M))
    # for some reason eigen(Y \ X) seems to perform better than eigen(X, Y)
    values, vectors = eigen(M/M_scale)
    values *= M_scale
    freqs = imag.(values)/(2π)
    if max_freq == nothing
        max_freq = maximum(freqs)
    end
    inds = findall(max_freq .>= freqs .>= min_freq)
    values, vectors = values[inds], vectors[:, inds]
    # enforce passivity using inverse power method
    pred(x, y) = ((real(y) <= 0) && (abs(real(x-y)) < real_tol) &&
        (abs(imag(x-y)) < imag_tol))
    for i in 1:length(values)
        if real(values[i]) > 0
            λ, v = inv_power_eigen(M, v0=vectors[:, i], λ0=values[i], pred=pred)
            values[i] = λ
            vectors[:, i] = v
        end
    end
    # the second half of the eigenvector is the spatial part
    vectors = vectors[size(K, 1)+1:end, :]
    # normalize the columns
    return values, vectors./sqrt.(sum(abs2.(vectors), dims=1))
end

"""
    lossless_modes_dense(pso::PSOModel; min_freq::Real=0,
        max_freq::Union{Real, Nothing}=nothing)

Use a dense svd solver to find the modes of the PSOModel neglecting loss. Each mode consists
of an eigenvalue `λ` and an eigenvector `v`. The frequency of the mode is `imag(λ)/(2π)`
and the decay rate is `-2*real(λ)/(2π)`. `min_freq` and `max_freq` give the
frequency range in which to find eigenvalues.
"""
function lossless_modes_dense(pso::PSOModel; min_freq::Real=0,
    max_freq::Union{Real, Nothing}=nothing)
    K, _, C = get_Y(pso)
    K = .5 * (transpose(K) + K)
    C = .5 * (transpose(C) + C)
    sqrt_C, sqrt_K = sqrt(collect(C)), sqrt(collect(K))
    M = sqrt_C \ sqrt_K
    scale = maximum(abs.(M))
    U, S, Vt = svd(M/scale)
    freqs = S * scale/(2π)
    vectors = real.(sqrt_C \ U)
    if max_freq == nothing
        max_freq = maximum(freqs)
    end
    inds = findall(max_freq .>= freqs .>= min_freq)
    freqs, vectors = freqs[inds], vectors[:, inds]
    inds = sortperm(freqs)
    return freqs[inds] * 2π * 1im, vectors[:, inds]
end

"""
    lossless_modes_sparse(pso::PSOModel; num_modes=1, maxiter=200)

Use a sparse generalized eigenvalue solver to find the `num_modes` lowest frequency modes
of the PSOModel neglecting loss. Each mode consists of an eigenvalue `λ` and an eigenvector
`v`. The frequency of the mode is `imag(λ)/(2π)` and the decay rate is `-2*real(λ)/(2π)`.
"""
function lossless_modes_sparse(pso::PSOModel; num_modes=1, maxiter=200)
    K, _, C = get_Y(pso)
    K = (K + transpose(K))/2
    C = (C + transpose(C))/2
    K_scale, C_scale = maximum(abs.(K)), maximum(abs.(C))
    result = lobpcg(K/K_scale, C/C_scale, false, num_modes, maxiter=maxiter)
    values = result.λ * (K_scale/C_scale)
    vectors = result.X
    return real.(sqrt.(values)) * 1im, vectors
end

function port_matrix(circuit::Circuit, port_edges::Vector{<:Tuple})
    coord_matrix = coordinate_matrix(circuit)
    if length(port_edges) > 0
        port_indices = [(findfirst(isequal(p[1]), circuit.vertices),
                         findfirst(isequal(p[2]), circuit.vertices)) for p in port_edges]
        return hcat([coord_matrix[:, i1] - coord_matrix[:, i2]
            for (i1, i2) in port_indices]...)
    else
        return spzeros(size(coord_matrix, 1), 0)
    end
end

circuit_to_pso_matrix(m::AbstractMatrix{<:Real}) =
    (-m + spdiagm(0=>sum(m, dims=2)[:,1]))[2:end, 2:end]

"""
    PSOModel(circuit::Circuit, port_edges::Vector{<:Tuple},
        port_names::AbstractVector)

Create a PSOModel for a circuit with ports on given edges. The spanning tree where the
first vertex is chosen as ground and all other vertices are neighbors of ground is used.
"""
function PSOModel(circuit::Circuit, port_edges::Vector{<:Tuple},
        port_names::AbstractVector)
    Y = [circuit_to_pso_matrix(m) for m in matrices(circuit)]
    P = port_matrix(circuit, port_edges)
    return PSOModel(Y..., P, P, port_names)
end

coordinate_transform(m::AbstractMatrix{<:Real}) = m[:,2:end] .- m[:,1]

function coordinate_transform(coord_matrix_from::AbstractMatrix{<:Real},
    coord_matrix_to::AbstractMatrix{<:Real})
    return coordinate_transform(coord_matrix_to)/coordinate_transform(coord_matrix_from)
end

"""
    PSOModel(circuit::Circuit, port_edges::Vector{<:Tuple},
        port_names::AbstractVector, tree::SpanningTree)

Create a PSOModel for a circuit with ports on given edges and using the given spanning tree.
"""
function PSOModel(circuit::Circuit, port_edges::Vector{<:Tuple},
        port_names::AbstractVector, tree::SpanningTree)
    pso = PSOModel(circuit, port_edges, port_names)
    transform = transpose(coordinate_transform(coordinate_matrix(circuit, tree)))
    return apply_transform(pso, transform)
end
