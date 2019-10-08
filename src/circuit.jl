using UniqueVectors: UniqueVector
using LinearAlgebra: issymmetric, diag, I
using SparseArrays: spzeros

export Circuit
export partial_copy, matrices
export get_inv_inductance, get_inductance
export get_conductance, get_resistance
export get_elastance, get_capacitance
export set_inv_inductance!, set_inductance!
export set_conductance!, set_resistance!
export set_capacitance!, set_elastance!

export SpanningTree, coordinate_matrix
export cascade, unite_vertices, cascade_and_unite

SparseMat = SparseMatrixCSC{Float64,Int}

struct Circuit{T}
    k::SparseMat # inverse inductances
    g::SparseMat # conductances
    c::SparseMat # capacitances
    vertices::UniqueVector{T}
    function Circuit{T}(k::SparseMat, g::SparseMat, c::SparseMat,
        vertices::UniqueVector{T}) where T
        for m in [k, g, c]
            @assert issymmetric(m)
            @assert size(m) == (length(vertices), length(vertices))
            @assert all(diag(m) .== 0)
            @assert all(m .>= 0)
        end
        return new{T}(k, g, c, vertices)
    end
end

Circuit(k::SparseMat, g::SparseMat, c::SparseMat, vertices::AbstractVector{T}) where T =
    Circuit{T}(k, g, c, UniqueVector(vertices))

Circuit(k::AbstractMatrix{<:Real}, g::AbstractMatrix{<:Real}, c::AbstractMatrix{<:Real},
    vertices::AbstractVector) = Circuit(sparse(1.0 * k), sparse(1.0 * g), sparse(1.0 * c),
    vertices)

function Circuit(vertices::AbstractVector{T}) where T
    z() = spzeros(length(vertices), length(vertices))
    return Circuit(z(), z(), z(), vertices)
end

function partial_copy(circ::Circuit{T};
   k::Union{SparseMat, Nothing}=nothing,
   g::Union{SparseMat, Nothing}=nothing,
   c::Union{SparseMat, Nothing}=nothing,
   vertices::Union{AbstractVector{U}, Nothing}=nothing) where {T, U}
   k = k == nothing ? circ.k : k
   g = g == nothing ? circ.g : g
   c = c == nothing ? circ.c : c
   vertices = vertices == nothing ? circ.vertices : vertices
   return Circuit(k, g, c, vertices)
end

# don't remove space between Base. and ==
function Base. ==(circ1::Circuit, circ2::Circuit)
    if typeof(circ1) != typeof(circ2)
        return false
    end
    return all([getfield(circ1, name) == getfield(circ2, name)
        for name in fieldnames(Circuit)])
end

function Base.isapprox(circ1::Circuit, circ2::Circuit)
    if typeof(circ1) != typeof(circ2)
        return false
    end
    return all([getfield(circ1, name) ≈ getfield(circ2, name)
        for name in fieldnames(Circuit)[1:end-1]])
end

matrices(c::Circuit) = [c.k, c.g, c.c]

function get_matrix_element(c::Circuit, matrix_name::Symbol, v0, v1)
    i0 = findfirst(isequal(v0), c.vertices)
    i1 = findfirst(isequal(v1), c.vertices)
    return getfield(c, matrix_name)[i0, i1]
end

get_inv_inductance(c::Circuit, v0, v1) = get_matrix_element(c, :k, v0, v1)
get_inductance(c::Circuit, v0, v1) = inv(get_inv_inductance(c, v0, v1))
get_conductance(c::Circuit, v0, v1) = get_matrix_element(c, :g, v0, v1)
get_resistance(c::Circuit, v0, v1) = inv(get_conductance(c, v0, v1))
get_capacitance(c::Circuit, v0, v1) = get_matrix_element(c, :c, v0, v1)
get_elastance(c::Circuit, v0, v1) = inv(get_capacitance(c, v0, v1))

function set_matrix_element!(c::Circuit, matrix_name::Symbol, v0, v1, value)
    @assert value >= zero(value)
    @assert !isinf(value)
    i0 = findfirst(isequal(v0), c.vertices)
    i1 = findfirst(isequal(v1), c.vertices)
    @assert i0 != i1
    matrix = getfield(c, matrix_name)
    matrix[i0, i1] = value
    matrix[i1, i0] = value
    return c
end

set_inv_inductance!(c::Circuit, v0, v1, value) =
    set_matrix_element!(c, :k, v0, v1, value)

set_inductance!(c::Circuit, v0, v1, value) =
    set_inv_inductance!(c, v0, v1, inv(value))

set_conductance!(c::Circuit, v0, v1, value) =
    set_matrix_element!(c, :g, v0, v1, value)

set_resistance!(c::Circuit, v0, v1, value) =
    set_conductance!(c, v0, v1, inv(value))

set_capacitance!(c::Circuit, v0, v1, value) =
    set_matrix_element!(c, :c, v0, v1, value)

set_elastance!(c::Circuit, v0, v1, value) =
    set_capacitance!(c, v0, v1, inv(value))

#######################################################
# spanning trees
#######################################################

struct SpanningTree{T}
    root::T
    edges::Vector{Tuple{T, T}}
end

# depth-1 spanning tree
function SpanningTree(vertices::AbstractVector{T}) where T
    @assert length(vertices) > 0
    root = vertices[1]
    edges = [(v, root) for v in vertices[2:end]]
    return SpanningTree{T}(root, edges)
end

# T matrix in section IIC with a given spanning tree.
# The direction of the edges passed in is ignored and enforced as being outwards
# from the root
function coordinate_matrix(c::Circuit{T}, tree::SpanningTree{T}) where T
    V = length(c.vertices)
    @assert tree.root in c.vertices
    @assert length(tree.edges) == V-1 # the correct number of edges for a spanning tree
    mapping = Dict(tree.root => spzeros(V-1)) # the function l in equation 2
    edge_set = Set(enumerate(tree.edges))
    while !isempty(edge_set)
        for e in edge_set
            (index, (v0, v1)) = e
            @assert !(v0 in keys(mapping) && v1 in keys(mapping)) # edges contains a loop
            if v0 in keys(mapping) || v1 in keys(mapping)
                delete!(edge_set, e)
                if v0 in keys(mapping) # enforces all edges are oriented away from the root
                    mapping[v1] = 1.0 * mapping[v0]
                    mapping[v1][index] += 1
                else
                    mapping[v0] = 1.0 * mapping[v1]
                    mapping[v0][index] += 1
                end
            end
        end
    end
    @assert length(mapping) == V # all vertices have been assigned a coordinate
    return hcat([mapping[v] for v in c.vertices]...)
end

# T matrix in section IIC for depth-1 spanning tree
coordinate_matrix(num_vertices::Int) = transpose([spzeros(1, num_vertices-1); I])
coordinate_matrix(c::Circuit) = coordinate_matrix(length(c.vertices))

#######################################################
# cascade and unite
#######################################################

function cascade(circs::AbstractVector{Circuit{T}}) where T
    circuit_matrices = [cat(m..., dims=(1,2))
        for m in zip([matrices(circ) for circ in circs]...)]
    vertices = vcat([circ.vertices for circ in circs]...)
    return Circuit(circuit_matrices..., vertices)
end

cascade(circs::Vararg{Circuit{T}}) where T = cascade(collect(circs))

function unite_vertices(circ::Circuit{T}, vertices::AbstractVector{T}) where T
    if length(vertices) <= 1
        return circ
    end
    n = length(circ.vertices)
    vertex_inds = [findfirst(isequal(v), circ.vertices) for v in vertices]
    keep_inds = filter(x -> !(x in vertex_inds[2:end]), 1:n)
    function unite_indices(mat::AbstractMatrix)
        m = copy(mat)
        ind1 = vertex_inds[1]
        for ind2 in vertex_inds[2:end]
            m[ind1, :] += mat[ind2, :]
            m[:, ind1] += mat[:, ind2]
        end
        m = m[keep_inds, keep_inds]
        for i in 1:size(m, 1)
            m[i, i] = 0
        end
        return m
    end
    circuit_matrices = [unite_indices(mat) for mat in matrices(circ)]
    return Circuit(circuit_matrices..., circ.vertices[keep_inds])
end

unite_vertices(circ::Circuit{T}, vertices::Vararg{T}) where T =
    unite_vertices(circ, collect(vertices))

# cascade all circuits and unite vertices with the same name
function cascade_and_unite(circs::AbstractVector{Circuit{T}}) where T
    @assert length(circs) >= 1
    if length(circs) == 1
        return circs[1]
    end
    # number the vertices so that the names are all distinct and then cascade
    vertex_number = 1
    function rename(circ)
        vertices = [(vertex_number + i - 1, vertex)
            for (i, vertex) in enumerate(circ.vertices)]
        vertex_number += length(vertices)
        return Circuit(matrices(circ)..., vertices)
    end
    circ = cascade(map(rename, circs))
    # merge all vertices with the same name
    original_vertices = vcat([c.vertices for c in circs]...)
    for vertex in unique(original_vertices)
        inds = findall([v[2] == vertex for v in circ.vertices])
        circ = unite_vertices(circ, circ.vertices[inds])
    end
    # remove numbering
    return Circuit(matrices(circ)..., [v[2] for v in circ.vertices])
end

cascade_and_unite(circs::Vararg{Circuit{T}}) where T = cascade_and_unite(collect(circs))
