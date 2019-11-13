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

export SpanningTree, coordinate_matrix, unite_vertices

SparseMat{T} = SparseMatrixCSC{T, Int}

"""
    struct Circuit{T, K<:Number, G<:Number, C<:Number}
Model of a circuit, containing matrix representations of the inverse inductance,
conductance, and capacitance between any pair of named vertices. All vertices are
named.

## Fields
- `k`: inverse inductance matrix
- `g`: conductance matrix
- `c`: capacitance matrix
- `vertices`: vector of uniquely named vertices

## Note
The matrices are not in the conventional form, i.e. `c` is not a Maxwell capacitance
matrix. Rather, `c[i,j]` is only the direct capacitance between vertices `i` and `j`.
`iszero(c[i,i])` is demanded for all `i`.
"""
struct Circuit{T, K<:Number, G<:Number, C<:Number}
    k::SparseMat{K} # inverse inductances
    g::SparseMat{G} # conductances
    c::SparseMat{C} # capacitances
    vertices::UniqueVector{T}
    function Circuit{T}(k::SparseMat, g::SparseMat, c::SparseMat,
            vertices::UniqueVector{T}) where {T}
        for m in (k, g, c)
            @assert isreal(m) && issymmetric(m)
            @assert size(m) == (length(vertices), length(vertices))
            @assert all(iszero.(diag(m)))
            @assert all(m .>= zero(eltype(m)))
        end
        kk, gg, cc = float(k), float(g), float(c)
        K, G, C = eltype(kk), eltype(gg), eltype(cc)
        return new{T,K,G,C}(kk, gg, cc, vertices)
    end
end

"""
    Circuit(k::AbstractMatrix, g::AbstractMatrix, c::AbstractMatrix,
        vertices::AbstractVector{T}) where {T}
    Circuit(vertices::AbstractVector)

Construct a circuit with named vertices, defaulting to zero inverse inductance,
conductance, and capacitance between the vertices if the matrices are not given.
"""
function Circuit(k::AbstractMatrix, g::AbstractMatrix, c::AbstractMatrix,
        vertices::AbstractVector{T}) where {T}
    return Circuit{T}(sparse(float(k)), sparse(float(g)), sparse(float(c)),
        UniqueVector(vertices))
end

function Circuit(vertices::AbstractVector)
    z() = spzeros(length(vertices), length(vertices))
    return Circuit(z(), z(), z(), vertices)
end

"""
    partial_copy(circ::Circuit;
       k::Union{AbstractMatrix, Nothing} = nothing,
       g::Union{AbstractMatrix, Nothing} = nothing,
       c::Union{AbstractMatrix, Nothing} = nothing,
       vertices::Union{AbstractVector, Nothing} = nothing)

Create a new `Circuit` with the same fields except those given as keyword arguments.
"""
function partial_copy(circ::Circuit;
       k::Union{AbstractMatrix, Nothing} = nothing,
       g::Union{AbstractMatrix, Nothing} = nothing,
       c::Union{AbstractMatrix, Nothing} = nothing,
       vertices::Union{AbstractVector, Nothing} = nothing)
   k = isnothing(k) ? circ.k : k
   g = isnothing(g) ? circ.g : g
   c = isnothing(c) ? circ.c : c
   vertices = isnothing(vertices) ? circ.vertices : vertices
   return Circuit(k, g, c, vertices)
end

"""
    ==(circ1::Circuit, circ2::Circuit)
Test whether two circuits have the same circuit components and named vertices
(in order).
"""
# don't remove space between Base. and ==
function Base. ==(circ1::Circuit, circ2::Circuit)
    return all(getfield(circ1, name) == getfield(circ2, name)
        for name in fieldnames(Circuit))
end

"""
    isapprox(circ1::Circuit, circ2::Circuit)
Tests for approximate equality of the circuits' `k`, `g`, `c` matrices.
"""
function Base.isapprox(circ1::Circuit, circ2::Circuit)
    return all(getfield(circ1, name) ≈ getfield(circ2, name)
        for name in fieldnames(Circuit)[1:end-1])
end

matrices(c::Circuit) = (c.k, c.g, c.c)

"""
    get_matrix_element(c::Circuit{T}, matrix_name::Symbol, v0::T, v1::T) where {T}
Returns a matrix element from one of the circuit matrices corresponding to the
edge between vertices with names `v0` and `v1`. `matrix_name` must be in
`(:k, :g, :c)`.
"""
function get_matrix_element(c::Circuit{T}, matrix_name::Symbol, v0::T, v1::T) where {T}
    @assert matrix_name in (:k, :g, :c)
    i0 = findfirst(isequal(v0), c.vertices)
    i1 = findfirst(isequal(v1), c.vertices)
    return getfield(c, matrix_name)[i0, i1]
end

"""
    get_inv_inductance(c::Circuit{T,K}, v0, v1) where {T,K}
Returns the inverse of the inductance on the edge between between vertices `v0` and `v1`.
"""
get_inv_inductance(c::Circuit{T, K}, v0, v1) where {T, K} =
    get_matrix_element(c, :k, v0, v1)::K

"""
    get_inductance(c::Circuit, v0, v1)
Returns the inductance on the edge between between vertices `v0` and `v1`.
"""
get_inductance(c::Circuit, v0, v1) = inv(get_inv_inductance(c, v0, v1))

"""
    get_conductance(c::Circuit{T,K,G}, v0, v1) where {T,K,G}
Returns the conductance on the edge between between vertices `v0` and `v1`.
"""
get_conductance(c::Circuit{T, K, G}, v0, v1) where {T, K, G} =
    get_matrix_element(c, :g, v0, v1)::G

"""
    get_resistance(c::Circuit, v0, v1)
Returns the resistance on the edge between between vertices `v0` and `v1`.
"""
get_resistance(c::Circuit, v0, v1) = inv(get_conductance(c, v0, v1))

"""
    get_capacitance(c::Circuit{T,K,G,C}, v0, v1) where {T,K,G,C}
Returns the capacitance on the edge between between vertices `v0` and `v1`.
"""
get_capacitance(c::Circuit{T,K,G,C}, v0, v1) where {T,K,G,C} =
    get_matrix_element(c, :c, v0, v1)::C

"""
    get_elastance(c::Circuit, v0, v1)
Returns the elastance on the edge between between vertices `v0` and `v1`.
"""
get_elastance(c::Circuit, v0, v1) = inv(get_capacitance(c, v0, v1))

"""
    set_matrix_element!(c::Circuit, matrix_name::Symbol, v0, v1, value)
Sets a matrix element in one of the circuit matrices corresponding to the
edge between vertices with names `v0` and `v1`. `matrix_name` must be in
`(:k, :g, :c)`.
"""
function set_matrix_element!(c::Circuit, matrix_name::Symbol, v0, v1, value)
    @assert matrix_name in (:k, :g, :c)
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

"""
    set_inv_inductance!(c::Circuit, v0, v1)
Sets the inverse of the inductance on the edge between between vertices `v0` and `v1`.
"""
set_inv_inductance!(c::Circuit, v0, v1, value) =
    set_matrix_element!(c, :k, v0, v1, value)

"""
    set_inductance!(c::Circuit, v0, v1)
Sets the inductance on the edge between between vertices `v0` and `v1`.
"""
set_inductance!(c::Circuit, v0, v1, value) =
    set_inv_inductance!(c, v0, v1, inv(value))

"""
    set_conductance!(c::Circuit, v0, v1)
Sets the conductance on the edge between between vertices `v0` and `v1`.
"""
set_conductance!(c::Circuit, v0, v1, value) =
    set_matrix_element!(c, :g, v0, v1, value)

"""
    set_resistance!(c::Circuit, v0, v1)
Sets the resistance on the edge between between vertices `v0` and `v1`.
"""
set_resistance!(c::Circuit, v0, v1, value) =
    set_conductance!(c, v0, v1, inv(value))

"""
    set_capacitance!(c::Circuit, v0, v1)
Sets the capacitance on the edge between between vertices `v0` and `v1`.
"""
set_capacitance!(c::Circuit, v0, v1, value) =
    set_matrix_element!(c, :c, v0, v1, value)

"""
    set_elastance!(c::Circuit, v0, v1)
Sets the elastance on the edge between between vertices `v0` and `v1`.
"""
set_elastance!(c::Circuit, v0, v1, value) =
    set_capacitance!(c, v0, v1, inv(value))

#######################################################
# spanning trees
#######################################################

"""
    struct SpanningTree{T}
A spanning tree of a graph with vertices of type `T`.

## Fields
- `root`: the root of the spanning tree
- `edges`: a vector of tuples `(vertex1, vertex2)` defining tree edges.
"""
struct SpanningTree{T}
    root::T
    edges::Vector{Tuple{T, T}}
end

"""
    SpanningTree(vertices::AbstractVector{T}) where T
Construct a depth-1 spanning tree from a vector of vertices of a complete graph,
with the first vertex taken as the root of the spanning tree.
"""
function SpanningTree(vertices::AbstractVector{T}) where T
    @assert !isempty(vertices)
    root = first(vertices)
    edges = [(v, root) for v in vertices[eachindex(vertices)[2:end]]]
    return SpanningTree{T}(root, edges)
end

"""
    coordinate_matrix(c::Circuit{T}, tree::SpanningTree{T}) where T
Returns the `T` matrix described in section IIC of arXiv:1810.11510 for a
circuit `c` with a given spanning `tree`. The direction of the edges passed in
is ignored and enforced as being outwards from the root.
"""
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

"""
    coordinate_matrix(num_vertices::Int)
Returns the `T` matrix described in section IIC of arXiv:1810.11510 for a
circuit with `num_vertices`, where the spanning tree is depth-1 and rooted at
the first vertex of the circuit.
"""
coordinate_matrix(num_vertices::Int) = [spzeros(num_vertices - 1) I]

"""
    coordinate_matrix(num_vertices::Int)
Returns the `T` matrix described in section IIC of arXiv:1810.11510 for a
circuit `c`, where the spanning tree is depth-1 and rooted at the first vertex
of the circuit.
"""
coordinate_matrix(c::Circuit) = coordinate_matrix(length(c.vertices))

#######################################################
# cascade and unite
#######################################################

"""
    cascade(circs::AbstractVector{<:Circuit})
    cascade(circs::Circuit...)
Cascade the circuits into one larger block diagonal circuit.
"""
function cascade(circs::AbstractVector{<:Circuit})
    circuit_matrices = [cat(m..., dims=(1,2))
        for m in zip([matrices(circ) for circ in circs]...)]
    vertices = vcat([circ.vertices for circ in circs]...)
    return Circuit(circuit_matrices..., vertices)
end
cascade(circs::Circuit...) = cascade(collect(circs))

"""
    unite_vertices(circ::Circuit{T}, vertices::AbstractVector{T}) where T
    unite_vertices(circ::Circuit{T}, vertices::T...) where T
Unite the given vertices of a circuit.
"""
function unite_vertices(circ::Circuit{T}, vertices::AbstractVector{T}) where T
    length(vertices) <= 1 && return circ
    n = length(circ.vertices)
    vertex_inds = [findfirst(isequal(v), circ.vertices) for v in vertices]
    keep_inds = filter(!in(vertex_inds[2:end]), 1:n)
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
unite_vertices(circ::Circuit{T}, vertices::T...) where T =
    unite_vertices(circ, collect(vertices))

"""
    connect(models::AbstractVector{<:Circuit})
    connect(models::Circuit...)
Cascade all models and unite ports with the same name. This results in a
combined circuit with common ports connected.
"""
function connect(circs::AbstractVector{<:Circuit})
    @assert !isempty(circs)
    if length(circs) == 1
        return first(circs)
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
connect(circs::Circuit...) = connect(collect(circs))
