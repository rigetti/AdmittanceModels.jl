using LinearAlgebra: lu, I, ldiv!, rmul!, norm, qr, rank, nullspace
using SparseArrays: sparse, spzeros, SparseMatrixCSC, findnz
using Combinatorics: combinations
using MatrixNetworks: bfs, scomponents

export sparse_nullbasis, nullbasis

"""
    closest_permutation(mat::AbstractMatrix)

Given an `m × n` matrix `mat` where `m ≥ n`, find an assignment of rows to columns such
that each column is assigned a distinct row and the absolute values of the entries
corresponding to the assignment is as large as possible under lexicographical ordering.
Return a matrix of Bools where each column has a single true in it corresponding to its
assignment.
"""
function closest_permutation(mat::AbstractMatrix)
    nrows, ncols = size(mat)
    @assert nrows >= ncols
    ans = similar(BitArray, axes(mat)) # initialize answer as all false
    fill!(ans, false)

    ax2 = axes(mat, 2)

    # Do a first pass, ignoring collisions.
    colidxs = similar(ax2)
    for col in ax2
        @inbounds begin
            _, i = findmax(mat[:, col])
            ans[i, col] = true
            colidxs[col] = i
        end
    end

    # Prepare a BitArray `collisions` and mark collided columns in it.
    sp = sortperm(colidxs)
    sortedidxs = colidxs[sp]
    collisions = similar(BitArray, ax2)
    fill!(collisions, false)

    previndex = firstindex(sortedidxs)
    @inbounds for i in eachindex(sortedidxs)
        isequal(i, firstindex(sortedidxs)) && continue
        if isequal(sortedidxs[i], sortedidxs[previndex])
            collisions[sp[i]] = true
            collisions[sp[previndex]] = true
        end
        previndex = i
    end

    @inbounds stack = sp[collisions]
    isempty(stack) && return ans

    # reset columns in `ans` where collisions happened
    for col in stack
        @inbounds ans[:, col] .= false
    end

    # fall-back to single-threaded collision resolution on collided columns
    return _closest_permutation(mat, ans, stack)
end

function _closest_permutation(mat::AbstractMatrix, ans, stack)
    ax1 = axes(mat, 1)
    ax1l = length(ax1)
    possible_assignments = Vector{eltype(ax1)}() # row indices

    @inbounds while length(stack) > 0
        col = pop!(stack)
        resize!(possible_assignments, ax1l)
        possible_assignments .= axes(mat, 1)
        assigned = false
        while !assigned
            value, i = findmax(mat[possible_assignments, col])
            assignment = possible_assignments[i]
            collision = findfirst(ans[assignment, :])
            if collision == nothing
                ans[assignment, col] = true
                assigned = true
            elseif mat[assignment, collision] < value
                ans[assignment, col] = true
                assigned = true
                ans[assignment, collision] = false
                push!(stack, collision) # need to reassign collision
            else
                deleteat!(possible_assignments, i) # try somewhere else
            end
        end
    end
    return ans
end

"""
    match_vectors(port_vectors::AbstractMatrix{<:Number},
        eigenvectors::AbstractMatrix{<:Number})

For each column of `port_vectors` find a corresponding column of `eigenvectors` that has a
similar "shape". Match each `port_vector` to a different `eigenvector`.
"""
function match_vectors(port_vectors::AbstractMatrix{<:Number},
    eigenvectors::AbstractMatrix{<:Number})
    @assert size(port_vectors, 1) == size(eigenvectors, 1)
    @assert size(port_vectors, 2) <= size(eigenvectors, 2)
    mat = abs2.(eigenvectors' * port_vectors) # tall skinny matrix
    _, indices = findmax(closest_permutation(mat), dims=1)
    return [index[1] for index in vcat(indices...)]
end

"""
    inv_power_eigen(M::AbstractMatrix{<:Number};
        v0::AbstractVector{<:Number}=rand(Complex{real(eltype(M))}, size(M, 1)),
        λ0::Union{Number, Nothing}=nothing,
        pred::Function=(x,y) -> norm(x-y) < eps())

Use the inverse power method to find an eigenvalue and eigenvector.

# Arguments
- `v0::AbstractVector{<:Number}`: an initial value for the eigenvector. A random vector is
    used if `v0` is not given.
- `λ0::Union{Number, Nothing}`: an initial value for the eigenvalue. `v' * M * v` is used
    if `λ0` is not given.
- `pred::Function`: a function taking two successive eigenvalue estimates. The algorithm
    halts when `pred` returns true.
"""
function inv_power_eigen(M::AbstractMatrix{<:Number};
    v0::AbstractVector{<:Number}=rand(Complex{real(eltype(M))}, size(M, 1)),
    λ0::Union{Number, Nothing}=nothing,
    pred::Function=(x,y) -> norm(x-y) < eps())
    v = v0/norm(v0)
    l = length(v)
    @assert size(M) == (l, l)
    λ = (λ0 == nothing) ? v' * M * v : λ0
    done = false
    μ = λ
    A = lu(M - λ * I)
    while !done
        ldiv!(A, v)
        rmul!(v, 1/norm(v))
        μ = v' * M * v
        done = pred(λ, μ)::Bool
        λ = μ
    end
    return μ, v
end

#######################################################
# sparse nullbases
#######################################################

"""
    row_column_graph(mat::SparseMatrixCSC{<:Number, Int})

Let `mat` be a matrix with the following properties
    - for any two rows of `mat` there is at most one column on which both rows are nonzero
    - any column of `mat` has at most 2 nonzero values
The rows of `mat` form a simple graph where two rows have an edge if and only if they share
a nonzero column in `mat`. We add one additional vertex to this graph and for each column
with exactly one nonzero value, we add an edge between its nonzero row and the additional
vertex. In this graph every column with at least one nonzero value is represented by an
edge. Produce a matrix `A` where `A[i,j]` is the column shared by rows `i` and `j` if such a
column exists, and `0` otherwise. If `i == size(mat, 1) + 1`, then `A[i,j]` is chosen
arbitrarily from the columns whose unique nonzero value is in row `j`, or 0 if no such
columns exist. The case where `j == size(mat, 1) + 1` is analogous. `A` is an adjacency
matrix for the graph with extra information relating the graph to `mat`.
"""
function row_column_graph(mat::SparseMatrixCSC{<:Number, Int})
    num_vertices = size(mat, 1) + 1
    row_inds, col_inds, vals = Int[], Int[], Int[]
    matT = sparse(transpose(mat)) # SparseMatrixCSC row access is slow
    for (r1, r2) in combinations(1:(num_vertices-1), 2)
        inds = findall((matT[:, r1] .!= 0) .& (matT[:, r2] .!= 0))
        @assert length(inds) <= 1 # any two rows share at most 1 common nonzero column
        if length(inds) == 1
            c = inds[1]
            push!(row_inds, r1); push!(col_inds, r2); push!(vals, c)
            push!(row_inds, r2); push!(col_inds, r1); push!(vals, c)
        end
    end
    root = num_vertices
    connected_to_root = Set{Int}()
    for c in 1:size(mat, 2)
        inds = findall(mat[:, c] .!= 0)
        @assert length(inds) <= 2 # any column has at most 2 nonzero values
        if length(inds) == 1
            r = inds[1]
            if !(r in connected_to_root)
                push!(connected_to_root, r)
                push!(row_inds, root); push!(col_inds, r); push!(vals, c)
                push!(row_inds, r); push!(col_inds, root); push!(vals, c)
            end
        end
    end
    adj = sparse(row_inds, col_inds, vals, num_vertices, num_vertices)
    return adj
end


"""
    connected_row_column_graph(mat::SparseMatrixCSC{<:Number, Int})

Let `mat` be a matrix with the following properties
    - for any two rows of `mat` there is at most one column on which both rows are nonzero
    - any column of `mat` has at most 2 nonzero values
Produce a matrix `mat_connected` with the same nullspace as `mat` such that
`row_column_graph(mat_connected)` encodes a connected graph. Return `mat_connected` and
`row_column_graph(mat_connected)`.
"""
function connected_row_column_graph(mat::SparseMatrixCSC{<:Number, Int})
    adj = row_column_graph(mat)
    components = scomponents(adj)
    if components.number > 1 # adj is not connected
        root_component = components.map[end]
        remove_rows = Int[]
        zero_columns = Set{Int}()
        matT = sparse(transpose(mat)) # SparseMatrixCSC row access is slow
        for component in 1:components.number
            if component != root_component
                inds = findall(components.map .== component)
                num_dependent_rows = length(inds) - find_rank(matT[:, inds])
                if num_dependent_rows > 0 # some rows are redundant, remove them
                    append!(remove_rows, inds[1:num_dependent_rows])
                else # submat has full rank and trivial nullspace
                    append!(remove_rows, inds) # remove these rows
                    for i in inds # set columns in these rows to 0
                        push!(zero_columns, findall(matT[:, i] .!= 0)...)
                    end
                end
            end
        end
        num_zeros = length(zero_columns)
        zero_rows = sparse(1:num_zeros, collect(zero_columns), ones(num_zeros),
            num_zeros, size(mat, 2))
        remaining_inds = setdiff(1:size(mat, 1), remove_rows)
        mat = vcat(transpose(matT[:, remaining_inds]), zero_rows)
        adj = row_column_graph(mat) # adj is now connected
    end
    return mat, adj
end

find_rank(m::Matrix) = rank(m)
find_rank(m::SparseMatrixCSC) = rank(qr(1.0 * m))

"""
    bfs_tree(adj::AbstractMatrix{Int}, root::Int)

Produce the vertices and edges of the tree found by performing breadth first on the graph
with adjacency matrix `adj` starting from the vertex `root`. The vertices includes all
vertices of the tree besides the root.
"""
function bfs_tree(adj::AbstractMatrix{Int}, root::Int)
    dists, _, predecessors = bfs(adj, root)
    tree_vertices, tree_edges = Int[], Tuple{Int, Int}[]
    for i1 in reverse(sortperm(dists))
        i2 = predecessors[i1]
        if dists[i1] > 0 # exclude unreachables and the root
            push!(tree_vertices, i1)
            push!(tree_edges, tuple(sort([i1, i2])...))
        end
    end
    return tree_vertices, tree_edges
end

# Find the nullspace of mat by solving the equations corresponding to each of its rows. We
# solve row `tree_rows[i]`` by eliminating variable `tree_columns[i]`.
function sparse_nullbasis(mat::SparseMatrixCSC{<:Number, Int},
    tree_rows::AbstractVector{Int}, tree_columns::AbstractVector{Int})
    @assert length(tree_rows) == length(tree_columns) == size(mat, 1)
    non_tree_columns = setdiff(1:size(mat, 2), tree_columns)
    # first create mapping that preserves the non_tree_columns
    row_inds, col_inds, vals = Int[], Int[], Float64[]
    for (col_ind, row_ind) in enumerate(non_tree_columns)
        push!(row_inds, row_ind); push!(col_inds, col_ind); push!(vals, 1)
    end
    # work with transposes because SparseMatrixCSC row access is slow
    nullbasisT = sparse(col_inds, row_inds, vals, length(non_tree_columns), size(mat, 2))
    matT = sparse(transpose(mat))
    # now solve row constraints by walking the tree from the leaves to the root
    for (mat_row, mat_col) in zip(tree_rows, tree_columns)
        null_row = spzeros(size(nullbasisT, 1))
        for other_mat_col in findall(matT[:, mat_row] .!= 0)
            if other_mat_col != mat_col
                null_row += (-mat[mat_row, other_mat_col]/mat[mat_row, mat_col]) *
                    nullbasisT[:, other_mat_col]
            end
        end
        nullbasisT[:, mat_col] = null_row
    end
    return sparse(transpose(nullbasisT))
end

"""
    sparse_nullbasis(mat::SparseMatrixCSC{<:Number, Int})

Let `mat` be a matrix with the following properties
    - for any two rows of `mat` there is at most one column on which both rows are nonzero
    - any column of `mat` has at most 2 nonzero values
Produce a sparse matrix whose columns form a basis for the nullspace of `mat`.
"""
function sparse_nullbasis(mat::SparseMatrixCSC{<:Number, Int})
    mat, adj = connected_row_column_graph(mat)
    tree_rows, tree_edges = bfs_tree(adj, size(adj, 1))
    tree_columns = [adj[r1, r2] for (r1, r2) in tree_edges]
    return sparse_nullbasis(mat, tree_rows, tree_columns)
end

"""
    nullbasis(mat::AbstractMatrix{<:Number}; warn::Bool=true)

Return a `SparseMatrixCSC` whose columns form a basis for the null space of mat. If `mat`
satisfies the conditions of `sparse_nullbasis`, a sparse matrix is returned. Otherwise a
warning is issued and the `LinearAlgebra.nullspace` function is used.
"""
function nullbasis(mat::AbstractMatrix{<:Number}; warn::Bool=true)
    try
        return sparse_nullbasis(sparse(mat))
    catch AssertionError
        if warn
            @warn "sparse nullbasis not found"
        end
        return sparse(nullspace(collect(mat)))
    end
end
