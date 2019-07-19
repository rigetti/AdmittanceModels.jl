using Test, AdmittanceModels
using LinearAlgebra: eigen, rank, nullspace
using SparseArrays: sparse, SparseMatrixCSC

@testset "closest_permutation" begin
    mat = [[1, 0, 2, 0, 1.5] [1, 0, 3, 0, 2] [1, 0, 1.3, 0, 1.4]]
    u = AdmittanceModels.closest_permutation(mat)
    @test vcat(sum(u, dims=1)...) == [1,1,1]
    _, indices = findmax(u, dims=1)
    @test [index[1] for index in vcat(indices...)] == [5,3,1]
end

@testset "inv_power_eigen" begin
    M = [1im 2.2 4; 3.1 0.2 3; 4 1 2im]
    F = eigen(M)
    for i in 1:size(M, 1)
        v0, λ0 = F.vectors[:,i], F.values[i]
        λ, v = AdmittanceModels.inv_power_eigen(M, v0=v0)
        @test λ ≈ λ0
        λ, v = AdmittanceModels.inv_power_eigen(M, λ0=λ0)
        @test λ ≈ λ0
        λ, v = AdmittanceModels.inv_power_eigen(M, v0=v0, λ0=λ0)
        @test λ ≈ λ0
    end
end

@testset "row_column_graph" begin
    rows = [[1.0, 2.0, -3.0,    0, 0, 0,  0],
        [    0,     0,  1.0, -4.0, 0, 0,  0],
        [    0.1,   0,    0, -0.3, 0, 0,  0],
        [    0,     0,    0,    0, 1, 1,  0],
        [    0,     0,    0,    0, 0, 1, -1]]
    mat = sparse(transpose(hcat(rows...)))
    correct_adj = zeros(Int, size(rows, 1)+1, size(rows, 1)+1)
    correct_adj[1, 2] = correct_adj[2, 1] = 3
    correct_adj[1, 3] = correct_adj[3, 1] = 1
    correct_adj[1, 6] = correct_adj[6, 1] = 2
    correct_adj[2, 3] = correct_adj[3, 2] = 4
    correct_adj[4, 5] = correct_adj[5, 4] = 6
    correct_adj[4, 6] = correct_adj[6, 4] = 5
    correct_adj[5, 6] = correct_adj[6, 5] = 7
    adj = AdmittanceModels.row_column_graph(mat)
    @test adj == correct_adj
end

@testset "sparse_nullbasis and nullbasis" begin
    # example with no redundancies or inconsistentices
    rows = [[1.0, 2.0, -3.0,    0, 0, 0,  0],
            [  0,   0,  1.0, -4.0, 0, 0,  0],
            [0.1,   0,    0, -0.3, 0, 0,  0],
            [  0,   0,    0,    0, 1, 1,  0],
            [  0,   0,    0,    0, 0, 1, -1]]
    mat = sparse(transpose(hcat(rows...)))
    null_basis = AdmittanceModels.sparse_nullbasis(mat)
    @test count(!iszero, mat * null_basis) == 0
    @test rank(collect(mat)) + rank(collect(null_basis)) == size(mat, 2) # rank nullity theorem
    @test count(!iszero, nullspace(collect(mat))) == 14 # dense
    @test count(!iszero, null_basis) == 7 # sparse
    @test AdmittanceModels.nullbasis(mat) == null_basis
    # example with redundancies and inconsistentices
    rows = [[1, -1,  0, 0,  0,  0],
            [1,  0, -1, 0,  0,  0],
            [0,  1,  1, 0,  0,  0],
            [0,  0,  0, 1, -1,  0],
            [0,  0,  0, 1,  0, -1],
            [0,  0,  0, 0,  1, -1]]
    mat = sparse(transpose(hcat(rows...)))
    null_basis = AdmittanceModels.sparse_nullbasis(mat)
    @test count(!iszero, mat * null_basis) == 0
    @test rank(collect(mat)) + rank(collect(null_basis)) == size(mat, 2) # rank nullity theorem
    @test AdmittanceModels.sparse_nullbasis(sparse(mat)) == null_basis
    @test AdmittanceModels.nullbasis(mat) == null_basis
    # example where sparse nullbasis algorithm does not apply
    mat = reshape(collect(1:9), 3,3)
    @test_throws AssertionError AdmittanceModels.sparse_nullbasis(sparse(mat))
    @test AdmittanceModels.nullbasis(mat, warn=false) == nullspace(mat)
end
