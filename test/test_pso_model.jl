using Test, AdmittanceModels, LinearAlgebra
const ce = CircuitExample
const ae = PSOModelExample
const cce = CircuitCascadeExample
const lrce = LRCExample
const hwe = HalfWaveExample
using SparseArrays: SparseMatrixCSC

@testset "PSOModel" begin
    @test length(ae.pso.ports) == 2
    for m in get_Y(ae.pso)
        @test size(m) == (4,4)
        @test issymmetric(m)
    end
    @test size(get_P(ae.pso)) == (4, 2)
end

@testset "coordinate_transform" begin
    coord_matrix_from = coordinate_matrix(ce.circuit)
    coord_matrix_to = coordinate_matrix(ce.circuit, ce.tree)
    transform = AdmittanceModels.coordinate_transform(coord_matrix_from, coord_matrix_to)
    @test AdmittanceModels.coordinate_transform(coord_matrix_from) == I
    @test AdmittanceModels.coordinate_transform(coord_matrix_to) != I
    @test transform * AdmittanceModels.coordinate_transform(coord_matrix_from) ==
        AdmittanceModels.coordinate_transform(coord_matrix_to)
end

@testset "Circuit to PSOModel" begin
    converted_pso = PSOModel(ce.circuit, [(ce.Q0_SIGNAL, ce.Q0_GROUND),
                                          (ce.Q1_SIGNAL, ce.Q1_GROUND)],
                             get_ports(ae.pso), ce.tree)
    @test ae.pso == converted_pso
end

@testset "cascade, unite_ports, open_ports, short_ports" begin
    # cascade and unite_ports
    circ = cascade(cce.circuit0, cce.circuit1, cce.circuit2)
    united_circ = unite_vertices(circ, 0, 4, 8)
    united_circ = unite_vertices(united_circ, 1, 5, 9)
    port_edges = [(united_circ.vertices[1], v) for v in united_circ.vertices[2:end]]
    port_names = united_circ.vertices[2:end]
    pso_from_circuit = PSOModel(united_circ, port_edges, port_names)
    pso0 = PSOModel(cce.circuit0, [(0,1), (0,2), (0, 3)], [1, 2, 3])
    pso1 = PSOModel(cce.circuit1, [(4,5), (4,6), (4,7)], [5, 6, 7])
    pso2 = PSOModel(cce.circuit2, [(8,9), (8,10), (8,11)], [9, 10, 11])
    pso = cascade(pso0, pso1, pso2)
    united_pso = unite_ports(pso, 1, 5, 9)
    transform = transpose(get_P(united_pso)/get_P(pso_from_circuit))
    pso_transform = apply_transform(pso_from_circuit, transform)
    @test pso_transform ≈ united_pso
    # open_ports
    port_edges = port_edges[1:2]
    port_names = port_names[1:2]
    pso_from_circuit = PSOModel(united_circ, port_edges, port_names)
    pso_transform = apply_transform(pso_from_circuit, transform)
    united_pso = open_ports(united_pso, get_ports(united_pso)[3:end])
    @test pso_transform ≈ united_pso
    # short_ports
    united_circ = unite_vertices(united_circ, 0, 2, 6)
    port_edges = [(united_circ.vertices[1], v) for v in united_circ.vertices[2:end]]
    port_names = united_circ.vertices[2:end]
    pso_from_circuit = PSOModel(united_circ, port_edges, port_names)
    united_pso = unite_ports(pso, 1, 5, 9)
    united_pso = short_ports(united_pso, 2, 6)
    transform = transpose(get_P(united_pso)/get_P(pso_from_circuit))
    pso_transform = apply_transform(pso_from_circuit, transform)
    @test pso_transform ≈ united_pso
end

@testset "lossy_modes_dense and lossless_modes_dense" begin
    eigenvalues, eigenvectors = lossy_modes_dense(lrce.pso)
    @test length(eigenvalues) == 1
    @test size(eigenvectors) == (1,1)
    decay_rate = -2 * real(eigenvalues[1])
    frequency = imag(eigenvalues[1])/(2π)
    @test frequency ≈ lrce.ω/(2π)
    @test decay_rate ≈ lrce.κ
    eigenvalues, eigenvectors = lossless_modes_dense(lrce.pso)
    @test length(eigenvalues) == 1
    @test size(eigenvectors) == (1,1)
    decay_rate = -2 * real(eigenvalues[1])
    frequency = imag(eigenvalues[1])/(2π)
    @test frequency ≈ 1/(sqrt(lrce.l * lrce.c) * 2π)
    @test decay_rate == 0
end

@testset "lossless_modes_sparse and tline resonator modes" begin
    ν, Z0 = 1.2e8, 50.0
    L = 5e-3
    δ = 50e-6
    resonator = TransmissionLine(["0", "1"], ν, Z0, L, δ=δ)
    pso = PSOModel(resonator)
    pso = short_ports(pso, pso.ports)
    for Y in get_Y(pso) # pso model is sparse
        @test count(!iszero, Y)/(size(Y,1) * size(Y, 2)) < .05
    end
    num_modes = 5
    values_s, vectors_s = lossless_modes_sparse(pso, num_modes=num_modes)
    values_d, vectors_d = lossless_modes_dense(pso)
    values_d, vectors_d = values_d[1:num_modes], vectors_d[:, 1:num_modes]
    @test isapprox(values_s, values_d, rtol=1e-6)
    for i in 1:5
        v_s, v_d = vectors_s[:, i], vectors_d[:, i]
        j = argmax(abs.(v_s))
        v_s /= v_s[j]
        v_d /= v_d[j]
        @test isapprox(v_s, v_d, rtol=1e-2)
    end
    @test isapprox(ν/(2*L) * (1:num_modes), imag.(values_s)/(2π), rtol=1e-3)
    @test isapprox(ν/(2*L) * (1:num_modes), imag.(values_d)/(2π), rtol=1e-3)
end

@testset "canonical_gauge" begin
    circ = cce.circuit0
    port_edges = [(1,2), (2,3)]
    port_names = [1, 2]
    pso = PSOModel(circ, port_edges, port_names)
    P = canonical_gauge(pso).P
    @test typeof(pso.P) <: SparseMatrixCSC
    @test typeof(P) <: Matrix
    m, n = size(P)
    @test P ≈ [Matrix(I, n, n); zeros(m-n, n)]

    m, n = size(hwe.pso.P)
    @test canonical_gauge(hwe.pso).P ≈ [Matrix(I, n, n); zeros(m-n, n)]
end
