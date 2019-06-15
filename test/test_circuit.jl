using Test, AdmittanceModels, LinearAlgebra
import .CircuitExample, .CircuitCascadeExample
const ce = CircuitExample
const cce = CircuitCascadeExample

@testset "Circuit" begin
    @test length(ce.circuit.vertices) == 5
    for m in matrices(ce.circuit)
        @test size(m) == (5,5)
        @test issymmetric(m)
    end
    @test get_inv_inductance(ce.circuit, ce.Q0_SIGNAL, ce.Q0_GROUND) == .1
    @test get_inv_inductance(ce.circuit, ce.Q0_GROUND, ce.Q0_SIGNAL) == .1
    @test get_conductance(ce.circuit, ce.Q0_SIGNAL, ce.Q0_GROUND) == 0
    @test get_conductance(ce.circuit, ce.Q0_GROUND, ce.Q0_SIGNAL) == 0
    @test get_capacitance(ce.circuit, ce.Q0_SIGNAL, ce.Q0_GROUND) == 100
    @test get_capacitance(ce.circuit, ce.Q0_GROUND, ce.Q0_SIGNAL) == 100
end

@testset "coordinate_matrix" begin
    @test coordinate_matrix(3) == [[0,0] [1,0] [0,1]]
    @test coordinate_matrix(ce.circuit) == coordinate_matrix(5)
    coordinate_matrix(ce.circuit, SpanningTree(ce.circuit.vertices)) == coordinate_matrix(5)
    correct_matrix = [[1, 1, 0, 0] [0, 1, 0, 0] [0, 0, 1, 1] [0, 0, 0, 1] [0, 0, 0, 0]]
    @test coordinate_matrix(ce.circuit, ce.tree) == correct_matrix
end

@testset "cascade and unite_vertices" begin
    circ = cascade(cce.circuit0, cce.circuit1, cce.circuit2)
    @test circ.vertices == 0:11
    l = length(circ.vertices)
    @test size(circ.c) == (l, l)
    united_circ = unite_vertices(circ, 0, 4, 8)
    united_circ = unite_vertices(united_circ, 1, 5, 9)
    @test united_circ.vertices == [0, 1, 2, 3, 6, 7, 10, 11]
    l = length(united_circ.vertices)
    @test size(united_circ.c) == (l, l)
    @test get_capacitance(united_circ, 2, 3) == get_capacitance(circ, 2, 3)
    @test get_inv_inductance(united_circ, 6, 1) == get_inv_inductance(circ, 6, 5)
    s = get_conductance(circ, 0, 1) + get_conductance(circ, 4, 5) + get_conductance(circ, 8, 9)
    @test get_conductance(united_circ, 0, 1) == s
    united_circ = unite_vertices(united_circ, 0, 1)
    s = get_capacitance(circ, 3, 0) + get_capacitance(circ, 3, 1)
    @test get_capacitance(united_circ, 3, 0) == s
end
