using Test, AdmittanceModels, LinearAlgebra

@testset "TransmissionLine with terminations" begin
    Z0 = 50.0
    tline = TransmissionLine(["0", "1"], 1e8, Z0, 5e-3, δ=50e-6)
    ω = 2π * collect(range(4, stop=6, length=100)) * 1e9
    pso = PSOModel(tline)
    bbox0 = canonical_gauge(Blackbox(ω, pso))
    bbox1 = Blackbox(ω, tline)
    @test isapprox(admittance_matrices(bbox0), admittance_matrices(bbox1), atol=1e-5)
    @test !isapprox(admittance_matrices(bbox0), admittance_matrices(bbox1), atol=1e-6)
    # now terminate it with Z0
    resistor = ParallelComponent("1", 0, 1/Z0, 0)
    pso_casc = open_ports_except(cascade_and_unite(PSOModel.([tline, resistor])), "0")
    bb_casc = open_ports_except(cascade_and_unite(Blackbox.(Ref(ω), [tline, resistor])), "0")
    S = [x[1,1] for x in scattering_matrices(Blackbox(ω, pso_casc), [Z0])]
    @test all([abs(x) < 1e-4 for x in S])
    S = [x[1,1] for x in scattering_matrices(bb_casc, [Z0])]
    @test all([abs(x) < 1e-15 for x in S])
    # now terminate it with Z0/2
    z0 = Z0/2
    resistor = ParallelComponent("1", 0, 1/z0, 0)
    pso_casc = open_ports_except(cascade_and_unite(PSOModel.([tline, resistor])), "0")
    bb_casc = open_ports_except(cascade_and_unite(Blackbox.(Ref(ω), [tline, resistor])), "0")
    S = [x[1,1] for x in scattering_matrices(Blackbox(ω, pso_casc), [Z0])]
    correct_S = ((z0 - Z0)/(z0 + Z0)) * exp.(-2im * ω * tline.len/tline.propagation_speed)
    @test isapprox(S, correct_S, rtol=1e-4)
    S = [x[1,1] for x in scattering_matrices(bb_casc, [Z0])]
    @test isapprox(S, correct_S, rtol=1e-14)
end

@testset "reflection λ/4 resonator" begin
    Z0 = 50.0
    tline = TransmissionLine(["open", "coupler", "short"], 1e8, Z0, 5e-3, locations=[1e-3], δ=50e-6)
    capacitor = SeriesComponent("in", "coupler", 0, 0, 10e-15)
    p(x) = open_ports_except(short_ports(x, "short"), "in")
    ω = 2π * collect(range(4, stop=6, length=500)) * 1e9
    pso_casc = p(cascade_and_unite(PSOModel.([tline, capacitor])))
    bb_casc = p(cascade_and_unite(Blackbox.(Ref(ω), [tline, capacitor])))
    bbox0 = canonical_gauge(Blackbox(ω, pso_casc))
    bbox1 = bb_casc
    circ = cascade_and_unite(Circuit(tline, "tline"), Circuit(capacitor))
    g = AdmittanceModels.ground
    circ = unite_vertices(circ, g, "short")
    pso = PSOModel(circ, [("in", g)], ["in"])
    bbox2 = canonical_gauge(Blackbox(ω, pso))
    @test isapprox(admittance_matrices(bbox0), admittance_matrices(bbox1), atol=1e-3)
    @test isapprox(admittance_matrices(bbox0), admittance_matrices(bbox2), atol=1e-11)
    @test !isapprox(admittance_matrices(bbox0), admittance_matrices(bbox1), atol=1e-4)
end
