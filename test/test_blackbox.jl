using Test, AdmittanceModels, LinearAlgebra
import .PSOModelExample, .LRCExample, .CircuitCascadeExample
const ae = PSOModelExample
const lrce = LRCExample
const cce = CircuitCascadeExample

@testset "PSOModel to Blackbox" begin
    # test 1
    ω = collect(range(4, stop=6, length=1000)) * 1e9 * 2π
    bbox = Blackbox(ω, ae.pso)
    @test get_ports(bbox) == get_ports(ae.pso)
    @test length(get_Y(bbox)) == length(ω)
    @test size(get_Y(bbox)[1]) == size(ae.pso.K)
    bbox = canonical_gauge(bbox)
    @test size(get_Y(bbox)[1]) == (2,2)
    # test 2
    l, r, c, ω₀ = lrce.l, lrce.r, lrce.c, lrce.ω
    ω = collect(range(.8, stop=1.2, length=1000)) * ω₀
    bbox = Blackbox(ω, lrce.pso)
    z = 1 ./(1 ./(l * 1im * ω) .+ 1/r .+ c * 1im * ω)
    @test length(get_Y(bbox)) == length(ω)
    impedance = impedance_matrices(bbox)
    @test all([size(x) == (1,1) for x in impedance])
    @test [x[1,1] for x in impedance] ≈ z
end

@testset "cascade, unite_ports, open_ports, short_ports" begin
    # cascade
    pso0 = PSOModel(cce.circuit0, [(0,1), (0,2)], [1, 2])
    pso1 = PSOModel(cce.circuit1, [(4,5), (4,6)], [5, 6])
    pso2 = PSOModel(cce.circuit2, [(8,9), (8,10)], [9, 10])
    pso = cascade(pso0, pso1, pso2)
    ω = collect(range(.1, stop=100, length=1000)) * 2π
    bbox_from_pso = Blackbox(ω, pso)
    @test get_ports(bbox_from_pso) == get_ports(pso)
    bboxes0 = [Blackbox(ω, a) for a in [pso0, pso1, pso2]]
    bbox0 = cascade(bboxes0)
    @test bbox0 ≈ bbox_from_pso
    bboxes1 = [canonical_gauge(b) for b in bboxes0]
    bbox1 = cascade(bboxes1)
    @test bbox1 ≈ canonical_gauge(bbox_from_pso)
    # unite_ports
    united_pso = unite_ports(pso, 1, 5, 9)
    bbox_from_pso = Blackbox(ω, united_pso)
    @test get_ports(bbox_from_pso) == get_ports(united_pso)
    united_bbox0 = unite_ports(bbox0, 1, 5, 9)
    @test united_bbox0 ≈ bbox_from_pso
    united_bbox1 = unite_ports(bbox1, 1, 5, 9)
    @test canonical_gauge(united_bbox1) ≈ canonical_gauge(bbox_from_pso)
    # open_ports
    open_pso = open_ports(united_pso, 2, 6)
    bbox_from_pso = Blackbox(ω, open_pso)
    open_bbox = open_ports(united_bbox1, 2, 6)
    @test canonical_gauge(open_bbox) ≈ canonical_gauge(bbox_from_pso)
    # short_ports
    short_pso = short_ports(united_pso, 2, 6)
    bbox_from_pso = Blackbox(ω, short_pso)
    short_bbox = short_ports(united_bbox1, 2, 6)
    @test canonical_gauge(short_bbox) ≈ canonical_gauge(bbox_from_pso)
end

# Pozar p 192
@testset "transfer function conversions" begin
    a, b, c, d = 1.0, 2.0, 3.0, 4.0
    Z0 = 10.0
    Y = [[d, -1] [b * c - a * d, a]] ./ b
    Z = [[a, 1] [a * d - b * c, d]] ./ c
    denom = a + b/Z0 + c * Z0 + d
    S = [[a + b/Z0 - c * Z0 - d, 2] [2 * (a * d - b * c), -a + b/Z0 - c * Z0 + d]] ./ denom
    @test inv(Y) ≈ Z
    Z0s = [Z0, Z0]
    @test AdmittanceModels.admittance_to_scattering(Y, Z0s) ≈ S
    @test AdmittanceModels.scattering_to_admittance(S, Z0s) ≈ Y
    @test AdmittanceModels.impedance_to_scattering(Z, Z0s) ≈ S
    @test AdmittanceModels.scattering_to_impedance(S, Z0s) ≈ Z
end
