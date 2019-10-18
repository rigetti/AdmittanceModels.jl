@testset "Q3D plain text parsing" begin
    data(x) = joinpath(@__DIR__, "data", x)

    # file should satisfy some basic checks
    @test_throws AssertionError Circuit(data("empty.txt"))
    @test_throws AssertionError Circuit(data("one_block.txt"))

    # test expected output
    d = data("dummy_gc_1.txt")
    @test Circuit(d).vertices == ["net1", "net2", "net3", "net4"]

    design_variation, vertex_names, capacitance_matrix =
        AdmittanceModels.parse_q3d_txt(d, :capacitance)

    # test negative val + no units
    @test design_variation["dummy1"] == (-1.23, "")

    # test negative exponent + units
    @test design_variation["dummy2"] == (9.0e-7, "mm")

    # test negative val, positive exponent + units
    @test design_variation["dummy3"] == (-1.0e21, "pF")

    # test units that start with "e" (the parser should look for things like 'e09' or 'e-12')
    @test design_variation["dummy4"] == (1.2, "eunits")

    # unit not implemented
    @test_throws ErrorException try
        Circuit(data("dummy_gc_2.txt"); matrix_types = [:conductance])
    catch e
        buf = IOBuffer()
        showerror(buf, e)
        message = String(take!(buf))
        @test occursin("not implemented", message)
        rethrow(e)
    end

    @test_throws ErrorException try
        Circuit(data("dummy_gc_3.txt"); matrix_types = [:conductance])
    catch e
        buf = IOBuffer()
        showerror(buf, e)
        message = String(take!(buf))
        @test occursin("units not given", message)
        rethrow(e)
    end
end
