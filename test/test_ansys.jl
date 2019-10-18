@testset "Q3D plain text parsing" begin
    data(x) = joinpath(dirname(@__FILE__), "data", x)

    # file should satisfy some basic checks
    @test_throws AssertionError Circuit(data("empty.txt"))
    @test_throws AssertionError Circuit(data("one_block.txt"))

    # test expected output
    @test Circuit(data("dummy_gc_1.txt")).vertices == ["net1", "net2", "net3", "net4"]

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
