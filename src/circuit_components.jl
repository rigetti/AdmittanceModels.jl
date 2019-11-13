export CircuitComponent, SeriesComponent, ParallelComponent, TransmissionLine

"""
    abstract type CircuitComponent

An abstract representation of a circuit component, including lumped circuit
elements like a capacitor, inductor, or resistor, but also distributed structures
such as a transmission line.
"""
abstract type CircuitComponent end

# the empty string is used for the ground net
ground = ""

"""
    Blackbox(ω::AbstractVector{<:Number}, comp::CircuitComponent)

Create a `Blackbox` model for a `CircuitComponent`.
"""
function Blackbox(ω::AbstractVector{<:Number}, comp::CircuitComponent)
    @assert isreal(ω)
    return canonical_gauge(Blackbox(ω, PSOModel(comp)))
end

"""
    SeriesComponent(p1::AbstractString, p2::AbstractString,
        k::Number, g::Number, c::Number)

Constructs an inductance (`1/k`), resistance (`1/g`), and capacitance (`c`)
in parallel connecting two ports `p1` and `p2`:

             ┌─ 1/k ─┐
    ─────┬───┼─ 1/g ─┼───┬─────
      p1 ╎   └─  c  ─┘   ╎ p2
         ╎               ╎
    ─────┴───────────────┴─────
"""
struct SeriesComponent <: CircuitComponent
    p1::String
    p2::String
    k::Float64
    g::Float64
    c::Float64
    function SeriesComponent(p1::AbstractString, p2::AbstractString, k::Number,
            g::Number, c::Number)
        @assert p1 != ground
        @assert p2 != ground
        @assert isreal(k) && isreal(g) && isreal(c)
        return new(p1, p2, k, g, c)
    end
end

"""
    Circuit(comp::SeriesComponent)

Create a `Circuit` model for a `SeriesComponent`.
"""
function Circuit(comp::SeriesComponent)
    c = Circuit([ground, comp.p1, comp.p2])
    set_inv_inductance!(c, comp.p1, comp.p2, comp.k)
    set_conductance!(c, comp.p1, comp.p2, comp.g)
    set_capacitance!(c, comp.p1, comp.p2, comp.c)
    return c
end

"""
    PSOModel(comp::SeriesComponent)

Create a `PSOModel` for a `SeriesComponent`.
"""
function PSOModel(comp::SeriesComponent)
    return PSOModel(Circuit(comp), [(comp.p1, ground), (comp.p2, ground)], [comp.p1, comp.p2])
end

"""
    ParallelComponent(p::AbstractString, k::Number, g::Number, c::Number)

Constructs an inductance (`1/k`), resistance (`1/g`), and capacitance (`c`) in
parallel across a port `p`:

    ──────────┬──────────
              │ port p
          ┌───┼───┐
          │   │   │
         1/k  c  1/g
          │   │   │
          └───┼───┘
              │
    ──────────┴──────────
"""
struct ParallelComponent <: CircuitComponent
    p::String
    k::Float64
    g::Float64
    c::Float64
    function ParallelComponent(p::AbstractString, k::Number, g::Number, c::Number)
        @assert p != ground
        @assert isreal(k) && isreal(g) && isreal(c)
        return new(p, k, g, c)
    end
end

"""
    Circuit(comp::ParallelComponent)

Create a `Circuit` model for a `ParallelComponent`.
"""
function Circuit(comp::ParallelComponent)
    c = Circuit([ground, comp.p])
    set_inv_inductance!(c, comp.p, ground, comp.k)
    set_conductance!(c, comp.p, ground, comp.g)
    set_capacitance!(c, comp.p, ground, comp.c)
    return c
end

"""
    PSOModel(comp::ParallelComponent)

Create a `PSOModel` for a `ParallelComponent`.
"""
PSOModel(comp::ParallelComponent) = PSOModel(Circuit(comp), [(comp.p, ground)], [comp.p])

struct TransmissionLine <: CircuitComponent
    ports::Vector{String}
    propagation_speed::Float64
    characteristic_impedance::Float64
    len::Float64 # length, but that's the name of a built in function
    port_locations::UniqueVector{Float64}
    δ::Float64 # ignored when converted directly to Blackbox
    function TransmissionLine(ports::AbstractVector{<:AbstractString},
            propagation_speed::Number, characteristic_impedance::Number,
            len::Number, locations::AbstractVector{<:Number}, δ::Number)
        @assert isreal(propagation_speed) && isreal(characteristic_impedance)
        @assert isreal(len) && isreal(δ)
        @assert !(ground in ports)
        @assert propagation_speed > zero(propagation_speed)
        @assert characteristic_impedance > zero(characteristic_impedance)
        @assert len > zero(len)
        @assert all(locations .> zero(eltype(locations)))
        @assert all(locations .< len)
        port_locations = UniqueVector([zero(len); locations; len])
        @assert length(ports) == length(port_locations)
        @assert δ >= zero(δ)
        return new(ports, propagation_speed, characteristic_impedance,
            len, port_locations, δ)
    end
end

"""
    TransmissionLine(ports::AbstractVector{<:AbstractString},
        propagation_speed::Number, characteristic_impedance::Number, len::Number;
        locations::AbstractVector{<:Number} = Float64[], δ::Number = len/100)

Create an abstract representation of a transmission line with given propagation
speed, characteristic impedance, and length. Ports will be present at the two
ends of the transmission line in addition to the `locations`. `δ` is the maximum
length of an LC stage in the circuit approximation to a transmission line. It is
ignored when constructing a Blackbox from a TransmissionLine.
"""
TransmissionLine(ports::AbstractVector{<:AbstractString}, propagation_speed::Number,
        characteristic_impedance::Number, len::Number;
        locations::AbstractVector{<:Number} = Float64[],
        δ::Number = len/100) =
    TransmissionLine(ports, propagation_speed, characteristic_impedance,
        len, locations, δ)

function Circuit_and_ports(comp::TransmissionLine,
        stages_per_segment::Union{Nothing, AbstractVector{Int}} = nothing)
    sort_inds = sortperm(comp.port_locations)
    port_locations, ports = comp.port_locations[sort_inds], comp.ports[sort_inds]
    ν, Z0, δ = comp.propagation_speed, comp.characteristic_impedance, comp.δ
    segment_lengths = port_locations[2:end] - port_locations[1:end-1]
    if isnothing(stages_per_segment)
        stages_per_segment = [ceil(Int, len / δ) for len in segment_lengths]
    end
    num_vertices = sum(stages_per_segment) + 2 # ground is 0
    c = Circuit(0:(num_vertices-1))
    v0 = 0
    port_edges = Tuple{Int, Int}[]
    for (segment_length, num_stages) in zip(segment_lengths, stages_per_segment)
        stage_length = float(segment_length / num_stages)
        inductance = stage_length * Z0 / ν
        capacitance = stage_length / (Z0 * ν)
        for v in (v0 + 1):(v0 + num_stages)
            set_inductance!(c, v, v + 1, inductance)
        end
        for v in [v0 + 1, v0 + num_stages + 1]
            set_capacitance!(c, v, 0, 0.5 * capacitance + get_capacitance(c, v, 0))
        end
        for v in (v0 + 2):(v0 + num_stages)
            set_capacitance!(c, v, 0, capacitance)
        end
        push!(port_edges, (v0 + 1, 0))
        v0 += num_stages
    end
    @assert v0 + 1 == num_vertices-1
    push!(port_edges, (num_vertices-1, 0))
    return c, port_edges, ports
end

"""
    Circuit(comp::TransmissionLine, vertex_prefix::AbstractString,
        stages_per_segment::Union{Nothing, AbstractVector{Int}} = nothing)

Create a `Circuit` model for a `TransmissonLine`.
"""
function Circuit(comp::TransmissionLine, vertex_prefix::AbstractString,
        stages_per_segment::Union{Nothing, AbstractVector{Int}} = nothing)
    c, port_edges, ports = Circuit_and_ports(comp, stages_per_segment)
    vertices = [ground; ["$(vertex_prefix)$v" for v in c.vertices[2:end]]]
    for (e, p) in zip(port_edges, ports)
        vertices[e[1]+1] = p
    end
    return Circuit(matrices(c)..., vertices)
end

"""
    PSOModel(comp::TransmissionLine)

Create a `PSOModel` for a `TransmissionLine`.
"""
PSOModel(comp::TransmissionLine) = PSOModel(Circuit_and_ports(comp)...)

"""
    Blackbox(ω::Vector{<:Number}, comp::TransmissionLine)

Create a `Blackbox` model from a `TransmissionLine`.
"""
function Blackbox(ω::Vector{<:Number}, comp::TransmissionLine)
    @assert isreal(ω)
    ν = comp.propagation_speed
    Z0 = comp.characteristic_impedance
    function Blackbox_from_params(ports::Vector, len::Real)
        u = exp.(-im * ω * len/ν)
        # S matrix is [[0, u] [u, 0]]
        # turn this into Y matrix using closed form expression
        denom = (1 .- u.^2) * Z0
        Y11_Y22 = (1 .+ u.^2) ./ denom
        Y12_Y21 = -2 * u ./ denom
        Y = [[[a, b] [b, a]] for (a, b) in zip(Y11_Y22, Y12_Y21)]
        U = eltype(eltype(Y))
        P = Matrix{U}(I, 2, 2)
        return Blackbox(ω, Y, P, ports)
    end
    sort_inds = sortperm(comp.port_locations)
    sort_port_locations = comp.port_locations[sort_inds]
    sort_ports = comp.ports[sort_inds]
    params = [(ports = [sort_ports[i], sort_ports[i + 1]],
              len = sort_port_locations[i+1] - sort_port_locations[i])
              for i in 1:(length(comp.ports)-1)]
    return connect([Blackbox_from_params(p...) for p in params])
end
