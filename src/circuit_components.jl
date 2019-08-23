export CircuitComponent, SeriesComponent, ParallelComponent, TransmissionLine
export Cascade, PortOperation

"""
An abstract representation of a circuit component e.g. a capacitor, inductor, or
transmission line.
"""
abstract type CircuitComponent end

# the empty string is used for the ground net
ground = ""

"""
    Blackbox(ω::Vector{<:Real}, comp::CircuitComponent)
    Blackbox(ω::Vector{<:Real}, comp::TransmissionLine)
    Blackbox(ω::Vector{<:Real}, comp::Cascade)

Create a Blackbox model for a CircuitComponent.
"""
function Blackbox(ω::Vector{<:Real}, comp::CircuitComponent)
    return canonical_gauge(Blackbox(ω, PSOModel(comp)))
end

"""
    SeriesComponent(p1::String, p2::String, inv_inductance::Float64, conductance::Float64,
        capacitance::Float64)
    SeriesComponent(p1::String, p2::String, k::Real, g::Real, c::Real)

A parallel inductor, capacitor, and resistor between two ports.
"""
struct SeriesComponent <: CircuitComponent
    p1::String
    p2::String
    inv_inductance::Float64
    conductance::Float64
    capacitance::Float64
    function SeriesComponent(p1::String, p2::String, inv_inductance::Float64,
        conductance::Float64, capacitance::Float64)
        @assert p1 != ground
        @assert p2 != ground
        return new(p1, p2, inv_inductance, conductance, capacitance)
    end
end

SeriesComponent(p1::String, p2::String, k::Real, g::Real, c::Real) =
    SeriesComponent(p1, p2, 1.0 * k, 1.0 * g, 1.0 * c)

"""
    Circuit(comp::SeriesComponent)
    Circuit(comp::ParallelComponent)
    Circuit(comp::TransmissionLine, vertex_prefix::AbstractString,
        stages_per_segment::Union{Nothing, AbstractVector{Int}}=nothing)

Create a Circuit for a CircuitComponent.
"""
function Circuit(comp::SeriesComponent)
    c = Circuit([ground, comp.p1, comp.p2])
    set_inv_inductance!(c, comp.p1, comp.p2, comp.inv_inductance)
    set_conductance!(c, comp.p1, comp.p2, comp.conductance)
    set_capacitance!(c, comp.p1, comp.p2, comp.capacitance)
    return c
end

"""
    PSOModel(comp::SeriesComponent)
    PSOModel(comp::ParallelComponent)
    PSOModel(comp::TransmissionLine)
    PSOModel(comp::Cascade)

Create a PSOModel for a CircuitComponent.
"""
function PSOModel(comp::SeriesComponent)
    return PSOModel(Circuit(comp), [(comp.p1, ground), (comp.p2, ground)], [comp.p1, comp.p2])
end

"""
    ParallelComponent(p::String, inv_inductance::Float64,
        conductance::Float64, capacitance::Float64)
    ParallelComponent(p::String, k::Real, g::Real, c::Real)

A parallel inductor, capacitor, and resistor with one port.
"""
struct ParallelComponent <: CircuitComponent
    p::String
    inv_inductance::Float64
    conductance::Float64
    capacitance::Float64
    function ParallelComponent(p::String, inv_inductance::Float64,
        conductance::Float64, capacitance::Float64)
        @assert p != ground
        return new(p, inv_inductance, conductance, capacitance)
    end
end

ParallelComponent(p::String, k::Real, g::Real, c::Real) =
    ParallelComponent(p, 1.0 * k, 1.0 * g, 1.0 * c)

function Circuit(comp::ParallelComponent)
    c = Circuit([ground, comp.p])
    set_inv_inductance!(c, comp.p, ground, comp.inv_inductance)
    set_conductance!(c, comp.p, ground, comp.conductance)
    set_capacitance!(c, comp.p, ground, comp.capacitance)
    return c
end

PSOModel(comp::ParallelComponent) = PSOModel(Circuit(comp), [(comp.p, ground)], [comp.p])

"""
    TransmissionLine(ports::Vector{String}, propagation_speed::Float64,
        characteristic_impedance::Float64, len::Float64,
        locations::AbstractVector{Float64}, δ::Float64)
    TransmissionLine(ports::Vector{String}, propagation_speed::Real,
        characteristic_impedance::Real, len::Real; locations::Vector{<:Real}=Float64[],
        δ::Real=len/100)

A transmission line with given propagation speed and characteristic impedance. Ports will
be present at the two ends of the transmission line in addition to the `locations`. `δ` is
the maximum length of an LC stage in the circuit approximation to a transmission line. It
is ignored when constructing a Blackbox from a TransmissionLine.
"""
struct TransmissionLine <: CircuitComponent
    ports::Vector{String}
    propagation_speed::Float64
    characteristic_impedance::Float64
    len::Float64 # length, but that's the name of a built in function
    port_locations::UniqueVector{Float64}
    δ::Float64 # ignored when converted directly to Blackbox
    function TransmissionLine(ports::Vector{String}, propagation_speed::Float64,
        characteristic_impedance::Float64, len::Float64, locations::AbstractVector{Float64},
        δ::Float64)
        @assert !(ground in ports)
        @assert propagation_speed > 0
        @assert characteristic_impedance > 0
        @assert len > 0
        @assert all(locations .> 0)
        @assert all(locations .< len)
        port_locations = UniqueVector([0; locations; len])
        @assert length(ports) == length(port_locations)
        @assert δ >= 0
        return new(ports, propagation_speed, characteristic_impedance,
            len, port_locations, δ)
    end
end

TransmissionLine(ports::Vector{String}, propagation_speed::Real,
    characteristic_impedance::Real, len::Real; locations::Vector{<:Real}=Float64[],
    δ::Real=len/100) = TransmissionLine(ports, propagation_speed,
    characteristic_impedance, len, locations, δ)

function Circuit_and_ports(comp::TransmissionLine,
    stages_per_segment::Union{Nothing, AbstractVector{Int}}=nothing)
    sort_inds = sortperm(comp.port_locations)
    port_locations, ports = comp.port_locations[sort_inds], comp.ports[sort_inds]
    ν, Z0, δ = comp.propagation_speed, comp.characteristic_impedance, comp.δ
    segment_lengths = port_locations[2:end] - port_locations[1:end-1]
    if stages_per_segment == nothing
        stages_per_segment = [ceil(Int, len / δ) for len in segment_lengths]
    end
    num_vertices = sum(stages_per_segment) + 2 # ground is 0
    c = Circuit(0:(num_vertices-1))
    v0 = 0
    port_edges = Tuple{Int, Int}[]
    for (segment_length, num_stages) in zip(segment_lengths, stages_per_segment)
        stage_length = 1.0 * segment_length / num_stages
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

function Circuit(comp::TransmissionLine, vertex_prefix::AbstractString,
    stages_per_segment::Union{Nothing, AbstractVector{Int}}=nothing)
    c, port_edges, ports = Circuit_and_ports(comp, stages_per_segment)
    vertices = [ground; ["$(vertex_prefix)$v" for v in c.vertices[2:end]]]
    for (e, p) in zip(port_edges, ports)
        vertices[e[1]+1] = p
    end
    return Circuit(matrices(c)..., vertices)
end

PSOModel(comp::TransmissionLine) = PSOModel(Circuit_and_ports(comp)...)

function Blackbox(ω::Vector{<:Real}, comp::TransmissionLine)
    ν = comp.propagation_speed
    Z0 = comp.characteristic_impedance
    function Blackbox_from_params(ports::Vector, len::Real)
        u = exp.(-1im * ω * len/ν)
        # S matrix is [[0, u] [u, 0]]
        # turn this into Y matrix using closed form expression
        denom = (1 .- u.^2) * Z0
        Y11_Y22 = (1 .+ u.^2) ./ denom
        Y12_Y21 = -2 * u ./ denom
        Y = [[[a, b] [b, a]] for (a, b) in zip(Y11_Y22, Y12_Y21)]
        U = eltype(eltype(Y))
        P = Matrix{U}(I, 2, 2)
        return Blackbox(ω, Y, P, P, ports)
    end
    sort_inds = sortperm(comp.port_locations)
    sort_port_locations = comp.port_locations[sort_inds]
    sort_ports = comp.ports[sort_inds]
    params = [(ports = [sort_ports[i], sort_ports[i + 1]],
              len = sort_port_locations[i+1] - sort_port_locations[i])
              for i in 1:(length(comp.ports)-1)]
    return cascade_and_unite([Blackbox_from_params(p...) for p in params])
end

"""
The functions allowed in a Cascade.
"""
PortOperation = Union{typeof(short_ports),
                      typeof(short_ports_except),
                      typeof(open_ports),
                      typeof(open_ports_except),
                      typeof(unite_ports),
                      typeof(canonical_gauge)}

"""
    Cascade(components::Vector{CircuitComponent},
        operations::Vector{Pair{PortOperation, Vector{String}}})

A system created by cascading and uniting several CircuitComponents and then applying
PortOperations to the ports.
"""
struct Cascade <: CircuitComponent
    components::Vector{CircuitComponent}
    operations::Vector{Pair{PortOperation, Vector{String}}}
end

function PSOModel(comp::Cascade)
    model = cascade_and_unite([PSOModel(m) for m in comp.components])
    for (op, args) in comp.operations
        model = op(model, args...)
    end
    return model
end

function Blackbox(ω::Vector{<:Real}, comp::Cascade)
    model = cascade_and_unite([Blackbox(ω, m) for m in comp.components])
    for (op, args) in comp.operations
        model = op(model, args...)
    end
    return model
end
