export AdmittanceModel
export get_Y, get_P, get_ports, partial_copy, compatible, canonical_gauge
export apply_transform, cascade, connect, cascade_and_unite, ports_to_indices
export unite_ports, open_ports, open_ports_except, short_ports, short_ports_except

"""
    abstract type AdmittanceModel{T}

An abstract representation of a linear mapping from inputs `x` to outputs `y` of
the form `YΦ = Px`, `y = QᵀΦ`. The type parameter `T` is the type of a port name
object (typically `String`, `Symbol`, or `Int`). Subtypes `U <: AdmittanceModel`
are expected to implement:

    get_Y(am::U)
    get_P(am::U)
    get_ports(am::U)
    partial_copy(am::U; Y, P, ports)
    compatible(AbstractVector{U})
"""
abstract type AdmittanceModel{T} end

"""
    get_Y(pso::PSOModel)
    get_Y(bbox::Blackbox)

Return a vector of admittance matrices.
"""
function get_Y end

"""
    get_P(pso::PSOModel)
    get_P(bbox::Blackbox)

Return an input port matrix.
"""
function get_P end

"""
    get_ports(pso::PSOModel)
    get_ports(bbox::Blackbox)

Return a vector of port identifiers.
"""
function get_ports end

"""
    partial_copy(pso::PSOModel{T, U};
        Y::Union{Vector{V}, Nothing}=nothing,
        P::Union{V, Nothing}=nothing,
        Q::Union{V, Nothing}=nothing,
        ports::Union{AbstractVector{W}, Nothing}=nothing) where {T, U, V, W}

    partial_copy(bbox::Blackbox{T, U};
        Y::Union{Vector{V}, Nothing}=nothing,
        P::Union{V, Nothing}=nothing,
        Q::Union{V, Nothing}=nothing,
        ports::Union{AbstractVector{W}, Nothing}=nothing) where {T, U, V, W}

Create a new model with the same fields except those given as keyword arguments.
"""
function partial_copy end

"""
    compatible(psos::AbstractVector{PSOModel{T, U}}) where {T, U}
    compatible(bboxes::AbstractVector{Blackbox{T, U}}) where {T, U}

Check if the models can be cascaded. Always true for PSOModels and true for Blackboxes
that share the same value of `ω`.
"""
function compatible end

"""
    canonical_gauge(pso::PSOModel)
    canonical_gauge(bbox::Blackbox)

Apply an invertible transformation that takes the model to coordinates in which
`P` is `[I ; 0]` (up to floating point errors). Note this will create a dense model.
"""
function canonical_gauge end

"""
    ==(am1::AdmittanceModel, am2::AdmittanceModel)

Test whether two `AdmittanceModel`s are equal. Two models must be of the same type
to be equal, as well as have the same admittances, port matrices, and port names.
"""
# don't remove space between Base. and ==
function Base. ==(am1::T, am2::T) where {T <: AdmittanceModel}
    return all(getfield(am1, name) == getfield(am2, name) for name in fieldnames(T))
end

"""
    isapprox(am1::AdmittanceModel, am2::AdmittanceModel)

Test whether two `AdmittanceModel`s are approximately equal.
"""
Base.isapprox(am1::AdmittanceModel, am2::AdmittanceModel) = false
function Base.isapprox(am1::T, am2::T) where {T <: AdmittanceModel}
    return all(getfield(am1, name) ≈ getfield(am2, name) for name in fieldnames(T))
end

"""
    apply_transform(am::AdmittanceModel, transform::AbstractMatrix{<:Number})

Apply a linear transformation `transform` to the coordinates of the model.
"""
function apply_transform(am::AdmittanceModel, transform::AbstractMatrix{<:Number})
    Y = [transpose(transform) * m * transform for m in get_Y(am)]
    P = eltype(Y)(transpose(transform) * get_P(am))
    return partial_copy(am, Y=Y, P=P)
end

"""
    cascade(ams::AbstractVector{<:AdmittanceModel})
    cascade(ams::AdmittanceModel...)

Cascade the models into one larger block diagonal model.
"""
function cascade(ams::AbstractVector{<:AdmittanceModel})
    @assert !isempty(ams)
    if length(ams) == 1
        return ams[1]
    end
    @assert compatible(ams)
    Y = [cat(m..., dims=(1,2)) for m in zip([get_Y(am) for am in ams]...)]
    P = cat([get_P(am) for am in ams]..., dims=(1,2))
    ports = vcat([get_ports(am) for am in ams]...)
    return partial_copy(ams[1], Y=Y, P=P, ports=ports)
end
cascade(ams::AdmittanceModel...) = cascade(collect(ams))

"""
    ports_to_indices(am::AdmittanceModel, ports::AbstractVector)
    ports_to_indices(am::AdmittanceModel, ports...)

Find the indices corresponding to given ports.
"""
function ports_to_indices(am::AdmittanceModel, ports::AbstractVector)
    am_ports = get_ports(am)
    return [findfirst(isequal(p), am_ports) for p in ports]
end
ports_to_indices(am::AdmittanceModel, ports...) = ports_to_indices(am, collect(ports))

"""
    unite_ports(am::AdmittanceModel, ports::AbstractVector)
    unite_ports(am::AdmittanceModel, ports...)

Unite the given ports into one port.
"""
function unite_ports(am::AdmittanceModel, ports::AbstractVector)
    if length(ports) <= 1
        return am
    end
    port_inds = ports_to_indices(am, ports)
    keep_inds = filter(!in(port_inds[2:end]), 1:length(get_ports(am))) # keep the first port
    P = get_P(am)
    first_vector = P[:, port_inds[1]]
    constraint_mat = transpose(hcat([first_vector - P[:, i] for i in port_inds[2:end]]...))
    constrained_am = apply_transform(am, nullbasis(constraint_mat))
    return partial_copy(constrained_am, P=get_P(constrained_am)[:, keep_inds],
        ports=get_ports(constrained_am)[keep_inds])
end
unite_ports(am::AdmittanceModel, ports...) = unite_ports(am, collect(ports))

"""
    open_ports(am::AdmittanceModel, ports::AbstractVector)
    open_ports(am::AdmittanceModel, ports...)

Remove the given ports.
"""
function open_ports(am::AdmittanceModel, ports::AbstractVector)
    if length(ports) == 0
        return am
    end
    port_inds = ports_to_indices(am, ports)
    keep_inds = filter(!in(port_inds), 1:length(get_ports(am)))
    return partial_copy(am, P=get_P(am)[:, keep_inds],
        ports=get_ports(am)[keep_inds])
end
open_ports(am::AdmittanceModel, ports...) = open_ports(am, collect(ports))

"""
    open_ports_except(am::AdmittanceModel, ports::AbstractVector)
    open_ports_except(am::AdmittanceModel, ports...)

Remove all ports except those specified.
"""
function open_ports_except(am::AdmittanceModel, ports::AbstractVector)
    return open_ports(am, filter(!in(ports), get_ports(am)))
end
open_ports_except(am::AdmittanceModel, ports...) = open_ports_except(am, collect(ports))

"""
    short_ports(am::AdmittanceModel, ports::AbstractVector)
    short_ports(am::AdmittanceModel, ports...)

Replace the given ports by short circuits.
"""
function short_ports(am::AdmittanceModel, ports::AbstractVector)
    isempty(ports) && return am
    port_inds = ports_to_indices(am, ports)
    keep_inds = filter(!in(port_inds), 1:length(get_ports(am)))
    constraint_mat = transpose(hcat([get_P(am)[:, i] for i in port_inds]...))
    constrained_am = apply_transform(am, nullbasis(constraint_mat))
    return partial_copy(constrained_am, P = get_P(constrained_am)[:, keep_inds],
        ports = get_ports(constrained_am)[keep_inds])
end
short_ports(am::AdmittanceModel, ports...) = short_ports(am, collect(ports))

"""
    short_ports_except(am::AdmittanceModel, ports::AbstractVector)
    short_ports_except(am::AdmittanceModel, ports...)

Replace all ports with short circuits, except those specified.
"""
function short_ports_except(am::AdmittanceModel, ports::AbstractVector)
    return short_ports(am, filter(!in(ports), get_ports(am)))
end
short_ports_except(am::AdmittanceModel, ports...) =
    short_ports_except(am, collect(ports))

"""
    connect(models::AbstractVector{<:AdmittanceModel})
    connect(models::AdmittanceModel...)

Cascade all models and unite ports with the same name. This results in a
combined admittance model with common ports connected.
"""
function connect(models::AbstractVector{<:AdmittanceModel})
    @assert length(models) >= 1
    if length(models) == 1
        return models[1]
    end
    # number the ports so that the names are all distinct and then cascade
    port_number = 1
    function rename(model)
        ports = [(port_number + i - 1, port) for (i, port) in enumerate(get_ports(model))]
        port_number += length(ports)
        return partial_copy(model, ports=ports)
    end
    model = cascade(map(rename, models))
    # merge all ports with the same name
    original_ports = vcat([get_ports(m) for m in models]...)
    for port in unique(original_ports)
        inds = findall([p[2] == port for p in get_ports(model)])
        model = unite_ports(model, get_ports(model)[inds])
    end
    # remove numbering
    return partial_copy(model, ports=[p[2] for p in get_ports(model)])
end
connect(models::AdmittanceModel...) = connect(collect(models))
