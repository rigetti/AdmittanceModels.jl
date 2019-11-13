module AdmittanceModels
using Compat

@deprecate cascade_and_unite connect

include("linear_algebra.jl")
include("admittance_model.jl")
include("circuit.jl")
include("pso_model.jl")
include("blackbox.jl")
include("circuit_components.jl")
include("ansys.jl")

end
