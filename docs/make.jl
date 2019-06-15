using Documenter, AdmittanceModels

makedocs(modules = [AdmittanceModels],
    sitename="AdmittanceModels.jl",
    pages = ["Main" => "index.md"])

deploydocs(
    repo = "github.com/rigetti/AdmittanceModels.jl.git",
)
