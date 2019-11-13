using Test, AdmittanceModels
using ProgressMeter: @showprogress
using LinearAlgebra: norm, normalize

# set to false to create the figures in the paper, though it will take longer
run_fast = true
# to save data so figures in the paper can be created, set to true
write_data = false
# to make plots and print information set to true
show_results = false

if write_data
    using DataFrames, CSV, NPZ
    output_folder = joinpath(@__DIR__, "hybridization_results")
    try
        mkdir(output_folder)
    catch SystemError
    end
end

if show_results
    using PlotlyJS
end

#######################################################
# Parameters
#######################################################

begin
    ν = 1.2e8 # propagation_speed
    Z0 = 50.0 # characteristic_impedance
    δ = 50e-6 # discretization length
    pfilter_num = 202
    pfilter_len = pfilter_num * δ # purcell filter length
    pfilter_res_0_coupler_num = floor(Int, pfilter_num/2) - 5
    pfilter_res_coupler_0_len = pfilter_res_0_coupler_num * δ # position of res 0 coupler on filter
    pfilter_res_1_coupler_num = floor(Int, pfilter_num/2) + 5
    pfilter_res_coupler_1_len = pfilter_res_1_coupler_num * δ # position of res 1 coupler on filter
    res_0_num = 102
    res_0_len = res_0_num * δ
    res_1_num = 101
    res_1_len = res_1_num * δ
    res_coupler_num = 16
    res_coupler_len = res_coupler_num * δ # res coupler length from open
    capacitance_0 = 12e-15 # res 0 - purcell filter capacitance
    capacitance_1 = 12e-15 # res 1 - purcell filter capacitance
end

#######################################################
# Two resonators and Purcell filter
# Circuit version
#######################################################

begin
    pfilter_circ = Circuit(TransmissionLine(["p/short_0", "p/short_1"], ν, Z0, pfilter_len, δ=δ),
                           "p/", [pfilter_num])
    res_0_circ = Circuit(TransmissionLine(["r0/open", "r0/short"], ν, Z0, res_0_len, δ=δ),
                         "r0/", [res_0_num])
    res_1_circ = Circuit(TransmissionLine(["r1/open", "r1/short"], ν, Z0, res_1_len, δ=δ),
                         "r1/", [res_1_num])
    pfilter_res_0_coupler_name = "p/$(1+pfilter_res_0_coupler_num)"
    res_0_coupler_name = "r0/$(1+res_coupler_num)"
    pfilter_res_1_coupler_name = "p/$(1+pfilter_res_1_coupler_num)"
    res_1_coupler_name = "r1/$(1+res_coupler_num)"
    capacitor_0 = Circuit(SeriesComponent(pfilter_res_0_coupler_name, res_0_coupler_name, 0, 0, capacitance_0))
    capacitor_1 = Circuit(SeriesComponent(pfilter_res_1_coupler_name, res_1_coupler_name, 0, 0, capacitance_1))
    circ = connect(pfilter_circ, res_0_circ, res_1_circ, capacitor_0, capacitor_1)
    ground = AdmittanceModels.ground
    circ = unite_vertices(circ, ground, "p/short_0", "p/short_1", "r0/short", "r1/short")

    input_output(stub_num) = ("p/$(1+stub_num)", "p/$(pfilter_num-stub_num+1)")
end

function bbox_from_circ(stub_num::Int, ω::Vector{<:Real})
    input, output = input_output(stub_num)
    pso = PSOModel(circ, [(input, ground), (output, ground)], ["in", "out"])
    return Blackbox(ω, pso) |> canonical_gauge
end

#######################################################
# Two resonators and Purcell filter
# Admittance model version
#######################################################

function components_from_model(stub_num::Int, discretization::Real=δ)
    pfilter = TransmissionLine(["p/short_0", "in", "p/res_0_coupler", "p/middle",
                                "p/res_1_coupler", "out", "p/short_1"],
                               ν, Z0, pfilter_len, locations=[stub_num * δ, pfilter_res_coupler_0_len,
                                                              .5 * pfilter_len, pfilter_res_coupler_1_len,
                                                              pfilter_len - stub_num * δ], δ=discretization)
    res_0 = TransmissionLine(["r0/open", "r0/pfilter_coupler", "r0/short"],
                             ν, Z0, res_0_len, locations=[res_coupler_len], δ=discretization)
    res_1 = TransmissionLine(["r1/open", "r1/pfilter_coupler", "r1/short"],
                         ν, Z0, res_1_len, locations=[res_coupler_len], δ=discretization)
    capacitor_0 = SeriesComponent("p/res_0_coupler", "r0/pfilter_coupler", 0, 0, capacitance_0)
    capacitor_1 = SeriesComponent("p/res_1_coupler", "r1/pfilter_coupler", 0, 0, capacitance_1)
    return [pfilter, res_0, res_1, capacitor_0, capacitor_1]
end

function bbox_from_pso(stub_num::Int, ω::Vector{<:Real}, discretization::Real=δ)
    model = connect(PSOModel.(components_from_model(stub_num, discretization)))
    model = short_ports(model, ["p/short_0", "p/short_1", "r0/short", "r1/short"])
    model = open_ports_except(model, ["in", "out"])
    return canonical_gauge(Blackbox(ω, model))
end

function bbox_from_bbox(stub_num::Int, ω::Vector{<:Real})
    model = connect(Blackbox.(Ref(ω), components_from_model(stub_num)))
    model = short_ports(model, ["p/short_0", "p/short_1", "r0/short", "r1/short"])
    model = open_ports_except(model, ["in", "out"])
    return canonical_gauge(model)
end

#######################################################
# Compare them
#######################################################

begin
    ω_circ = collect(range(5.4, stop=6.4, length=run_fast ? 100 : 2000)) * 2π * 1e9

    bbox_circ_10 = bbox_from_circ(10, ω_circ)
    bbox_circ_30 = bbox_from_circ(30, ω_circ)

    bbox_pso_10 = bbox_from_pso(10, ω_circ)
    bbox_pso_30 = bbox_from_pso(30, ω_circ)

    bbox_bbox_10 = bbox_from_bbox(10, ω_circ)
    bbox_bbox_30 = bbox_from_bbox(30, ω_circ)
end

s21(bbox) = [x[1,2] for x in scattering_matrices(bbox, [Z0, Z0])]
freqs = ω_circ/(2π)

if write_data
    df = DataFrame(freqs = freqs,
        circ_S_p5 = s21(bbox_circ_10),
        circ_S_1p5 = s21(bbox_circ_30),
        pso_S_p5 = s21(bbox_pso_10),
        pso_S_1p5 = s21(bbox_pso_30),
        bbox_S_p5 = s21(bbox_bbox_10),
        bbox_S_1p5 = s21(bbox_bbox_30))
    CSV.write(joinpath(output_folder, "S12.csv"), df)
end

if show_results
    plot([scatter(x=freqs/1e6, y=abs.(s21(bbox_circ_10)), name="0.5mm circ"),
          scatter(x=freqs/1e6, y=abs.(s21(bbox_circ_30)), name="1.5mm circ"),
          scatter(x=freqs/1e6, y=abs.(s21(bbox_pso_10)), name="0.5mm pso"),
          scatter(x=freqs/1e6, y=abs.(s21(bbox_pso_30)), name="1.5mm pso"),
          scatter(x=freqs/1e6, y=abs.(s21(bbox_bbox_10)), name="0.5mm bbox", line_dash="dash"),
          scatter(x=freqs/1e6, y=abs.(s21(bbox_bbox_30)), name="1.5mm bbox", line_dash="dash")],
          Layout(xaxis_title="frequency", yaxis_title="|S21|"))
end

if show_results
    abs_diff_10 = abs.(s21(bbox_circ_10) .- s21(bbox_bbox_10))
    abs_diff_30 = abs.(s21(bbox_circ_30) .- s21(bbox_bbox_30))
    plot([scatter(x=ω_circ/(2π*1e6), y=abs_diff_10, name="0.5mm error"),
        scatter(x=ω_circ/(2π*1e6), y=abs_diff_30, name="1.5mm error")],
        Layout(xaxis_title="frequency", yaxis_title="|S21_circ - S21_bbox|"))
end

if show_results
    density(m) = count(!iszero, m)/(size(m,1) * size(m,2))
    model = connect(PSOModel.(components_from_model(10, δ)))
    model = short_ports(model, ["p/short_0", "p/short_1", "r0/short", "r1/short"])
    println("pso matrix densities: $(map(density, get_Y(model)))")
end

#######################################################
# All three models now match so we can use the circuit
# one for making mode plots
#######################################################

function resistor_circ(stub_num::Int)
    input, output = input_output(stub_num)
    resistor_0 = Circuit(ParallelComponent(input, 0, 1/Z0, 0))
    resistor_1 = Circuit(ParallelComponent(output, 0, 1/Z0, 0))
    return connect(circ, resistor_0, resistor_1)
end

begin
    pfilter_vertices = pfilter_circ.vertices[3:end-1]
    res_0_vertices = res_0_circ.vertices[2:end-1]
    res_1_vertices = res_1_circ.vertices[2:end-1]
    vertices = [pfilter_vertices; res_0_vertices; res_1_vertices]
    tree = SpanningTree(ground, [(v, ground) for v in vertices])
    port_edges = [("p/$(floor(Int, (2+pfilter_num)/2))", ground),
                  ("r0/open", ground),
                  ("r1/open", ground)]
    ports = ["pfilter_max", "res_0_max", "res_1_max"]
    mode_names = ["purcell filter", "resonator 0", "resonator 1"]
end

function resistor_pso(stub_num::Int)
    return PSOModel(resistor_circ(stub_num), port_edges, ports, tree)
end

function eigenmodes(stub_num::Int)
    pso = resistor_pso(stub_num)
    eigenvalues, eigenvectors = lossy_modes_dense(pso, min_freq=4e9, max_freq=8e9)
    port_inds = ports_to_indices(pso, ports)
    mode_inds = AdmittanceModels.match_vectors(get_P(pso)[:, port_inds], eigenvectors)
    return eigenvalues[mode_inds], eigenvectors[:,mode_inds]
end

begin
    stub_nums = run_fast ? (1:10:60) : (1:60)
    eigenmodes_arr = @showprogress [eigenmodes(i) for i in stub_nums]
    complex_freqs_arr = hcat([p[1] for p in eigenmodes_arr]...)
    eigenvectors = [p[2] for p in eigenmodes_arr]
end

if write_data
    df = DataFrame(Dict(Symbol(mode_names[i]) => complex_freqs_arr[i,:] for i in 1:3))
    df[:stub_len] = stub_nums * δ
    CSV.write(joinpath(output_folder, "complex_frequencies.csv"), df)

    df = DataFrame(transpose(hcat(eigenvectors...)))
    labels = vcat([[(i, m) for m in mode_names] for i in stub_nums]...)
    df[:stub_len] = [l[1] * δ for l in labels]
    df[:mode_name] = [l[2] for l in labels]
    CSV.write(joinpath(output_folder, "eigenvectors.csv"), df)
end

freq_func(s) = imag(s)/(2π)
decay_func(s) = -2 * real(s)/(2π)
T1_func(s) = 1/(-2 * real(s))

if show_results
    frequency_plots = [scatter(x=stub_nums * δ * 1e3, y=freq_func.(complex_freqs_arr[i, :])/1e9,
        name="$(mode_names[i])", mode="lines+markers") for i in 1:3]
    plot(frequency_plots, Layout(xaxis_title="stub length [mm]", yaxis_title="frequency [GHz]",
        title="Frequencies"))
end

if show_results
    decay_plots = [scatter(x=stub_nums * δ * 1e3, y=decay_func.(complex_freqs_arr[i, :]),
        name="$(mode_names[i])", mode="lines+markers") for i in 1:3]
    plot(decay_plots, Layout(xaxis_title="stub length [mm]", yaxis_title="decay rate [Hz]",
        yaxis_type="log", title="Decay rates"))
end

function spatial_profile(v)
    w = [0; v[1:length(pfilter_vertices)]; 0;
        v[length(pfilter_vertices)+1:length(pfilter_vertices)+length(res_0_vertices)]; 0;
        v[length(pfilter_vertices)+length(res_0_vertices)+1:end]; 0]
    return normalize(abs.(w))
end

mode_1 = run_fast ? 2 : 10
mode_2 = run_fast ? 4 : 30

function spatial_plot(mode_num::Int)
    plot([scatter(y=spatial_profile(eigenvectors[mode_1][:, mode_num]), name= "mode 1"),
          scatter(y=spatial_profile(eigenvectors[mode_2][:, mode_num]), name= "mode 2")],
         Layout(title=mode_names[mode_num]))
end

if show_results
    spatial_plot(1)
end

if show_results
    spatial_plot(2)
end

if show_results
    spatial_plot(3)
end

if write_data
    df = DataFrame(Dict(Symbol(mode_names[mode_num]) => spatial_profile(eigenvectors[mode_1][:, mode_num])
        for mode_num in 1:3))
    CSV.write(joinpath(output_folder, "mode_profile_p5.csv"), df)

    df = DataFrame(Dict(Symbol(mode_names[mode_num]) => spatial_profile(eigenvectors[mode_2][:, mode_num])
        for mode_num in 1:3))
    CSV.write(joinpath(output_folder, "mode_profile_1p5.csv"), df)
end

function regional_support(v)
    x = norm(v[1:length(pfilter_vertices)])
    y = norm(v[length(pfilter_vertices)+1:length(pfilter_vertices)+length(res_0_vertices)])
    z = norm(v[length(pfilter_vertices)+length(res_0_vertices)+1:end])
    return normalize([x, y, z]).^2
end

begin
    support_vects = [[regional_support(vects[:, i]) for vects in eigenvectors] for i in 1:3]
    standard_basis = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    hybridizations = [[norm(v - standard_basis[i],2) for v in support_vects[i]] for i in 1:3]
    get_x(v) = v[2] + v[3]/2
    get_y(v) = v[3] * sqrt(3)/2
    xs = [get_x.(support_vects[i]) for i in 1:3]
    ys = [get_y.(support_vects[i]) for i in 1:3]
end

if show_results
    plot([scatter(x=stub_nums * δ * 1e3, y=hybridizations[i], name=mode_names[i], mode="lines+markers") for i in 1:3],
        Layout(xaxis_title="stub length [mm]", yaxis_title="Distance"))
end

if show_results
    plot([scatter(x=xs[i], y=ys[i], name=mode_names[i]) for i in 1:3],
    Layout(xaxis_range=[0,1], yaxis_range=[0,sqrt(3)/2]))
end

if write_data
    df = DataFrame(Dict(Symbol(mode_names[i]) => hybridizations[i] for i in 1:3))
    df[:stub_len] = stub_nums * δ
    CSV.write(joinpath(output_folder, "hybridization_curves.csv"), df)

    df = DataFrame(Dict(Symbol(mode_names[i]) => xs[i] for i in 1:3))
    CSV.write(joinpath(output_folder, "hybridization_scatter_x.csv"), df)

    df = DataFrame(Dict(Symbol(mode_names[i]) => ys[i] for i in 1:3))
    CSV.write(joinpath(output_folder, "hybridization_scatter_y.csv"), df)
end

#######################################################
# Convergence
#######################################################

function eigenvals(stub_num::Int, discretization::Real)
    comp = components_from_model(stub_num, discretization)
    model = connect(PSOModel.(comp))
    model = short_ports(model, ["p/short_0", "p/short_1", "r0/short", "r1/short"])
    terminations = PSOModel.([ParallelComponent("in",  0, 1/Z0, 0),
                              ParallelComponent("out", 0, 1/Z0, 0)])
    pso = connect([model; terminations])
    eigenvalues, eigenvectors = lossy_modes_dense(pso, min_freq=4e9, max_freq=8e9)
    port_inds = ports_to_indices(pso, "p/middle", "r0/open", "r1/open")
    mode_inds = AdmittanceModels.match_vectors(get_P(pso)[:, port_inds], eigenvectors)
    return eigenvalues[mode_inds]
end

if !run_fast
    discretizations = [60e-6, 50e-6, 40e-6]
    eigenvals_arr = [@showprogress [eigenvals(i, d) for i in stub_nums] for d in discretizations]

    if write_data
        npzwrite(joinpath(output_folder, "convergence_eigenvalues.npy"),
            cat([hcat(eigenvals_arr[i]...) for i in 1:length(eigenvals_arr)]..., dims=3))
    end
end

#######################################################
# Declare victory
#######################################################

@testset "hybridization" begin
    @test true
end
