using Test, AdmittanceModels
using ProgressMeter: @showprogress

# set to false to create the figures in the paper, though it will take longer
run_fast = true
# to save data so figures in the paper can be created, set to true
write_data = false
# to make plots and print information set to true
show_results = false

if write_data
    using DataFrames, CSV, NPZ
    output_folder = joinpath(@__DIR__, "radiative_loss_results")
    try
        mkdir(output_folder)
    catch SystemError
    end
end

if show_results
    using PlotlyJS
end

#######################################################
# Shared parameters
#######################################################

begin
    ν = 1.2e8 # propagation_speed
    Z0 = 50.0 # characteristic_impedance
    δ = run_fast ? 200e-6 : 50e-6 # discretization length
    cg = 7e-15 # qubit - resonator coupling capacitance
    cj = 100e-15 # qubit capacitance
    lj = 10e-9 # qubit inductance
    x0 = 800e-6 # coupler location on resonator
end

#######################################################
# Transmission line model
#######################################################

begin
    L = .00499171 # resonator length
    F = 2e-3 # length of tline
    y0 = F/2 # coupler location on tline
    cc = 10e-15 # resonator - tline coupling capacitance
    tline_params = (L=L, x0=x0, F=F, y0=y0, cc=cc)
end

function create_tline_components(δ::Real)
    resonator = TransmissionLine(["qr_coupler", "er_coupler_0", "mode_2_max", "short"],
                                 ν, Z0, L, locations=[x0, 2*L/3], δ=δ)
    environment = TransmissionLine(["in", "er_coupler_1", "out"],
                                   ν, Z0, F, locations=[y0], δ=δ)
    er_coupler = SeriesComponent("er_coupler_0", "er_coupler_1", 0, 0, cc)
    qr_coupler = SeriesComponent("qubit", "qr_coupler", 0, 0, cg)
    return [resonator, environment, er_coupler, qr_coupler]
end
create_tline_pso(δ) = short_ports(
    connect(PSOModel.(create_tline_components(δ))), "short")

tline_cascade_comp    = create_tline_components(δ)
tline_cascade_pso     = create_tline_pso(δ)
tline_cascade_bbox(ω) = short_ports(
    connect(Blackbox.(Ref(ω), create_tline_components(δ))), "short")

#######################################################
# Purcell filter model
#######################################################

begin
    L = .00503299# resonator length
    F = .0102522 # length of pfilter
    y0 = 5e-3 # coupler location on pfilter
    cc = 2.6e-15 # resonator - pfilter coupling capacitance
    st = 880e-6 # pfilter stub length
    pfilter_params = (L=L, x0=x0, F=F, y0=y0, cc=cc, st=st)
end

function create_pfilter_components(δ::Real)
    resonator = TransmissionLine(["qr_coupler", "er_coupler_0", "mode_2_max", "short 1"],
                                 ν, Z0, L, locations=[x0, 2*L/3], δ=δ)
    environment = TransmissionLine(["short 2", "in", "er_coupler_1", "out", "short 3"],
                                   ν, Z0, F, locations=[st, y0, F-st], δ=δ)
    er_coupler = SeriesComponent("er_coupler_0", "er_coupler_1", 0, 0, cc)
    qr_coupler = SeriesComponent("qubit", "qr_coupler", 0, 0, cg)
    return [resonator, environment, er_coupler, qr_coupler]
end
create_pfilter_pso(δ) = short_ports(connect(
    PSOModel.(create_pfilter_components(δ))), ["short 1", "short 2", "short 3"])

pfilter_cascade_comp = create_pfilter_components(δ)
pfilter_cascade_pso  = create_pfilter_pso(δ)
pfilter_cascade_bbox(ω) = short_ports(connect(
    Blackbox.(Ref(ω), pfilter_cascade_comp)), ["short 1", "short 2", "short 3"])
#######################################################
# Scattering parameters with qubit
#######################################################

begin
    factor = run_fast ? 2 : 10
    ω = [range(1.5, stop=5.89, length=10 * factor);
         range(5.89, stop=5.92, length=60 * factor); # first mode for both
         range(5.92, stop=17.69, length=10 * factor);
         range(17.69, stop=17.72, length=100 * factor); # 2nd mode pfilter
         range(17.72, stop=17.76, length=10 * factor);
         range(17.76, stop=17.85, length=100 * factor); # 2nd mode tline
         range(17.85, stop=25, length=10 * factor)] * 2π * 1e9
    qubit = ParallelComponent("qubit", 1/lj, 0, cj)

    tline_pso_bbox = let
        model = connect(tline_cascade_pso, PSOModel(qubit))
        model = open_ports_except(model, ["in", "out"])
        Blackbox(ω, model)
    end
    tline_pso_bbox_s = [x[1,2] for x in scattering_matrices(tline_pso_bbox, [Z0, Z0])]
    tline_bbox = let
        model = connect(tline_cascade_bbox(ω), Blackbox(ω, qubit))
        model = open_ports_except(model, ["in", "out"])
    end
    tline_bbox_s = [x[1,2] for x in scattering_matrices(tline_bbox, [Z0, Z0])]
    pfilter_pso_bbox = let
        model = connect(pfilter_cascade_pso, PSOModel(qubit))
        Blackbox(ω, open_ports_except(model, ["in", "out"]))
    end
    pfilter_pso_bbox_s = [x[1,2] for x in scattering_matrices(pfilter_pso_bbox, [Z0, Z0])]
    pfilter_bbox = let
        model = connect(pfilter_cascade_bbox(ω), Blackbox(ω, qubit))
        open_ports_except(model, ["in", "out"])
    end
    pfilter_bbox_s = [x[1,2] for x in scattering_matrices(pfilter_bbox, [Z0, Z0])]
end

if write_data
    CSV.write(joinpath(output_folder, "S12.csv"), DataFrame(
        freqs = ω/(2π),
        tline_S = tline_pso_bbox_s,
        pfilter_S = pfilter_pso_bbox_s))
end

if show_results
    plot([scatter(x=ω/(2π*1e9), y=abs.(tline_pso_bbox_s), name="tline pso"),
          scatter(x=ω/(2π*1e9), y=abs.(tline_bbox_s), name="tline bbox"),
          scatter(x=ω/(2π*1e9), y=abs.(pfilter_pso_bbox_s), name="pfilter pso", line_dash="dash"),
          scatter(x=ω/(2π*1e9), y=abs.(pfilter_bbox_s), name="pfilter bbox", line_dash="dash")],
          Layout(xaxis_title="frequency", yaxis_title="|S21|"))
end

#######################################################
# Find effective capacitance
#######################################################

function terminated(model::PSOModel)
    resistors = ParallelComponent.(["in", "out"], 0, 1/Z0, 0)
    return connect([model; PSOModel.(resistors)])
end
function admittances(model::Blackbox)
    resistors = ParallelComponent.(["in", "out"], 0, 1/Z0, 0)
    model = connect([model; Blackbox.(Ref(model.ω), resistors)])
    bb = canonical_gauge(open_ports_except(model, "qubit"))
    return [x[1,1] for x in admittance_matrices(bb)]
end

begin
    fit_ω = collect(range(0.1e9, stop=2.5e9, length=run_fast ? 100 : 1000)) * 2π
    plot_ω = collect(range(0.01e9, stop=10e9, length=run_fast ? 100 : 1000)) * 2π
    linreg(x, y) = hcat(fill!(similar(x), 1), x) \ y
    tline_pso, Y = terminated(tline_cascade_pso), admittances(tline_cascade_bbox(fit_ω))
    _, tline_slope = linreg(fit_ω, imag.(Y))
    tline_Y = admittances(tline_cascade_bbox(plot_ω))
    Y = admittances(pfilter_cascade_bbox(fit_ω))
    _, pfilter_slope = linreg(fit_ω, imag.(Y))
    pfilter_pso, pfilter_Y = terminated(tline_cascade_pso), admittances(tline_cascade_bbox(plot_ω))
    tline_eff_capacitance = tline_slope + cj
    pfilter_eff_capacitance = pfilter_slope + cj
end


if show_results
    println("tline_slope: $tline_slope, pfilter_slope: $pfilter_slope, cg: $cg")
    println("tline_eff_capacitance: $tline_eff_capacitance, pfilter_eff_capacitance: $pfilter_eff_capacitance")
    density(m) = count(!iszero, m)/(size(m,1) * size(m,2))
    println("tline matrix densities: $(map(density, get_Y(tline_pso)))")
    println("pfilter matrix densities: $(map(density, get_Y(tline_pso)))")
end

if write_data
    CSV.write(joinpath(output_folder, "Y.csv"), DataFrame(
        freqs = plot_ω/(2π),
        tline_Y = tline_Y,
        pfilter_Y = pfilter_Y))
end

if show_results
    plot([scatter(x=plot_ω/(2π), y=imag.(tline_Y), name="imag(tline_Y)"),
          scatter(x=plot_ω/(2π), y=imag.(pfilter_Y), name="imag(pfilter_Y)", line_dash="dash"),
          scatter(x=plot_ω/(2π), y=cg * plot_ω, name="cg * ω")],
          Layout(xaxis_title="frequency", yaxis_title="imag(Y)", yaxis_range=[-.001, .002]))
end

#######################################################
# Eigenmodes
#######################################################

function get_bare_qubit_freqs(num_points)
    f0 = ν/(4 * tline_params.L)
    f1 = ν/(2 * (tline_params.L - tline_params.x0))
    f2 = 3 * f0
    _eps = f0/10
    freqs_0 = range(1.5e9, stop=f0-_eps, length=num_points)
    freqs_1 = range(f0-_eps, stop=f0+_eps, length=num_points)
    freqs_2 = range(f0+_eps, stop=f1-_eps, length=num_points)
    freqs_3 = range(f1-_eps, stop=f1+_eps, length=num_points)
    freqs_4 = range(f1+_eps, stop=f2-_eps, length=num_points)
    freqs_5 = range(f2-_eps, stop=f2+_eps, length=num_points)
    freqs_6 = range(f2+_eps, stop=25e9, length=num_points)
    return [freqs_0; freqs_1; freqs_2; freqs_3; freqs_4; freqs_5; freqs_6]
end

function eigenmodes(lj::Real, pso::PSOModel)
    qubit = ParallelComponent("qubit", 1/lj, 0, cj)
    pso_full = connect(pso, PSOModel(qubit))
    eigenvalues, eigenvectors = lossy_modes_dense(pso_full, min_freq=1e9, max_freq=25e9)
    port_inds = ports_to_indices(pso_full, "qubit", "qr_coupler", "mode_2_max")
    mode_inds = AdmittanceModels.match_vectors(get_P(pso_full)[:, port_inds], eigenvectors)
    return eigenvalues[mode_inds]
end

function eigenmodes(casc::PSOModel, eff_capacitance::Real, ω::Vector{<:Real})
    pso, Y = terminated(casc), admittances(Blackbox(ω, casc))
    T1_Y = eff_capacitance ./ real.(Y)
    ljs = (1 ./ (ω .^2 * eff_capacitance))
    if show_results
        println("dimension: $(size(pso.K,1)), num_ljs: $(length(ljs))")
    end
    λ_lists = @showprogress [eigenmodes(lj, pso) for lj in ljs]
    λs = cat(λ_lists..., dims=2)
    return λs, T1_Y
end

ω = 2π * get_bare_qubit_freqs(run_fast ? 10 : 120)
tline_λs, tline_T1_Y = eigenmodes(tline_cascade_pso, tline_eff_capacitance, ω)

if show_results
    plot([scatter(x=ω/(2π*1e9), y=imag.(tline_λs[i,:])/(2π*1e9), name="$i",
        mode="lines+markers") for i in 1:size(tline_λs,1)],
        Layout(xaxis_title="bare qubit freq", yaxis_title="mode freq", title="tline freqs"))
end

if show_results
    traces = [scatter(x=ω/(2π*1e9), y=-.5 ./ real.(tline_λs[i,:]), name="$i",
               mode="lines+markers") for i in 1:size(tline_λs,1)]
    push!(traces, scatter(x=ω/(2π*1e9), y=tline_T1_Y, name="C/Re[Y]", mode="lines+markers"))
    plot(traces, Layout(xaxis_title="bare qubit freq", yaxis_title="T1", yaxis_type="log", title="tline T1s"))
end

if write_data
    df = DataFrame(transpose(tline_λs))
    df[:freqs] = ω/(2π)
    df[:T1_Y] = tline_T1_Y
    CSV.write(joinpath(output_folder, "tline_eigenmodes.csv"), df)
end

pfilter_λs, pfilter_T1_Y = eigenmodes(pfilter_cascade_pso, pfilter_eff_capacitance, ω)

if show_results
    plot([scatter(x=ω/(2π*1e9), y=imag.(pfilter_λs[i,:])/(2π*1e9), name="$i",
        mode="lines+markers") for i in 1:size(pfilter_λs,1)],
        Layout(xaxis_title="bare qubit freq", yaxis_title="mode freq", title="pfilter freqs"))
end

if show_results
    traces = [scatter(x=ω/(2π*1e9), y=-.5 ./ real.(pfilter_λs[i,:]), name="$i",
        mode="lines+markers") for i in 1:size(pfilter_λs,1)]
        push!(traces, scatter(x=ω/(2π*1e9), y=pfilter_T1_Y, name="C/Re[Y]", mode="lines+markers"))
        plot(traces, Layout(xaxis_title="bare qubit freq", yaxis_title="T1", yaxis_type="log", title="pfilter T1s"))
end

if write_data
    df = DataFrame(transpose(pfilter_λs))
    df[:freqs] = ω/(2π)
    df[:T1_Y] = pfilter_T1_Y
    CSV.write(joinpath(output_folder, "pfilter_eigenmodes.csv"), df)
end

#######################################################
# Convergence
#######################################################

freqs_func(vects, mode_ind, δ_ind) = imag.(vects[δ_ind][mode_ind,:])/(2π*1e9)
T1_func(vects, mode_ind, δ_ind) = -.5 ./ real.(vects[δ_ind][mode_ind,:])
function diff(vects, func, mode_ind, δ_ind_bad, δ_ind_good)
    f_bad = func(vects, mode_ind, δ_ind_bad)
    f_good = func(vects, mode_ind, δ_ind_good)
    return abs.((f_bad - f_good) ./ f_good) * 100
end

if !run_fast
    δs = [60, 50, 40] * 1e-6
    tline_λ_vects = [eigenmodes(create_tline_pso(δ), tline_eff_capacitance, ω)[1] for δ in δs]

    if write_data
        npzwrite(joinpath(output_folder, "tline_eigenvalues.npy"), cat(tline_λ_vects..., dims=3))
    end

    if show_results
        plot([scatter(x=ω/(2π*1e9), y=diff(tline_λ_vects, freqs_func, i, 1, 3), name="$i", mode="markers") for i in 1:3],
            Layout(xaxis_title="bare qubit freq", yaxis_title="% error", yaxis_type="log", title="tline freqs"))
    end

    if show_results
        plot([scatter(x=ω/(2π*1e9), y=diff(tline_λ_vects, T1_func, i, 2, 3), name="$i", mode="markers") for i in 1:3],
            Layout(xaxis_title="bare qubit freq", yaxis_title="% error", yaxis_type="log", title="tline freqs"))
    end

    pfilter_λ_vects = [eigenmodes(create_pfilter_pso(δ), pfilter_eff_capacitance, ω)[1] for δ in δs]

    if write_data
        npzwrite(joinpath(output_folder, "pfilter_eigenvalues.npy"), cat(pfilter_λ_vects..., dims=3))
    end

    if show_results
        plot([scatter(x=ω/(2π*1e9), y=diff(pfilter_λ_vects, freqs_func, i, 2, 3), name="$i", mode="markers") for i in 1:3],
            Layout(xaxis_title="bare qubit freq", yaxis_title="% error", yaxis_type="log", title="tline freqs"))
    end

    if show_results
        plot([scatter(x=ω/(2π*1e9), y=diff(pfilter_λ_vects, T1_func, i, 2, 3), name="$i", mode="markers") for i in 1:3],
            Layout(xaxis_title="bare qubit freq", yaxis_title="% error", yaxis_type="log", title="tline freqs"))
    end
end

#######################################################
# Declare victory
#######################################################

@testset "radiative loss" begin
    @test true
end
