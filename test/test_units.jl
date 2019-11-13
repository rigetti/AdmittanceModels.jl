using Unitful

c = Circuit(zeros(typeof(1.0/u"nH"), 3, 3),
            zeros(typeof(1.0/u"Ω"), 3, 3),
            zeros(typeof(1.0u"fF"), 3, 3), 0:2)

set_inductance!(c, 0, 1, 10u"nH")
set_inductance!(c, 0, 2, 10u"nH")
set_capacitance!(c, 0, 1, 100u"fF")
set_capacitance!(c, 0, 2, 110u"fF")
set_capacitance!(c, 1, 2, 1u"fF")

pso = PSOModel(c, [(0,1), (0,2)], ["q1", "q2"])
bbox = Blackbox(collect(1:0.1:10)u"GHz", pso)

@test dimension(eltype(eltype(bbox.Y))) == dimension(u"Ω^-1")
