# Currently, running this requires
# - Enzyme#main (eg, > v0.11.11)

using Oceananigans
using Enzyme

Enzyme.API.runtimeActivity!(true)
# Enzyme.API.printall!(true)
# Enzyme.API.printactivity!(true)
Enzyme.API.looseTypeAnalysis!(true)
Enzyme.EnzymeRules.inactive_type(::Type{<:Oceananigans.Grids.AbstractGrid}) = true
Enzyme.EnzymeRules.inactive_type(::Type{<:Oceananigans.Clock}) = true
Enzyme.EnzymeRules.inactive_noinl(::typeof(Core._compute_sparams), args...) = nothing

Nx = Ny = 64
Nz = 8

x = y = (-π, π)
z = (-0.5, 0.5)
topology = (Periodic, Periodic, Bounded)

grid = RectilinearGrid(size=(Nx, Ny, Nz); x, y, z, topology)

# TODO:
# 1. Make the velocity fields evolve
# 2. Add surface fluxes
# 3. Do a problem where we invert for the tracer fluxes (maybe with CATKE)

model = HydrostaticFreeSurfaceModel(; grid,
                                    tracers = :c,
                                    buoyancy = nothing)

function set_initial_condition!(model, amplitude)
    # Set initial condition
    amplitude = Ref(amplitude)

    # This has a "width" of 0.1
    cᵢ(x, y, z) = amplitude[]
    set!(model, c=cᵢ)

    return nothing
end


# Now for real
amplitude = 1.0
dmodel = Enzyme.make_zero(model)
@show model.tracers.c
@show typeof(model.tracers.c)
#=
dc²_dκ = autodiff(Enzyme.Reverse,
                  set_initial_condition!,
                  Duplicated(model, dmodel),
                  Const(amplitude))

@info """ \n
Enzyme computed $dc²_dκ
"""
=#