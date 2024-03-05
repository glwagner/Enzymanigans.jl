# Currently, running this requires
# - Enzyme#main (eg, > v0.11.11)

using Oceananigans
using Oceananigans.TurbulenceClosures: with_tracers
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: ConstantField
using Oceananigans.Models.HydrostaticFreeSurfaceModels: tracernames
using Oceananigans.Utils: apply_regionally!
using Enzyme

Enzyme.API.runtimeActivity!(true)
# Enzyme.API.printall!(true)
# Enzyme.API.printactivity!(true)
Enzyme.API.looseTypeAnalysis!(true)
Enzyme.EnzymeRules.inactive_type(::Type{<:Oceananigans.Grids.AbstractGrid}) = true
Enzyme.EnzymeRules.inactive_type(::Type{<:Oceananigans.Clock}) = true
Enzyme.EnzymeRules.inactive_noinl(::typeof(Core._compute_sparams), args...) = nothing

const maximum_diffusivity = 100

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
                                    tracer_advection = WENO(),
                                    tracers = :c,
                                    buoyancy = nothing)

function set_initial_condition!(model, amplitude)
    # Set initial condition
    amplitude = Ref(amplitude)

    # This has a "width" of 0.1
    cᵢ(x, y, z) = amplitude[]
    set!(model.tracers.c, cᵢ)

    return nothing
end

# Now for real
amplitude = 1.0
κ = 1.0
dmodel = Enzyme.make_zero(model)

autodiff(Enzyme.Reverse, set_initial_condition!, Duplicated(model, dmodel), Const(amplitude))
