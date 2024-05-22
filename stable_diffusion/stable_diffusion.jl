# Currently, running this requires
# - Enzyme#main (eg, > v0.11.11)

using Oceananigans
using Oceananigans.TurbulenceClosures: with_tracers
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: ConstantField
using Oceananigans.Models.HydrostaticFreeSurfaceModels: tracernames
using Enzyme

Enzyme.API.runtimeActivity!(true)
#Enzyme.API.printall!(true)
#Enzyme.API.printactivity!(true)
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
diffusion = VerticalScalarDiffusivity(κ=0.1)

u = XFaceField(grid)
v = YFaceField(grid)

U = 1
u₀(x, y, z) = - U * cos(x + π/4) * sin(y) * (z + 0.5)
v₀(x, y, z) = + U * sin(x + π/4) * cos(y) * (z + 0.5)

set!(u, u₀)
set!(v, v₀)
fill_halo_regions!(u)
fill_halo_regions!(v)

@inline function tracer_flux(x, y, t, c, p)
    c₀ = p.surface_tracer_concentration
    u★ = p.piston_velocity
    return - u★ * (c₀ - c)
end

parameters = (surface_tracer_concentration = 1,
              piston_velocity = 0.1)

top_c_bc = FluxBoundaryCondition(tracer_flux, field_dependencies=:c; parameters)
c_bcs = FieldBoundaryConditions(top=top_c_bc)

# TODO:
# 1. Make the velocity fields evolve
# 2. Add surface fluxes
# 3. Do a problem where we invert for the tracer fluxes (maybe with CATKE)

model = HydrostaticFreeSurfaceModel(; grid,
                                    tracer_advection = WENO(),
                                    tracers = :c,
                                    buoyancy = nothing,
                                    velocities = PrescribedVelocityFields(; u, v)
                                    boundary_conditions = (; c=c_bcs))

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

dc²_dκ = autodiff(Enzyme.Reverse,
                  set_initial_condition!,
                  Duplicated(model, dmodel),
                  Const(amplitude))
