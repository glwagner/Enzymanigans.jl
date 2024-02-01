# Currently, running this requires
# - Enzyme#main (eg, > v0.11.11)
# - Oceananigans#glw/type-stable-with-tracers
# - KernelAbstractions#enzymeact

using Oceananigans
using Oceananigans: prognostic_fields
using Oceananigans.Models.HydrostaticFreeSurfaceModels: tracernames
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

u = XFaceField(grid)
v = YFaceField(grid)

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
                                    tracers = :c,
                                    buoyancy = nothing,
                                    velocities = PrescribedVelocityFields(; u, v),
                                    boundary_conditions = (; c=c_bcs))

function set_initial_condition!(maybe_nested_tuple)

    fields = flatten_tuple(Tuple(tuplify(ai) for ai in maybe_nested_tuple))
    
    # Fill the rest
    bc = map(boundary_conditions, fields)
    
    sides = [:bottom_and_top]
    bc = Tuple((map(extract_bottom_bc, bc), map(extract_top_bc, bc)) for side in sides)

    return nothing
end

function boundary_conditions(f::Field)
    return f.boundary_conditions
end

@inline extract_bottom_bc(thing) = thing.bottom
@inline extract_top_bc(thing) = thing.top

# Utility for extracting values from nested NamedTuples
@inline tuplify(a::NamedTuple) = Tuple(tuplify(ai) for ai in a)
@inline tuplify(a) = a

# Outer-inner form
@inline flatten_tuple(a::Tuple) = tuple(inner_flatten_tuple(a[1])..., inner_flatten_tuple(a[2:end])...)
@inline flatten_tuple(a::Tuple{<:Any}) = tuple(inner_flatten_tuple(a[1])...)

@inline inner_flatten_tuple(a) = tuple(a)
@inline inner_flatten_tuple(a::Tuple) = flatten_tuple(a)
@inline inner_flatten_tuple(a::Tuple{}) = ()

# Now for real
dmodel = Enzyme.make_zero(model)

#@show prognostic_fields(model)[1]
#@show c_bcs


dc²_dκ = autodiff(Enzyme.Reverse,
                  set_initial_condition!,
                  Duplicated(prognostic_fields(model), prognostic_fields(dmodel)))
