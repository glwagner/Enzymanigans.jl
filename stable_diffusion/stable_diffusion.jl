# Currently, running this requires
# - Enzyme#main (eg, > v0.11.11)
# - Oceananigans#glw/type-stable-with-tracers
# - KernelAbstractions#enzymeact

using Oceananigans
using Oceananigans.TurbulenceClosures: with_tracers
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: ConstantField
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
diffusion = VerticalScalarDiffusivity(κ=0.1)

u = XFaceField(grid)
v = YFaceField(grid)

# ψ(x, y) = cos(x) * cos(y)
# u = - ∂y ψ = - cos(x) * sin(y)
# v = + ∂x ψ = + sin(x) * cos(y)
# ... and scale with U:
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
                                    velocities = PrescribedVelocityFields(; u, v),
                                    boundary_conditions = (; c=c_bcs),
                                    closure = diffusion)

#=
L = 0.5
cᵢ(x, y, z) = exp(-(x^2 + y^2) / 2L^2)
set!(model, c=cᵢ)
Δt = U * 0.2π / Nx
simulation = Simulation(model; Δt, stop_time = 1)
progress(sim) = @info string("Iter: ", iteration(sim), ", time: ", time(sim))
simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))
run!(simulation)

c = model.tracers.c
using GLMakie
fig = Figure()
axu = Axis(fig[1, 1])
axv = Axis(fig[1, 2])
axc = Axis(fig[1, 3])
heatmap!(axu, interior(u, :, :, 1))
heatmap!(axv, interior(v, :, :, 1))
heatmap!(axc, interior(c, :, :, 1))
display(fig)
=#

"""
    set_diffusivity!(model, diffusivity)

Change diffusivity of model to `diffusivity`.
"""
function set_diffusivity!(model, diffusivity)
    closure = VerticalScalarDiffusivity(; κ=diffusivity)
    names = tuple(:c) # tracernames(model.tracers)
    closure = with_tracers(names, closure)
    model.closure = closure
    return nothing
end

function set_initial_condition!(model, amplitude)
    # Set initial condition
    amplitude = Ref(amplitude)

    # This has a "width" of 0.1
    cᵢ(x, y, z) = amplitude[] * exp(-z^2 / 0.02 - (x^2 + y^2) / 0.05)
    set!(model, c=cᵢ)

    return nothing
end

function stable_diffusion!(model, amplitude, diffusivity)
    set_diffusivity!(model, diffusivity)
    set_initial_condition!(model, amplitude)
    
    # Do time-stepping
    Nx, Ny, Nz = size(model.grid)
    κ_max = maximum_diffusivity
    Δz = 2π / Nz
    Δt = 1e-1 * Δz^2 / κ_max

    model.clock.time = 0
    model.clock.iteration = 0

    for n = 1:10
        time_step!(model, Δt; euler=true)
    end

    # Compute scalar metric
    c = model.tracers.c

    # Hard way
    # c² = c^2
    # sum_c² = sum(c²)

    # Another way to compute it
    sum_c² = 0.0
    for i = 1:Nx, j = 1:Ny, k = 1:Nz
        sum_c² += c[i, j, k]^2
    end

    return sum_c²::Float64
end

# Compute derivative by hand
κ₁, κ₂ = 0.9, 1.1
c²₁ = stable_diffusion!(model, 1, κ₁)
c²₂ = stable_diffusion!(model, 1, κ₂)
dc²_dκ_fd = (c²₂ - c²₁) / (κ₂ - κ₁)

# Now for real
amplitude = 1.0
κ = 1.0
dmodel = Enzyme.make_zero(model)
set_diffusivity!(dmodel, 0)

#autodiff(Reverse, set_initial_condition!, Duplicated(model, dmodel), Active(amplitude))
#autodiff(Reverse, set_diffusivity!, Duplicated(model, dmodel), Active(κ))

dc²_dκ = autodiff(Enzyme.Reverse,
                  stable_diffusion!,
                  Duplicated(model, dmodel),
                  Const(amplitude),
                  Active(κ))

@info """ \n
Enzyme computed $dc²_dκ
Finite differences computed $dc²_dκ_fd
"""
