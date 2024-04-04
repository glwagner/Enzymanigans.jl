# Currently, running this requires
# - Enzyme#main (eg, > v0.11.11)

using Oceananigans
using Oceananigans.TurbulenceClosures: with_tracers
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: ConstantField
using Oceananigans.Models.HydrostaticFreeSurfaceModels: tracernames
using Enzyme
using LinearAlgebra

using CairoMakie

Enzyme.API.runtimeActivity!(true)
Enzyme.API.looseTypeAnalysis!(true)
Enzyme.EnzymeRules.inactive_type(::Type{<:Oceananigans.Grids.AbstractGrid}) = true
Enzyme.EnzymeRules.inactive_type(::Type{<:Oceananigans.Clock}) = true
Enzyme.EnzymeRules.inactive_noinl(::typeof(Core._compute_sparams), args...) = nothing

const maximum_diffusivity = 100

Nx = Ny = 64
Nz = 8

x = y = (-π, π)
z = (-0.5, 0.5)
topology = (Bounded, Bounded, Bounded)

grid = RectilinearGrid(size=(Nx, Ny, Nz); x, y, z, topology)
diffusion = VerticalScalarDiffusivity(κ=0.1)

u = XFaceField(grid)
v = YFaceField(grid)

U = 40
#u₀(x, y, z) = - U * 0.5# * (z + 0.5)#- U * cos(x + π/4) * sin(y) * (z + 0.5)

u₀(x, y, z) = - U * cos(x + π/4) * sin(y)
v₀(x, y, z) = + U * 0.5

# * (z + 0.5)

set!(u, u₀)
set!(v, v₀)
fill_halo_regions!(u)
fill_halo_regions!(v)

# TODO:
# 1. Make the velocity fields evolve
# 2. Add surface fluxes
# 3. Do a problem where we invert for the tracer fluxes (maybe with CATKE)

model = HydrostaticFreeSurfaceModel(; grid,
                                    tracer_advection = WENO(),
                                    tracers = :c,
                                    buoyancy = nothing,
                                    velocities = PrescribedVelocityFields(; u, v),
                                    closure = diffusion)

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

function set_initial_data!(model)
    # Set initial condition
    #cᵢ = zeros(size(model.tracers.c))
    #cᵢ[31:33, 31:33, 1:3] .= 10
    amplitude = Ref(1)
    cᵢ(x, y, z) = amplitude[] * exp(-z^2 / 0.02 - (x^2 + y^2) / 0.05) .+ (0.001 .* (rand()))
    set!(model, c=cᵢ)

    return nothing
end

# Generates the "real" data from a stable diffusion run:
function stable_diffusion_data!(model, diffusivity, n_max)
    
    set_diffusivity!(model, diffusivity)
    set_initial_data!(model)
    
    # Do time-stepping
    Nx, Ny, Nz = size(model.grid)
    κ_max = maximum_diffusivity
    Δz = 2π / Nz
    Δt = 1e-1 * Δz^2 / κ_max

    model.clock.time = 0
    model.clock.iteration = 0

    c₀ = deepcopy(model.tracers.c)

    for n = 1:n_max
        time_step!(model, Δt; euler=true)
    end

    # Compute scalar metric
    cₙ = deepcopy(model.tracers.c)

    return c₀, cₙ
end

function set_initial_condition!(model, cᵢ)

    # This has a "width" of 0.1
    #cᵢ(x, y, z) = amplitude[] * exp(-z^2 / 0.02 - (x^2 + y^2) / 0.05)
    set!(model, c=cᵢ)

    return nothing
end

# cᵢ is the proposed initial condition, cₙ is the actual data collected
# at the final time step:
function advection_diffusion_model!(model, diffusivity, n_max, cᵢ, cₙ)
    set_diffusivity!(model, diffusivity)
    set_initial_condition!(model, cᵢ)
    
    # Run the forward model:
    Nx, Ny, Nz = size(model.grid)
    κ_max = maximum_diffusivity
    Δz = 2π / Nz
    Δt = 1e-1 * Δz^2 / κ_max

    model.clock.time = 0
    model.clock.iteration = 0

    for n = 1:n_max
        time_step!(model, Δt; euler=true)
    end

    c = model.tracers.c
    # Compute the misfit of our forward model run with the true data cₙ:
    J = 0.0
    for i = 1:Nx, j = 1:Ny, k = 1:Nz
        J += (c[i, j, k] - cₙ[i, j, k])^2
    end

    return J::Float64
end

κ = 1
n_max  = 80
c₀, cₙ = stable_diffusion_data!(model, κ, n_max)

@show c₀
@show cₙ

cᵢ = zeros(size(model.tracers.c))
#cᵢ[29:35, 29:35, 1:3] .= 5

learning_rate = 0.01
max_steps = 300
δ = 0.01

# Update our guess of the initial tracer distribution, cᵢ:
for i = 1:max_steps
    dmodel = Enzyme.make_zero(model)
    dcᵢ = Enzyme.make_zero(cᵢ)
    dcₙ = Enzyme.make_zero(cₙ)
    set_diffusivity!(dmodel, 0)

    # Since we are only interested in duplicated variable cᵢ for this run,
    # we do not use dJ here:
    dJ = autodiff(Enzyme.Reverse,
                    advection_diffusion_model!,
                    Duplicated(model, dmodel),
                    Const(κ),
                    Const(n_max),
                    Duplicated(cᵢ, dcᵢ),
                    Duplicated(cₙ, dcₙ))

    @show i
    @show norm(dcᵢ)
    global cᵢ .= cᵢ .- (dcᵢ .* learning_rate)
    @show (norm(cᵢ - c₀) / norm(c₀))

    # Stop gradient descent if dcᵢ is sufficiently small:
    if norm(dcᵢ) < δ
        break
    end
end

#heatmap(interior(c₀, :, :, 1))
#heatmap(interior(cₙ, :, :, 1))

fig = Figure(size = (1600, 800))

axis_kwargs = (xlabel="x",
               ylabel="y",
               aspect = AxisAspect(grid.Lx/grid.Ly),
               limits = ((-grid.Lx/2, grid.Lx/2), (-grid.Ly/2, grid.Ly/2)))

# Coordinate arrays
srf_x, srf_y, srf_z = nodes(c₀)

ax_0  = Axis(fig[1, 1]; title = "Initial Tracer Distribution", axis_kwargs...)
ax_n  = Axis(fig[1, 3]; title = "Final Tracer Distribution", axis_kwargs...)
ax_i  = Axis(fig[1, 5]; title = "Inverted Tracer Distribution", axis_kwargs...)

hm_0 = heatmap!(ax_0, srf_x, srf_y, interior(c₀, :, :, 4))
Colorbar(fig[1, 2], hm_0; label = "concentration")
hm_n = heatmap!(ax_n, srf_x, srf_y, interior(cₙ, :, :, 4))
Colorbar(fig[1, 4], hm_n; label = "concentration")
hm_n = heatmap!(ax_i, srf_x, srf_y, cᵢ[:, :, 4])
Colorbar(fig[1, 6], hm_n; label = "concentration")

save("tracer_distribution.png", fig)