# Currently, running this requires
# - Enzyme#main (eg, > v0.11.11)

using Oceananigans
using Oceananigans.TurbulenceClosures: with_tracers
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: ConstantField
using Oceananigans.Models.HydrostaticFreeSurfaceModels: tracernames
using Enzyme
using LinearAlgebra

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


cᵢ = zeros(size(model.tracers.c))
cᵢ[32, 32, 2] = 1

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

function set_initial_condition!(model, amplitude, cᵢ)
    # Set initial condition
    amplitude = Ref(amplitude)

    # This has a "width" of 0.1
    #cᵢ(x, y, z) = amplitude[] * exp(-z^2 / 0.02 - (x^2 + y^2) / 0.05)
    set!(model, c=cᵢ)

    return nothing
end

function set_initial_data!(model, amplitude)#, cᵢ)
    # Set initial condition
    amplitude = Ref(amplitude)

    # This has a "width" of 0.1
    cᵢ(x, y, z) = amplitude[] * exp(-z^2 / 0.02 - (x^2 + y^2) / 0.05) .+ (0.025 * (rand()))
    #cᵢ = cᵢ .+ 
    set!(model, c=cᵢ)

    return nothing
end

# Generates the "real" data from a stable diffusion run:
function stable_diffusion_data!(model, amplitude, diffusivity, cᵢ)
    
    set_diffusivity!(model, diffusivity)
    set_initial_data!(model, amplitude)#, cᵢ)
    
    # Do time-stepping
    Nx, Ny, Nz = size(model.grid)
    κ_max = maximum_diffusivity
    Δz = 2π / Nz
    Δt = 1e-1 * Δz^2 / κ_max

    model.clock.time = 0
    model.clock.iteration = 0

    c₀ = deepcopy(model.tracers.c)

    for n = 1:10
        time_step!(model, Δt; euler=true)
    end

    # Compute scalar metric
    cₙ = deepcopy(model.tracers.c)

    return c₀, cₙ
end

function stable_diffusion!(model, amplitude, diffusivity, cᵢ, c_data)
    set_diffusivity!(model, diffusivity)
    set_initial_condition!(model, amplitude, cᵢ)
    
    # Do time-stepping
    Nx, Ny, Nz = size(model.grid)
    κ_max = maximum_diffusivity
    Δz = 2π / Nz
    Δt = 1e-1 * Δz^2 / κ_max

    model.clock.time = 0
    model.clock.iteration = 0

    for n = 1:100
        time_step!(model, Δt; euler=true)
    end

    # Compute scalar metric
    c = model.tracers.c
    #@show c

    # Another way to compute it
    J = 0.0
    for i = 1:Nx, j = 1:Ny, k = 1:Nz
        J += (c[i, j, k] - c_data[i, j, k])^2
    end

    return J::Float64
end

# First, let's compute some "real" data with added noise
amplitude = 1.0
κ = 1
c₀, cₙ = stable_diffusion_data!(model, amplitude, κ, cᵢ)
#J = stable_diffusion!(model, amplitude, κ, c_data)

@show c₀
@show cₙ
#@show J

#=
# Compute ∂J / ∂κ by hand:
κ₁, κ₂ = κ - 0.01, κ + 0.01
J¹ = stable_diffusion!(model, amplitude, κ₁, cᵢ, cₙ)
J² = stable_diffusion!(model, amplitude, κ₂, cᵢ, cₙ)
dJ_dκ_fd = (J² - J¹) / (κ₂ - κ₁)
@show dJ_dκ_fd
=#


dmodel = Enzyme.make_zero(model)
dcᵢ = Enzyme.make_zero(cᵢ)
dcₙ = Enzyme.make_zero(cₙ)
set_diffusivity!(dmodel, 0)

# I believe this snags the center x/y/z values of each grid cell:
@show nodes(grid, (Center(), Center(), Center()))

c = model.tracers.c
@show c
@show size(c)


cᵢ = zeros(size(model.tracers.c))
#cᵢ[32, 32, 2] = 5
#amplitude = 4
#κ = 9
for i = 1:40
    dJ = autodiff(Enzyme.Reverse,
                    stable_diffusion!,
                    Duplicated(model, dmodel),
                    Const(amplitude),
                    Const(κ),
                    Duplicated(cᵢ, dcᵢ),
                    Duplicated(cₙ, dcₙ))

    #@show dJ
    @show norm(dcᵢ)
    #global amplitude = amplitude - dJ[1][2] * 0.02
    #global κ = κ - dJ[1][3]
    #@show amplitude
    #@show κ
    global cᵢ .= cᵢ .- (dcᵢ .* 0.02)
    @show (norm(cᵢ - c₀) / norm(c₀))
    # TODO: show how this cᵢ compares to the true-data version using norm error
end

#=
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

dc²_dκ = autodiff(Enzyme.Reverse,
                  stable_diffusion!,
                  Duplicated(model, dmodel),
                  Const(amplitude),
                  Active(κ))

@info """ \n
Enzyme computed $dc²_dκ
Finite differences computed $dc²_dκ_fd
"""

@show dmodel.tracers.c
=#