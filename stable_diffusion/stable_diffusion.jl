using Oceananigans
using Oceananigans.TurbulenceClosures: with_tracers
using Oceananigans.Models.HydrostaticFreeSurfaceModels: tracernames
using Enzyme

Enzyme.API.runtimeActivity!(true)
# Enzyme.API.printall!(true)
# Enzyme.API.printactivity!(true)
Enzyme.EnzymeRules.inactive_type(::Type{<:Oceananigans.Grids.AbstractGrid}) = true
Enzyme.EnzymeRules.inactive_type(::Type{<:Oceananigans.Clock}) = true
Enzyme.EnzymeRules.inactive_noinl(::typeof(Core._compute_sparams), args...) = nothing

const maximum_diffusivity = 100

grid = RectilinearGrid(size=128, z=(-0.5, 0.5), topology=(Flat, Flat, Bounded))
diffusion = VerticalScalarDiffusivity(κ=1.0)

model = HydrostaticFreeSurfaceModel(; grid,
                                    tracers = :c,
                                    buoyancy = nothing,
                                    velocities = PrescribedVelocityFields(),
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

function set_initial_condition!(model, amplitude)
    # Set initial condition
    width = 0.1
    cᵢ(z) = amplitude * exp(-z^2 / (2width^2))
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
    c² = Array(interior(c).^2)
    sum_c² = sum(c²)

    return sum_c²::Float64
end

@show(model)

# Compute derivative by hand
κ₁, κ₂ = 0.9, 1.1
c²₁ = stable_diffusion!(model, 1, κ₁)
c²₂ = stable_diffusion!(model, 1, κ₂)
dc²_dκ = (c²₂ - c²₁) / (κ₂ - κ₁)

# Now for real
amplitude = 1.0
κ = 1.0
dmodel = deepcopy(model)
set_diffusivity!(dmodel, 0)

@show(model)
@show(c²₁)
@show(c²₂)

autodiff(Reverse, set_initial_condition!, Duplicated(model, dmodel), Active(amplitude))
#autodiff(Reverse, set_diffusivity!, Duplicated(model, dmodel), Active(κ))
#autodiff(Reverse, stable_diffusion!, Duplicated(model, dmodel), Const(amplitude), Active(κ))

