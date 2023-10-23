using Oceananigans
using Oceananigans.TurbulenceClosures: with_tracers
using Oceananigans.Models.HydrostaticFreeSurfaceModels: tracernames
using Enzyme

Enzyme.EnzymeRules.inactive_type(::Type{<:Oceananigans.Grids.AbstractGrid}) = true

const maximum_diffusivity = 100

grid = RectilinearGrid(size=128, z=(-0.5, 0.5), topology=(Flat, Flat, Bounded))
diffusion = VerticalScalarDiffusivity(κ=1)

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
    FT = eltype(model.grid)
    tracers = model.tracers
    closure = VerticalScalarDiffusivity(; κ=diffusivity)
    closure = with_tracers(tracernames(tracers), closure)
    model.closure = closure
    return nothing
end

function stable_diffusion!(model, amplitude, diffusivity)
    set_diffusivity!(model, diffusivity)

    # Set initial condition
    width = 0.1
    cᵢ(x, y, z) = amplitude * exp(-z^2 / (2width^2))
    set!(model, c=cᵢ)

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
    c² = c^2
    sum_c² = sum(c²)

    # Another way to compute it
    # c² = Array(interior(c).^2)
    # sum_c² = sum(c²)

    return sum_c²::Float64
end

# Compute derivative by hand
κ₁, κ₂ = 0.9, 1.1
c²₁ = stable_diffusion!(model, 1, κ₁)
c²₂ = stable_diffusion!(model, 1, κ₂)
dc²_dκ = (c²₂ - c²₁) / (κ₂ - κ₁)

# Now for real
amplitude = 1
dmodel = deepcopy(model)
set_diffusivity!(dmodel, 0)

autodiff(Reverse, stable_diffusion!, Duplicated(model, dmodel), Const(amplitude), Active(diffusivity))

