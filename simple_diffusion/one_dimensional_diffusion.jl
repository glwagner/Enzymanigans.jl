using Oceananigans

grid = RectilinearGrid(size=128, z=(-0.5, 0.5), topology=(Flat, Flat, Bounded))
closure = ScalarDiffusivity(κ=1)
model = NonhydrostaticModel(; grid, closure, tracers=:T)

width = 0.1
initial_temperature(x, y, z) = exp(-z^2 / (2width^2))
set!(model, T=initial_temperature)

## Time-scale for diffusion across a grid cell
min_Δz = minimum_zspacing(model.grid)
diffusion_time_scale = min_Δz^2 / model.closure.κ.T
Δt = 0.1 * diffusion_time_scale

T = model.tracers.T
@show maximum(T)

for n = 1:1000
    time_step!(model, Δt)
end

@show maximum(T)
