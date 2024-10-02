using Enzyme
# Required presently
#Enzyme.API.runtimeActivity!(true)
Enzyme.API.looseTypeAnalysis!(true)
Enzyme.API.maxtypeoffset!(2032)

using Oceananigans
using Oceananigans.TurbulenceClosures: with_tracers
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: ConstantField
using Oceananigans.Models.HydrostaticFreeSurfaceModels: tracernames
using Oceananigans.Fields: FunctionField
using Oceananigans: architecture
using KernelAbstractions


#EnzymeRules.inactive_type(::Type{<:Oceananigans.Clock}) = true

f(grid) = CenterField(grid)

const maximum_diffusivity = 100

function momentum_equation!(model)
    
    # Do time-stepping
    Nx, Ny, Nz = size(model.grid)
    Δz = 1 / Nz
    Δt = 1e-1 * Δz^2

    model.clock.time = 0
    model.clock.iteration = 0

    for _ = 1:100
        time_step!(model, Δt; euler=true)
    end

    # Compute scalar metric
    u = model.velocities.u

    # Hard way (for enzyme - the sum function sometimes errors with AD)
    # c² = c^2
    # sum_c² = sum(c²)

    # Another way to compute it
    sum_u² = 0.0
    for k = 1:Nz, j = 1:Ny,  i = 1:Nx
        sum_u² += u[i, j, k]^2
    end

    # Need the ::Float64 for type inference with automatic differentiation
    return sum_u²::Float64
end

Nx = Ny = 32
Nz = 4

Lx = Ly = L = 2π
Lz = 1

x = y = (-L/2, L/2)
z = (-Lz, 0)
topology = (Periodic, Periodic, Bounded)

grid = RectilinearGrid(size=(Nx, Ny, Nz); x, y, z, topology)

u = XFaceField(grid)
v = YFaceField(grid)

U = 1
u₀(x, y, z) = - U * cos(x + L/8) * sin(y) * (-z)
v₀(x, y, z) = + U * sin(x + L/8) * cos(y) * (-z)

set!(u, u₀)
set!(v, v₀)
fill_halo_regions!(u)
fill_halo_regions!(v)

# TODO:
# 1. Make the velocity fields evolve
# 2. Add surface fluxes
# 3. Do a problem where we invert for the tracer fluxes (maybe with CATKE)

buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState())

model = HydrostaticFreeSurfaceModel(; grid,
                                    momentum_advection = WENO(),
                                    #buoyancy = buoyancy,
                                    #tracers = (:T, :S),
                                    #velocities = PrescribedVelocityFields(; u, v),
                                    closure = nothing) #ScalarBiharmonicDiffusivity())

#set!(model, S = 34.7, T = 0.5)
set!(model, u=u₀, v=v₀)

#=
# Compute derivative by hand
κ₁, κ₂ = 0.9, 1.1
u²₁ = momentum_equation!(model, 1, κ₁)
u²₂ = momentum_equation!(model, 1, κ₂)
du²_dκ_fd = (u²₂ - u²₁) / (κ₂ - κ₁)
=#

# Now for real
dmodel = Enzyme.make_zero(model)

u_old = model.velocities.u[:]
@show model.velocities.u
@show dmodel.velocities.u

momentum_equation!(model)

#du²_dκ = autodiff(set_runtime_activity(Enzyme.Reverse),
#                  momentum_equation!,
#                  Duplicated(model, dmodel))

u_new = model.velocities.u[:]
@show model.velocities.u
@show dmodel.velocities.u

#@show u_new - u_old

#=
@info """ \n
Momentum Equation:
Enzyme computed $du²_dκ
Finite differences computed $du²_dκ_fd
"""
=#