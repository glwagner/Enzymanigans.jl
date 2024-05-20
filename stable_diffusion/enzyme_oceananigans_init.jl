using Oceananigans
using Enzyme
using Oceananigans.Fields: FunctionField
using Oceananigans: architecture
using KernelAbstractions

# Required presently
Enzyme.API.runtimeActivity!(true)

EnzymeRules.inactive_type(::Type{<:Oceananigans.Grids.AbstractGrid}) = true
EnzymeRules.inactive_type(::Type{<:Oceananigans.Clock}) = true

Enzyme.API.looseTypeAnalysis!(true)

function set_initial_condition!(model, cᵢ)
    set!(model, c=cᵢ)
    return nothing
end

Nx = Ny = 64
Nz = 8

x = y = (-π, π)
z = (-0.5, 0.5)
topology = (Periodic, Periodic, Bounded)

grid = RectilinearGrid(size=(Nx, Ny, Nz); x, y, z, topology)

model = HydrostaticFreeSurfaceModel(; grid,
                                        tracers = :c,
                                        buoyancy = nothing)


amplitude   = Ref(amplitude)
cᵢ(x, y, z) = amplitude[]

# Test differentiation of the high-level set interface
dmodel = Enzyme.make_zero(model)
autodiff(Enzyme.Reverse,
                set_initial_condition!,
                Duplicated(model, dmodel),
                Const(cᵢ))