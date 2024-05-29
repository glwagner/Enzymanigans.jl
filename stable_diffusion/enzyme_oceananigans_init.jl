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
Enzyme.API.maxtypeoffset!(3000)

function set_initial_condition!(model)
    amplitude = Ref(1.0)

    # This has a "width" of 0.1
    cᵢ(x, y, z) = amplitude[]
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

# Test differentiation of the high-level set interface
set_initial_condition!(model)

dmodel = Enzyme.make_zero(model)
autodiff(Enzyme.Reverse,
         set_initial_condition!,
         Duplicated(model, dmodel))