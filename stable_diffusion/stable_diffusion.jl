using Oceananigans
using Enzyme

using Oceananigans.Fields

Enzyme.API.runtimeActivity!(true)
Enzyme.API.printall!(true)

grid = RectilinearGrid(size=128, z=(-0.5, 0.5), topology=(Flat, Flat, Bounded))
diffusion = VerticalScalarDiffusivity(κ=1.0)

model = HydrostaticFreeSurfaceModel(; grid,
                                    tracers = :c,
                                    buoyancy = nothing,
                                    velocities = PrescribedVelocityFields(),
                                    closure = diffusion)

function reduced_fill_halo_regions!(ordinary_fields)
    bcs  = reduced_permute_boundary_conditions(map(boundary_conditions, ordinary_fields))
    Base.inferencebarrier(noop)(bcs)    
    return nothing
end

function boundary_conditions(f::Field)
    return f.boundary_conditions
end

function reduced_permute_boundary_conditions(boundary_conditions)
    sides = (:west_and_east, :bottom_and_top)
    boundary_conditions = Tuple(extract_bc(boundary_conditions, Val(side)) for side in sides)

    return boundary_conditions[1]
end

@inline extract_bc(bc, ::Val{:west_and_east})   = ((nothing,))
@inline function extract_bc(bc, ::Val{:bottom_and_top})
    res = bc[1].bottom
    return res
end

noop(c) = nothing

model = Oceananigans.prognostic_fields(model)
model = (model.c,)
dmodel = deepcopy(model)

reduced_fill_halo_regions!(model)

autodiff(Reverse, reduced_fill_halo_regions!, Duplicated(model, dmodel))
