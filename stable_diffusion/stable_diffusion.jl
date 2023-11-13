using Oceananigans
using Oceananigans.TurbulenceClosures: with_tracers
using Oceananigans.Models.HydrostaticFreeSurfaceModels: tracernames
using Enzyme

using Oceananigans.Fields
using Oceananigans.Fields: default_indices
using Oceananigans.BoundaryConditions
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, regularize_field_boundary_conditions
using Oceananigans: instantiated_location

using Oceananigans.BoundaryConditions: extract_bottom_bc

using OffsetArrays: OffsetArray

include("../../FlattenedTuples/FlattenedTuples.jl")
import .FlattenedTuples: reduced_flattened_unique_values

Enzyme.API.runtimeActivity!(true)
#Enzyme.API.printall!(true)
Enzyme.EnzymeRules.inactive_type(::Type{<:Oceananigans.Grids.AbstractGrid}) = true
Enzyme.EnzymeRules.inactive_noinl(::typeof(Core._compute_sparams), args...) = nothing
#=
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

function reduced_fill_halo_regions!(maybe_nested_tuple::Union{NamedTuple, Tuple})

    #ordinary_fields = maybe_nested_tuple
    # THIS function call (flattened_unique_values) leads to the seg fault, if we just set ordinary_fields = maybe_nested_tuple
    # then no errors
    ordinary_fields = reduced_flattened_unique_values(maybe_nested_tuple)

    @show ordinary_fields
    @show typeof(ordinary_fields)
    
    fill_halos! = [fill_west_and_east_halo!, fill_west_and_east_halo!]
    bcs  = reduced_permute_boundary_conditions(map(boundary_conditions, ordinary_fields))
    number_of_tasks   = length(fill_halos!)

    # Fill halo in the three permuted directions (1, 2, and 3), making sure dependencies are fulfilled
    for task = 1:number_of_tasks
        fill_halos![task](map(data, ordinary_fields), bcs[task])
    end
    
    return nothing
end

function boundary_conditions(f::Field)
    return f.boundary_conditions
end

function reduced_permute_boundary_conditions(boundary_conditions)
    #=
    @show boundary_conditions
    @show typeof(boundary_conditions)
    @show extract_bottom_bc(boundary_conditions)
    @show typeof(extract_bottom_bc(boundary_conditions))
    =#
    sides       = [:west_and_east, :bottom_and_top]
    boundary_conditions = Tuple(extract_bc(boundary_conditions, Val(side)) for side in sides)

    return boundary_conditions
end

@inline extract_bc(bc, ::Val{:west_and_east})   = ((nothing,))
@inline extract_bc(bc, ::Val{:bottom_and_top})  = (extract_bottom_bc(bc))

#@inline extract_bc(bc, ::Val{:bottom_and_top})  = ((ReducedBoundaryCondition(Flux(), nothing),), (ReducedBoundaryCondition(Flux(), nothing),))
fill_west_and_east_halo!(c, west_bc) = 0

#=
abstract type AbstractBoundaryConditionClassification end
struct Flux <: AbstractBoundaryConditionClassification end
struct ReducedBoundaryCondition{C<:AbstractBoundaryConditionClassification, T}
    classification :: C
    condition :: T
end
@show Oceananigans.prognostic_fields(model)
@show typeof(Oceananigans.prognostic_fields(model))

@show model.closure
@show typeof(model.closure)
=#
# Now for real
dmodel = deepcopy(model)
set_diffusivity!(dmodel, 0)

#reduced_fill_halo_regions!(Oceananigans.prognostic_fields(model))

autodiff(Reverse, reduced_fill_halo_regions!, Duplicated(Oceananigans.prognostic_fields(model), Oceananigans.prognostic_fields(dmodel)))

=#

big_tuple = ((1,2), (2,(4,5)), 3, 2, 1)

d_big_tuple = ((0,0), (0,(0,0)), 0, 0, 0)

flattened_big_tuple = flattened_unique_values(big_tuple)
reduced_flattened_big_tuple = reduced_flattened_unique_values(big_tuple)

@show flattened_big_tuple
@show reduced_flattened_big_tuple

function has_flattened_tuple!(maybe_nested_tuple::Union{NamedTuple, Tuple})

    flattened_tuple = flattened_unique_values(maybe_nested_tuple)
    return nothing
end

autodiff(Reverse, has_flattened_tuple!, Duplicated(big_tuple, d_big_tuple))
