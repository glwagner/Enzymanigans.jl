using Oceananigans
using Oceananigans.TurbulenceClosures: with_tracers
#using Oceananigans.Models.HydrostaticFreeSurfaceModels: tracernames
using Enzyme

using Oceananigans.Fields
#using Oceananigans.Fields: default_indices
#using Oceananigans.BoundaryConditions
#using Oceananigans.BoundaryConditions: FieldBoundaryConditions, regularize_field_boundary_conditions
#using Oceananigans: instantiated_location

using Oceananigans.BoundaryConditions: extract_bottom_bc

#using OffsetArrays: OffsetArray

# include("../../FlattenedTuples/FlattenedTuples.jl")
# import .FlattenedTuples: reduced_flattened_unique_values

# Enzyme.API.strictAliasing!(false)

#Enzyme.API.printunnecessary!(true)
Enzyme.API.runtimeActivity!(true)
#Enzyme.API.printall!(true)
Enzyme.EnzymeRules.inactive_type(::Type{<:Oceananigans.Grids.AbstractGrid}) = true
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

function reduced_fill_halo_regions!(maybe_nested_tuple::Union{NamedTuple, Tuple})
    tupled = tuplify(maybe_nested_tuple)
    ordinary_fields = flatten_tuple(tupled)

    bcs  = reduced_permute_boundary_conditions(map(boundary_conditions, ordinary_fields))

    Base.inferencebarrier(fill_west_and_east_halo!)(map(data, ordinary_fields), bcs[1])    
    return nothing
end

@inline function local_flattened_unique_values(a::Union{NamedTuple, Tuple})
    
    #Converts a from a named tuple into a tuple:
    tupled = tuplify(a)
    flattened = flatten_tuple(tupled)
    return flattened #  Tuple(last(push!(seen, f)) for f in flattened if !any(f === s for s in seen))
end

# Utility for extracting values from nested NamedTuples
@inline tuplify(a::NamedTuple) = Tuple(tuplify(ai) for ai in a)
@inline tuplify(a) = a

# Outer-inner form
@inline flatten_tuple(a::Tuple) = tuple(inner_flatten_tuple(a[1])..., inner_flatten_tuple(a[2:end])...)
@inline flatten_tuple(a::Tuple{<:Any}) = tuple(inner_flatten_tuple(a[1])...)

@inline inner_flatten_tuple(a) = tuple(a)
@inline inner_flatten_tuple(a::Tuple) = flatten_tuple(a)
#@inline inner_flatten_tuple(a::Tuple{}) = ()

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
    sides = (:west_and_east, :bottom_and_top)
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

reduced_fill_halo_regions!(Oceananigans.prognostic_fields(model))

autodiff(Reverse, reduced_fill_halo_regions!, Duplicated(Oceananigans.prognostic_fields(model), Oceananigans.prognostic_fields(dmodel)))
