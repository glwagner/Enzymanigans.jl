# Currently, running this requires
# - Enzyme#main (eg, > v0.11.11)
# - Oceananigans#glw/type-stable-with-tracers
# - KernelAbstractions#enzymeact

using Oceananigans
using Enzyme

Enzyme.API.runtimeActivity!(true)
# Enzyme.API.printall!(true)
Enzyme.API.instname!(true)
Enzyme.API.printdiffuse!(true)
Enzyme.API.printactivity!(true)
Enzyme.API.looseTypeAnalysis!(true)
Enzyme.EnzymeRules.inactive_type(::Type{<:Oceananigans.Grids.AbstractGrid}) = true
Enzyme.EnzymeRules.inactive_type(::Type{<:Oceananigans.Clock}) = truels
Enzyme.EnzymeRules.inactive_noinl(::typeof(Core._compute_sparams), args...) = nothing

Enzyme.autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(Oceananigans.Operators.interpolation_operator)}, Duplicated, Const{typeof((Oceananigans.Grids.Center, Oceananigans.Grids.Center, Oceananigans.Grids.Center))}, Const{typeof((Oceananigans.Grids.Center, Oceananigans.Grids.Center, Oceananigans.Grids.Center))})
# Enzyme.Compiler.runtime_generic_augfwd(Va({(false, false, false, true, false, false, false, false)), Val(1), Val((true, true, true, true, true, true, true, true)), Val(@NamedTuple{1, 2, 3}), f::typeof(Oceananigans.AbstractOperations._binary_operation), df::Nothing, primal_1::Tuple{DataType, DataType, DataType}, shadow_1_1::Nothing, primal_2::typeof(^), shadow_2_1::Nothing, primal_3::Field{Center, Center, Center, Nothing, RectilinearGrid{Float64, Flat, Flat, Bounded, Float64, Float64, Float64, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}, OffsetArrays.OffsetVector{Float64, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}}, CPU}, Tuple{Colon, Colon, Colon}, OffsetArrays.OffsetArray{Float64, 3, Array{Float64, 3}}, Float64, FieldBoundaryConditions{Nothing, Nothing, Nothing, Nothing, BoundaryCondition{Oceananigans.BoundaryConditions.Flux, Nothing}, BoundaryCondition{Oceananigans.BoundaryConditions.Flux, Nothing}, BoundaryCondition{Oceananigans.BoundaryConditions.Flux, Nothing}}, Nothing, Oceananigans.Fields.FieldBoundaryBuffers{Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing}}, shadow_3_1::Field{Center, Center, Center, Nothing, RectilinearGrid{Float64, Flat, Flat, Bounded, Float64, Float64, Float64, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}, OffsetArrays.OffsetVector{Float64, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}}, CPU}, Tuple{Colon, Colon, Colon}, OffsetArrays.OffsetArray{Float64, 3, Array{Float64, 3}}, Float64, FieldBoundaryConditions{Nothing, Nothing, Nothing, Nothing, BoundaryCondition{Oceananigans.BoundaryConditions.Flux, Nothing}, BoundaryCondition{Oceananigans.BoundaryConditions.Flux, Nothing}, BoundaryCondition{Oceananigans.BoundaryConditions.Flux, Nothing}}, Nothing, Oceananigans.Fields.FieldBoundaryBuffers{Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing}}, primal_4::Int64, shadow_4_1::Nothing, primal_5::Tuple{DataType, DataType, DataType}, shadow_5_1::Nothing, primal_6::Tuple{DataType, DataType, DataType}, shadow_6_1::Nothing, primal_7::RectilinearGrid{Float64, Flat, Flat, Bounded, Float64, Float64, Float64, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}, OffsetArrays.OffsetVector{Float64, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}}, CPU}, shadow_7_1::Nothing)
 

# dc²_dκ = autodiff(Reverse, stable_diffusion!, Duplicated(model, dmodel))
