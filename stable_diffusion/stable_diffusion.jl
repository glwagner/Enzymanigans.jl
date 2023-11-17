using Oceananigans
using Enzyme

using Oceananigans.Fields

Enzyme.API.runtimeActivity!(true)
Enzyme.API.printall!(true)

args = (
	Val{(false, false, false)},
	Val(1),
	Val((true, true, true)),
	Base.Val(NamedTuple{(Symbol("1"), Symbol("2"), Symbol("3")), Tuple{Any, Any, Any}}),
	Base.getindex,
	nothing,
	((nothing,), Oceananigans.BoundaryConditions.BoundaryCondition(Oceananigans.BoundaryConditions.Flux(),nothing)),
	((nothing,), Oceananigans.BoundaryConditions.BoundaryCondition(Oceananigans.BoundaryConditions.Flux(),nothing)),
	1,
	nothing
)

using InteractiveUtils
@show InteractiveUtils.@code_typed Enzyme.Compiler.runtime_generic_augfwd(args...)
@show InteractiveUtils.@code_llvm Enzyme.Compiler.runtime_generic_augfwd(args...)
Enzyme.Compiler.runtime_generic_augfwd(args...)
