using Oceananigans
using Oceananigans.Fields
using Enzyme

Enzyme.API.printunnecessary!(true)

Base.get_extension(Oceananigans, :OceananigansEnzymeExt)

Enzyme.API.runtimeActivity!(true)
#Enzyme.API.printall!(true)
Enzyme.EnzymeRules.inactive_type(::Type{<:Oceananigans.Grids.AbstractGrid}) = true
Enzyme.EnzymeRules.inactive_noinl(::typeof(Core._compute_sparams), args...) = nothing

# The function as it appears in Oceananigans, with its corresponding helpers:
@inline function local_flattened_unique_values(a::Union{NamedTuple, Tuple})
    tupled = Tuple(tuplify(ai) for ai in a)
    flattened = flatten_tuple(tupled)

    # Alternative implementation of `unique` for tuples that uses === comparison, rather than ==
    seen = []
    return Tuple(last(push!(seen, f)) for f in flattened if !any(f === s for s in seen))
end

# Utility for extracting values from nested NamedTuples
@inline tuplify(a::NamedTuple) = Tuple(tuplify(ai) for ai in a)
@inline tuplify(a) = a

# Outer-inner form
@inline flatten_tuple(a::Tuple) = tuple(inner_flatten_tuple(a[1])..., inner_flatten_tuple(a[2:end])...)
@inline flatten_tuple(a::Tuple{<:Any}) = tuple(inner_flatten_tuple(a[1])...)

@inline inner_flatten_tuple(a) = tuple(a)
@inline inner_flatten_tuple(a::Tuple) = flatten_tuple(a)
@inline inner_flatten_tuple(a::Tuple{}) = ()

# End of stuff that appears in Oceananigans


nested_tuple = ((1,2), (2,(4,5)), 3, 2, 1)
d_nested_tuple = ((0,0), (0,(0,0)), 0, 0, 0)

flattened_nested_tuple = flattened_unique_values(nested_tuple)
local_flattened_nested_tuple = local_flattened_unique_values(nested_tuple)

# Verifying outputs are the same
@show flattened_nested_tuple
@show local_flattened_nested_tuple

# The function we wish to differentiate:
function has_flattened_tuple!(maybe_nested_tuple::Union{NamedTuple, Tuple})

    flattened_tuple = flattened_unique_values(maybe_nested_tuple)
    return nothing
end

# The function we wish to differentiate, now using the locally-defined helper:
function local_has_flattened_tuple!(maybe_nested_tuple::Union{NamedTuple, Tuple})

    flattened_tuple = local_flattened_unique_values(maybe_nested_tuple)
    return nothing
end

autodiff(Reverse, has_flattened_tuple!, Duplicated(nested_tuple, d_nested_tuple))
println("Finished differentiating the helper defined in Oceananigans. Now let's try the locally-written version:")

nested_tuple = ((1,2), (2,(4,5)), 3, 2, 1)
d_nested_tuple = ((0,0), (0,(0,0)), 0, 0, 0)
autodiff(Reverse, local_has_flattened_tuple!, Duplicated(nested_tuple, d_nested_tuple))
