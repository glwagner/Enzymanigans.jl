using Enzyme

Enzyme.API.runtimeActivity!(true)
Enzyme.API.looseTypeAnalysis!(true)
Enzyme.EnzymeRules.inactive_noinl(::typeof(Core._compute_sparams), args...) = nothing


@inline extract_bc(bc, ::Val{:north}) = (bc.north)
@inline extract_bc(bc, ::Val{:top}) = (bc.top)

function permute_boundary_conditions(boundary_conditions)
    sides = [:north, :top]
    sides = sides[[2, 1]]
    boundary_conditions = Tuple(extract_bc(boundary_conditions, Val(side)) for side in sides)

    return nothing
end

parameters = (a = 1, b = 0.1)

bc   = (north=1, top=tuple(parameters, tuple(:c)))
d_bc = Enzyme.make_zero(bc)

@show bc
@show d_bc

dc²_dκ = autodiff(Enzyme.Reverse,
                  permute_boundary_conditions,
                  Duplicated(bc, d_bc))

@info """ \n
Enzyme computed $dc²_dκ
"""
