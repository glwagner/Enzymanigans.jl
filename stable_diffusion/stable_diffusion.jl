using Enzyme

Enzyme.API.runtimeActivity!(true)
Enzyme.API.looseTypeAnalysis!(true)
Enzyme.EnzymeRules.inactive_noinl(::typeof(Core._compute_sparams), args...) = nothing

@inline function tracer_flux(x, y, t, c, p)
    return c
end

parameters = (a = 1, b = 0.1)


@inline extract_bc(bc, ::Val{:north}) = (bc.north)
@inline extract_bc(bc, ::Val{:top}) = (bc.top)

function permute_boundary_conditions(boundary_conditions)
    sides = [:north, :top]
    sides = sides[[2, 1]]
    boundary_conditions = Tuple(extract_bc(boundary_conditions, Val(side)) for side in sides)

    return nothing
end


struct ContinuousBoundaryFunction{X, Y, Z, S, F, P, D}
    func :: F
    parameters :: P
    field_dependencies :: D

    """ Returns a location-less wrapper for `func`, `parameters`, and `field_dependencies`."""
    function ContinuousBoundaryFunction(func::F, parameters::P, field_dependencies) where {F, P}
    field_dependencies = tuple(field_dependencies)
    D = typeof(field_dependencies)
    return new{Nothing, Nothing, Nothing, Nothing, F, P, D}(func, parameters, field_dependencies)
    end
end

bc = (north=1, top=ContinuousBoundaryFunction(tracer_flux, parameters, :c))

d_bc = Enzyme.make_zero(bc)

@show bc
@show d_bc

dc²_dκ = autodiff(Enzyme.Reverse,
                  permute_boundary_conditions,
                  Duplicated(bc, d_bc))

@info """ \n
Enzyme computed $dc²_dκ
"""
