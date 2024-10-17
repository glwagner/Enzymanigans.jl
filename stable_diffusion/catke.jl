using Enzyme

#Enzyme.Compiler.DumpPreEnzyme[] = true

using Oceananigans
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity

Enzyme.API.looseTypeAnalysis!(true)
Enzyme.API.maxtypeoffset!(2032)

bᵢ(z) = 1e-5 * z

archs = [CPU()]

function buoyancy_variance!(model, e_min, Δt=10.0)
    new_closure = CATKEVerticalDiffusivity(minimum_tke=e_min)
    model.closure = new_closure
    model.clock.time = 0
    model.clock.iteration = 0
    set!(model, b=bᵢ)

    for n = 1:10
        time_step!(model, Δt)
    end

    b = model.tracers.b

    # Another way to compute it
    Nx, Ny, Nz = size(model.grid)
    sum_b² = 0.0
    for k = 1:Nz, j = 1:Ny,  i = 1:Nx
        sum_b² += b[i, j, k]^2
    end

    return sum_b²
end

for arch in archs
    grid = RectilinearGrid(arch, size=50, z=(-200, 0), topology=(Flat, Flat, Bounded))
    closure = CATKEVerticalDiffusivity()
    b_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(1e-7))
    tracers = (:b, :e)
    buoyancy = BuoyancyTracer()
    model = HydrostaticFreeSurfaceModel(; grid, closure, tracers, buoyancy,
                                        boundary_conditions=(; b_bcs))

    # Compute derivative by hand
    e₁, e₂ = 1e-5, 2e-5
    b²₁ = buoyancy_variance!(model, e₁)
    b²₂ = buoyancy_variance!(model, e₂)
    db²_de_fd = (b²₂ - b²₁) / (e₂ - e₁)

    # Now for real
    e = 1e-5
    dmodel = Enzyme.make_zero(model)
    buoyancy_variance!(dmodel, 0)

    db²_de = autodiff(Enzyme.set_runtime_activity(Enzyme.Reverse),
                      buoyancy_variance!,
                      Duplicated(model, dmodel),
                      Active(e))

    @info """ \n
        Enzyme computed $db²_de
        Finite differences computed $db²_de_fd
    """

    tol = 0.01
    rel_error = abs(db²_de[1][3] - db²_de_fd) / abs(db²_de_fd)
    @show db²_de, db²_de_fd
    @show rel_error < tol
end