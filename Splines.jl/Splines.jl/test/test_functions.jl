function test_fixed_fixed_UDL(numElem=2, degP=3)
    # Material properties and mesh
    ptLeft = 0.0
    ptRight = 1.0
    EI = 1.0
    EA = 1.0
    exact_sol(x) = x.^2/(24EI).*(1 .- x).^2    # fixed - fixed

    mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

    # boundary conditions
    Dirichlet_BC = [
        Boundary1D("Dirichlet", ptRight, 0.0; comp=1),
        Boundary1D("Dirichlet", ptRight, 0.0; comp=2),
        Boundary1D("Dirichlet", ptLeft, 0.0; comp=1),
        Boundary1D("Dirichlet", ptLeft, 0.0; comp=2)
    ]
    Neumann_BC = [
        Boundary1D("Neumann", ptLeft, 0.0; comp=2),
        Boundary1D("Neumann", ptRight, 0.0; comp=2)
    ]

    # make a problem
    operator = StaticFEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC, Neumann_BC)

    # uniform external loading at integration points
    force = zeros(2,length(uv_integration(operator))); force[2,:] .= 1.0
    result = lsolve!(operator, force)

    u0 = result[1:mesh.numBasis]
    w0 = result[mesh.numBasis+1:2mesh.numBasis]
    x = LinRange(ptLeft, ptRight, numElem+1)
    u = x .+ getSol(mesh, u0, 1)
    w = getSol(mesh, w0, 1)
    we = exact_sol(x)
    return norm(w .- we)
end


function test_fixed_fixed_gravity(numElem=2, degP=3)
    # Material properties and mesh
    ptLeft = 0.0
    ptRight = 1.0
    EI = 1.0
    EA = 1.0
    density(ξ) = 1.0
    exact_sol(x) = x.^2/(24EI).*(1 .- x).^2    # fixed - fixed

    mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

    # boundary conditions
    Dirichlet_BC = [
        Boundary1D("Dirichlet", ptRight, 0.0; comp=1),
        Boundary1D("Dirichlet", ptRight, 0.0; comp=2),
        Boundary1D("Dirichlet", ptLeft, 0.0; comp=1),
        Boundary1D("Dirichlet", ptLeft, 0.0; comp=2)
    ]
    Neumann_BC = [
        Boundary1D("Neumann", ptLeft, 0.0; comp=2),
        Boundary1D("Neumann", ptRight, 0.0; comp=2)
    ]

    # gravity function
    g(i,ξ) = i==2 ? 1.0 : 0.0

    # make a problem
    operator = StaticFEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC, Neumann_BC;
                                ρ=density, g=g)

    # uniform external loading at integration points
    force = zeros(2,length(uv_integration(operator)))
    result = lsolve!(operator, force)

    u0 = result[1:mesh.numBasis]
    w0 = result[mesh.numBasis+1:2mesh.numBasis]
    x = LinRange(ptLeft, ptRight, numElem+1)
    u = x .+ getSol(mesh, u0, 1)
    w = getSol(mesh, w0, 1)
    we = exact_sol(x)
   return norm(w .- we)
end


function test_pinned_pinned_UDL(numElem=2, degP=3)
    # Material properties and mesh
    ptLeft = 0.0
    ptRight = 1.0
    EI = 1.0
    EA = 1.0
    exact_sol(x) = 1.0/(24EI).*(x .- 2x.^3 .+ x.^4) # pinned - pinned

    mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

    # boundary conditions
    Dirichlet_BC = [
        Boundary1D("Dirichlet", ptRight, 0.0; comp=1),
        Boundary1D("Dirichlet", ptRight, 0.0; comp=2),
        Boundary1D("Dirichlet", ptLeft, 0.0; comp=1),
        Boundary1D("Dirichlet", ptLeft, 0.0; comp=2)
    ]
    Neumann_BC = []

    # make a problem
    operator = StaticFEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC, Neumann_BC)

    # uniform external loading at integration points
    force = zeros(2,length(uv_integration(operator))); force[2,:] .= 1.0
    result = lsolve!(operator, force)

    # get the solution
    u0 = result[1:mesh.numBasis]
    w0 = result[mesh.numBasis+1:2mesh.numBasis]
    x = LinRange(ptLeft, ptRight, numElem+1)
    u = x .+ getSol(mesh, u0, 1)
    w = getSol(mesh, w0, 1)
    we = exact_sol(x)
    return norm(w .- we)
end


function test_fixed_free_UDL(numElem=2, degP=3)
    # Material properties and mesh
    ptLeft = 0.0
    ptRight = 1.0
    EI = 1.0
    EA = 1.0
    exact_sol(x) = x.^2/(24EI).*(6 .- 4x .+ x.^2) # fixed - free

    mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

    # boundary conditions
    Dirichlet_BC = [
        Boundary1D("Dirichlet", ptLeft, 0.0; comp=1),
        Boundary1D("Dirichlet", ptLeft, 0.0; comp=2)
    ]
    Neumann_BC = [Boundary1D("Neumann", ptLeft, 0.0; comp=2)]

    # make a problem
    operator = StaticFEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC, Neumann_BC)

    # uniform external loading at integration points
    force = zeros(2,length(uv_integration(operator))); force[2,:] .= 1.0
    result = lsolve!(operator, force)

    # get the results
    u0 = result[1:mesh.numBasis]
    w0 = result[mesh.numBasis+1:2mesh.numBasis]
    x = LinRange(ptLeft, ptRight, numElem+1)
    u = x .+ getSol(mesh, u0, 1)
    w = getSol(mesh, w0, 1)
    we = exact_sol(x)
    return norm(w .- we)
end


function test_fixed_free_PL(numElem=2, degP=3)
    # Material properties and mesh
    ptLeft = 0.0
    ptRight = 1.0
    EI = 1.0
    EA = 1.0
    P = 1.0
    exact_sol(x) = P.*x.^2/(6EI).*(3 .- x) # fixed - free (Ponts Load)

    # mesh
    mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

    # boundary conditions: u(0)=w(0)=0.0, dw(0)/dx=0.0
    Dirichlet_BC = [
        Boundary1D("Dirichlet", ptLeft, 0.0; comp=1),
        Boundary1D("Dirichlet", ptLeft, 0.0; comp=2)
    ]
    Neumann_BC = [
        Boundary1D("Neumann", ptLeft, 0.0; comp=2)
    ]

    # make a problem
    operator = StaticFEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC, Neumann_BC)

    # linearized residuals
    integrate!(operator, zero(operator.resid), zeros(2,length(uv_integration(operator))))

    # compute the residuals
    operator.resid .= - operator.ext

    # apply BC
    applyBC!(operator)

    # solve static problem with point load
    operator.resid[2*mesh.numBasis] -= P

    # solve the problem
    result = -operator.jacob\operator.resid

    # get the results
    u0 = result[1:mesh.numBasis]
    w0 = result[mesh.numBasis+1:2mesh.numBasis]
    x = LinRange(ptLeft, ptRight, numElem+1)
    u = x .+ getSol(mesh, u0, 1)
    w = getSol(mesh, w0, 1)
    we = exact_sol(x)
    return norm(w .- we)
end


function test_fixed_free(numElem=2, degP=3)
    # Material properties and mesh
    ptLeft = 0.0
    ptRight = 1.0
    EI = 1
    EA = 12 #/0.1^2
    F = 3EI/2
    println(" EI: ", EI, " EA: ", EA, " F: ", F)
    f(x) = [0.0, 0.0]

    mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

    # boundary conditions
    Dirichlet_BC = [
        Boundary1D("Dirichlet", ptLeft, 0.0; comp=1),
        Boundary1D("Dirichlet", ptLeft, 0.0; comp=2)
    ]
    Neumann_BC = [
        Boundary1D("Neumann", ptLeft, 0.0; comp=1),
        Boundary1D("Neumann", ptLeft, 0.0; comp=2)
    ]

    # make a problem
    operator = StaticFEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC, Neumann_BC)

    # unpack pre-allocated storage and the convergence flag
    @unpack resid, jacob = operator
    f = zeros(2,length(uv_integration(operator)))
    x = zero(resid)

    # warp the residual and the jacobian
    function f!(resid, x) 
        residual!(resid, x, f, operator)
        # need a custom solver as the resdiuals are different
        resid[2*mesh.numBasis] -= F
        return nothing
    end
    j!(jacob, x) = jacobian!(jacob, x, f, operator)

    # prepare for solve
    df = NLsolve.OnceDifferentiable(f!, j!, x, resid, jacob)

    # # solve the system
    solver = NLsolve.nlsolve(df, x,
                            show_trace=false,
                            linsolve=(x, A, b) -> x .= ImplicitAD.implicit_linear(A, b),
                            method=:newton,
                            linesearch=LineSearches.BackTracking(maxstep=1e6),
                            ftol=1e-9,
                            iterations=1000)

    # update the state, residual, jacobian, and convergence flag
    result = solver.zero
    # resid .= df.F
    # jacob .= df.DF

    # get the results
    u0 = result[1:mesh.numBasis]
    w0 = result[mesh.numBasis+1:2mesh.numBasis]
    x = LinRange(ptLeft, ptRight, numElem+1)
    u = x .+ getSol(mesh, u0, 1)
    w = getSol(mesh, w0, 1)
    return 0.0
end

function test_nonlinear(numElem=20, degP=2)
    # Material properties and mesh
    ptLeft = 0.0
    ptRight = 1.0
    L = 1.0
    EI = 2
    EA = 50
    density(ξ) = 0.0
    α² = 1.0

    # mesh
    mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

    # boundary conditions
    Dirichlet_BC = [
        Boundary1D("Dirichlet", ptLeft, 0.0; comp=1),
        Boundary1D("Dirichlet", ptLeft, 0.0; comp=2)
    ]
    Neumann_BC = [
        Boundary1D("Neumann", ptLeft, 0.0; comp=1),
        Boundary1D("Neumann", ptLeft, 0.0; comp=2)
    ]

    # make a problem
    operator = StaticFEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC, Neumann_BC; ρ=density)
    
    # time steps
    Nₜ = 10;
    println("Number of time increment: ", Nₜ)
    # get the results
    xs = LinRange(ptLeft, ptRight, numElem+1)

    # unpack pre-allocated storage and the convergence flag
    @unpack resid, jacob = operator
    x = zero(resid)
    f = zeros(2,length(uv_integration(operator)))

    # time loops
    @time @gif for k = 0:Nₜ
        
        # warp the residual and the jacobian
        function f!(resid, x) 
            residual!(resid, x, f, operator)
            # need a custom solver as the resdiuals are different
            resid[2*mesh.numBasis] += π*k/Nₜ*α²*EI
            return nothing
        end
        j!(jacob, x) = jacobian!(jacob, x, f, operator)

        # prepare for solve
        df = NLsolve.OnceDifferentiable(f!, j!, x, resid, jacob)

        # # solve the system
        solver = NLsolve.nlsolve(df, x,
                                show_trace=false,
                                linsolve=(x, A, b) -> x .= ImplicitAD.implicit_linear(A, b),
                                method=:newton,
                                linesearch=LineSearches.BackTracking(maxstep=1e6),
                                ftol=1e-9,
                                iterations=1000)

        # update the state, residual, jacobian, and convergence flag
        result = solver.zero
        u0 = result[1:mesh.numBasis]
        w0 = result[mesh.numBasis+1:2mesh.numBasis]
        u = xs .+ getSol(operator.mesh, u0, 21)
        w = getSol(operator.mesh, w0, 21)
        Plots.plot(u, w, legend=:none, xlim=(-0.5*ptRight,ptRight*1.5), aspect_ratio=:equal, ylims=(0.5-2ptRight,0.5), 
                    title="tU/L = $k")
        println("beam length ", round(sum(sqrt.(diff(u).^2 + diff(w).^2)), digits=4))
    end

end

function test_dynamic(numElem=2, degP=3)
    # Material properties and mesh
    ptLeft = 0.0
    ptRight = 1.0
    L = 1.0
    EI = 0.35
    EA = 1e5
    density(ξ) = 0.0
    P = 3EI/2

    # natural frequencies
    ωₙ = [1.875, 4.694, 7.855]
    fhz = ωₙ.^2.0.*√(EI/(density(0.0)*L^4))/(2π)
    fhz[1] = 0.25
    @show 1.0./fhz

    # mesh
    mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

    # boundary conditions
    Dirichlet_BC = [
        Boundary1D("Dirichlet", ptLeft, 0.0; comp=1),
        Boundary1D("Dirichlet", ptLeft, 0.0; comp=2)
    ]
    Neumann_BC = [
        Boundary1D("Neumann", ptLeft, 0.0; comp=1),
        Boundary1D("Neumann", ptLeft, 0.0; comp=2)
    ]

    # make a problem
    operator = DynamicFEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC, Neumann_BC; ρ=density, ρ∞=0.5)

    # time steps
    Δt = 0.2
    T = 10*L/fhz[1]
    time = collect(0.0:Δt:T);
    Nₜ = length(time);
    println("Number of time steps: ", Nₜ)
    # get the results
    xs = LinRange(ptLeft, ptRight, numElem+1)

    # external force
    f_ext = zeros(2,length(uv_integration(operator)))

    # time loops
    @time @gif for k = 2:Nₜ
        # external loading
        f_ext .= 0.0
        f_ext[2,:] .= 2P*sin(2π*fhz[1]*time[k]/L)
        
        # integrate one in time
        solve_step!(operator, f_ext, Δt)

        # get the results
        u0 = operator.u[1][1:operator.mesh.numBasis]
        w0 = operator.u[1][operator.mesh.numBasis+1:2operator.mesh.numBasis]
        u = xs .+ getSol(operator.mesh, u0, 1)
        w = getSol(operator.mesh, w0, 1)
        ti =round(time[k],digits=3)
        Plots.plot(u, w, legend=:none, xlim=(-0.5*ptRight,ptRight*1.5), aspect_ratio=:equal, ylims=(-ptRight,ptRight), 
                title="tU/L = $(ti/L)")
        println("beam length ", round(sum(sqrt.(diff(u).^2 + diff(w).^2)), digits=4))
    end
end

# # linear tests
# test_fixed_fixed_UDL(3, 4)
# test_fixed_fixed_gravity(3,4)
# test_pinned_pinned_UDL(3, 4)
# test_fixed_free_UDL(3, 4)
# test_fixed_free_PL(3, 3)

# # non-linear tests, needs validations
# test_fixed_free(8, 4)
# test_nonlinear()
# test_dynamic()
