# Abstract FEA operator and time integrator
abstract type AbstractFEOperator end
abstract type GeneralizedAlpha end
abstract type Newmark end
"""
    StaticFEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC=[], Neumann_BC=[];
                     ρ::Function=(x)->-1.0, f=Array, T=Float64)

Defines a finite element operator for a beam element, with the following fields:
    x     : Vector of the current solution
    resid : Vector of residuals
    ext   : Vector of external force
    stiff : Stiffness matrix
    jacob : Jacobian matrix
    mass  : Mass or mass matrix, depending on the problem
    mesh  : Isogeometric mesh
    gauss_rule    : Gauss quadrature rule
    Dirichlet_BC  : Vector of Dirichlet boundary conditions
    Neumann_BC    : Vector of Neumann boundary conditions
    EI : Coefficient of bending stiffness
    EA : Coefficient of axial stiffness
"""
struct StaticFEOperator{T,Vf<:AbstractArray{T},Mf<:AbstractArray{T}} <: AbstractFEOperator
    x     :: Vf
    resid :: Vf
    ext   :: Vf
    stiff :: Mf
    jacob :: Mf
    mass  :: Union{T,Mf}
    mesh  :: Mesh
    gauss_rule   :: GaussQuad
    Dirichlet_BC :: Vector{Boundary1D}
    Neumann_BC   :: Vector{Boundary1D}
    EI :: Function
    EA :: Function
    ρ :: Union{Function,Nothing} # density as a function of curvilnear coordinate
    g :: Union{Function,Nothing} # gravitational acceleration
    function StaticFEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC=[], Neumann_BC=[];
                              ρ=nothing, g=nothing,
                              f=Array, T=Float64)
        # generate
        Nd = 2*mesh.numBasis+length(Neumann_BC); Ng = (Nd,Nd)
        x, r, q = zeros(T,Nd) |> f, zeros(T,Nd) |> f, zeros(T,Nd) |> f
        K, J, M = zeros(T,Ng) |> f, zeros(T,Ng) |> f, zeros(T,Ng) |> f
        # no mass if not provided
        global_mass!(M, mesh, ρ, gauss_rule)
        new{T,typeof(x),typeof(K)}(x,r,q,K,J,M,mesh,gauss_rule,Dirichlet_BC,Neumann_BC,F2F(EI),F2F(EA),ρ,g)
    end
end
F2F(a::Function) = a
F2F(a::Number) = (x)->a
"""
    DynamicFEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC=[], Neumann_BC=[]; 
                      ρ::Function=(x)->-1.0, f=Array, ρ∞=1.0, T=Float64, I=GeneralizedAlpha)

Defines a finite element operator for a beam element, with the following fields:
    x     : Vector of the current solution
    resid : Vector of residuals
    ext   : Vector of external force
    stiff : Stiffness matrix
    jacob : Jacobian matrix
    mass  : Mass or mass matrix, depending on the problem
    mesh  : Isogeometric mesh
    gauss_rule    : Gauss quadrature rule
    Dirichlet_BC  : Vector of Dirichlet boundary conditions
    Neumann_BC    : Vector of Neumann boundary conditions
    EI : Coefficient of bending stiffness
    EA : Coefficient of axial stiffness
    f : Type of storage for the structure
    ρ∞: Spectral radius of the time integrator
    I : Time integrator, GeneralizedAlpha or Newmark
"""
struct DynamicFEOperator{I,T,Vf<:AbstractArray{T},Mf<:AbstractArray{T}} <: AbstractFEOperator
    x    :: Vf
    resid:: Vf
    ext  :: Vf
    stiff:: Mf
    jacob:: Mf
    mass :: Mf
    mesh :: Mesh
    gauss_rule :: GaussQuad
    Dirichlet_BC :: Vector{Boundary1D}
    Neumann_BC   :: Vector{Boundary1D}
    EI :: Function
    EA :: Function
    u ::Union{AbstractVector,Tuple{Vararg{AbstractVector}}}
    αm :: T
    αf :: T
    β :: T
    γ :: T
    Δt :: Vector{T}
    ρ :: Union{Function,Nothing} # density as a function of curvilnear coordinate
    g :: Union{Function,Nothing} # gravitational acceleration
    function DynamicFEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC=[], Neumann_BC=[]; 
                               ρ::Function=(x)->-1.0,  ρ∞=1.0, g=nothing, 
                               f=Array, T=Float64, I=GeneralizedAlpha)
        # time integration parameters
        αm = (2.0 - ρ∞)/(ρ∞ + 1.0);
        αf = 1.0/(1.0 + ρ∞)
        γ = 0.5 - αf + αm;
        β = 0.25*(1.0 - αf + αm)^2;
        # generate
        Nd = 2*mesh.numBasis+length(Neumann_BC); Ng = (Nd,Nd)
        x, r, q = zeros(T,Nd) |> f, zeros(T,Nd) |> f, zeros(T,Nd) |> f
        K, J, M = zeros(T,Ng) |> f, zeros(T,Ng) |> f, zeros(T,Ng) |> f
        # can be made nicer
        global_mass!(M, mesh, ρ, gauss_rule) #never changes between time steps
        new{I,T,typeof(x),typeof(K)}(x,r,q,K,J,M,mesh,gauss_rule,Dirichlet_BC,Neumann_BC,F2F(EI),F2F(EA),
                                      (zero(r),zero(r),zero(r)),αm,αf,β,γ,[0.0],ρ,g)
    end
end

"""
    residual(resid, x, force, op::AbstractFEOperator)

Compute the residuals of a given operator `op`, using an initial solution `x` and an 
external force `force`.
"""
function residual!(resid, x, force, op::AbstractFEOperator)
    # update the jacobian, the residual and the external force
    integrate!(op, x, force)

    # apply BC
    applyBC!(op)

    # assign residual
    resid .= op.stiff*x - op.ext
    return nothing
end


"""
    jacobian!(jacob, x, force, op::AbstractFEOperator)

Compute the Jacobian of a  given operator `op`, using an initial solution `x` and an 
external force `force`.
"""
function jacobian!(jacob, x, force, op::AbstractFEOperator)
    # update the jacobian, the residual and the external force
    integrate!(op, x, force)

    # apply BC
    applyBC!(op)

    # assign jacobian
    jacob .= op.jacob
    return nothing
end

"""
    integrate!(op::AbstractFEOperator, x0::Vector{T}, fx)

Integrate the operator `op` given an initial solution `x0` and an external force. 
This performs the element-wise integration of the different term in the governing equations.
"""
function integrate!(op::AbstractFEOperator, x0::AbstractVector{T}, fx::AbstractArray{T}) where T
    # reset
    op.stiff .= 0.; op.jacob .= 0.; op.ext .= 0.;
    nB,degP = op.mesh.numBasis,op.mesh.degP[1]
    # integrate on each element
    for iElem = 1:op.mesh.numElem
        
        # compute the (B-)spline basis functions and derivatives with Bezier extraction
        Jac_ref_par,N,dN,ddN = BSplineBasis(op.mesh, iElem)
        
        # where are we in the stiffness matrix
        I = element(iElem,degP)
        In = nodes(iElem,degP)

        # integrate on element
        for iGauss = 1:length(op.gauss_rule.nodes)
            #compute the rational basis, this all allpcates at leats one
            RR = N[iGauss,:].* op.mesh.weights[In]
            dR = dN[iGauss,:].* op.mesh.weights[In]
            ddR = ddN[iGauss,:].* op.mesh.weights[In]
            w_sum = sum(RR)
            dw_xi = sum(dR)
            dR = dR/w_sum - RR*dw_xi/w_sum^2
            ddR = ddR/w_sum - 2*dR*dw_xi/w_sum^2 - RR*sum(ddR)/w_sum^2 + 2*RR*dw_xi^2/w_sum^3

            #compute the derivatives w.r.t to the physical space
            dxdxi = dR' * op.mesh.controlPoints[1, In]
            Jac_par_phys = det(dxdxi)
            RR /= w_sum
            phys_pt = RR' * op.mesh.controlPoints[1, In]

            # compute linearised terms using the current solution
            du0dx = Jac_ref_par * Jac_par_phys * dR' * x0[In]
            dw0dx = Jac_ref_par * Jac_par_phys * dR' * x0[In.+nB]

            # external force at the Gauss point and gravity
            fi = external(fx, iElem, iGauss, op); gravity!(op, fi, phys_pt, op.g)
            op.ext[In    ] += Jac_par_phys * Jac_ref_par * fi[1] * RR * op.gauss_rule.weights[iGauss]
            op.ext[In.+nB] += Jac_par_phys * Jac_ref_par * fi[2] * RR * op.gauss_rule.weights[iGauss]

            # compute the different terms
            op.stiff[I          ] += Jac_ref_par * Jac_par_phys^2 * op.EI(phys_pt) * (ddR*ddR') * op.gauss_rule.weights[iGauss]
            op.stiff[I.+δ(nB,nB)] += Jac_ref_par * Jac_par_phys^2 * op.EI(phys_pt) * (ddR*ddR') * op.gauss_rule.weights[iGauss]
            
            # only different entry of Jacobian
            # op.jacob[I.+δ(nB,nB)] += Jac_ref_par * Jac_par_phys * op.EA(phys_pt) * (dw0dx^2) * (dR*dR') * op.gauss_rule.weights[iGauss]
        end
    end
    # enforce symmetry K21 -> K12
    @loop op.stiff[I] = op.stiff[symmetric(I)] over I in symmetric(nB)
    # form jacobian
    @loop op.jacob[I] += op.stiff[I] over I in CartesianIndices(op.jacob)
end

"""
    external

Get the external force at the Gauss point `iGauss` of the element `iElem`.
"""
function external(fx, iElem, iGauss, op::AbstractFEOperator)
    return fx[:,(iElem-1)*length(op.gauss_rule.nodes)+iGauss]
end
"""
    gravity!

Add gravity contribution to the external force vector
"""
gravity!(op::AbstractFEOperator,f_ext,ξ,::Nothing) = nothing
gravity!(op::AbstractFEOperator,f_ext,ξ,::Function) = for i ∈ 1:2
    f_ext[i] += op.g(i,ξ)*op.ρ(ξ)
end
"""
    applyBC(op::AbstractFEOperator)

Apply the boundary conditions to the operator `op`. This modifies the stiffness matrix,
the residual and the Jacobian. The Neumann boundary conditions are applied using the
Lagrange multiplier method, while the Dirichlet boundary conditions are applied using
the penalty method.
"""
function applyBC!(op::AbstractFEOperator)
    numBasis = op.mesh.numBasis
    # apply Neumann BC
    for i=eachindex(op.Neumann_BC)
        off = (op.Neumann_BC[i].comp-1)*op.mesh.numBasis
        cdof_neu = findall(op.mesh.controlPoints[1,:].==op.Neumann_BC[i].x_val)
        for iElem=1:op.mesh.numElem
            # index of that element
            In = nodes(iElem,op.mesh.degP[1])
            if cdof_neu[1] in In
                # get the actual node position
                BC_nodes = [2*op.Neumann_BC[i].u_val-1] # map into parameter space
                # compute the Bernstein basis functions and derivatives
                B, dB, _ = bernsteinBasis(BC_nodes, op.mesh.degP[1])
                Jac_ref_par = Jξ(op.mesh.elemVertex,iElem)

                #compute the (B-)spline basis functions and derivatives with Bezier extraction
                N_mat = B * op.mesh.C[iElem]
                dN_mat = dB * op.mesh.C[iElem]/Jac_ref_par

                #compute the rational basis
                RR = N_mat[1,:].* op.mesh.weights[In]
                dR = dN_mat[1,:].* op.mesh.weights[In]
                w_sum = sum(RR)
                dw_xi = sum(dR)
                dR = dR/w_sum - RR*dw_xi/w_sum^2

                #compute the Jacobian of the transformation from parameter to physical space
                dxdxi = dR' * op.mesh.controlPoints[1,In]
                Jac_par_phys = det(dxdxi)

                # better indeexing
                op.stiff[2numBasis+i,In.+off] += Jac_ref_par * Jac_par_phys * dR * op.gauss_rule.weights[1] 
                op.stiff[In.+off,2numBasis+i] += Jac_ref_par * Jac_par_phys * dR * op.gauss_rule.weights[1] 

                # lagrange multiplier entry
                op.resid[2numBasis+i] = op.Neumann_BC[i].op_val
            end
        end
    end
    if length(op.Neumann_BC) > 0
        # propagate the Neumann BC to the jacobian
        op.jacob[2numBasis+1:end,:] .= op.stiff[2numBasis+1:end,:]
        op.jacob[:,2numBasis+1:end] .= op.stiff[:,2numBasis+1:end]
    end
    
    # apply Dirichlet BC
    for i=eachindex(op.Dirichlet_BC)
        bcdof = findall(op.mesh.controlPoints[1,:].==op.Dirichlet_BC[i].x_val)
        for j ∈ op.Dirichlet_BC[i].comp
            # get the actual node position
            bcdof .+= (j-1)*op.mesh.numBasis
            # get the BC value
            bcval = op.Dirichlet_BC[i].op_val
            op.resid .-= op.stiff[:,bcdof]*bcval
            op.resid[bcdof] .= bcval
            # reset stiffness and mass
            op.stiff[bcdof,:] .= op.jacob[bcdof,:] .= 0.0 
            op.stiff[:,bcdof] .= op.jacob[:,bcdof] .= 0.0
            op.stiff[bcdof,bcdof] .= op.jacob[bcdof,bcdof] .= 1.0
        end
    end
end

"""
    global_mass!(mass::AbstractArray{T}, mesh::Mesh{T}, ρ::Union{Nothing,Function}, gauss_rule)

Integrate the (consistant) global mass matrix over the mesh, given a density.
"""
global_mass!(mass::AbstractArray{T}, mesh::Mesh{T}, ::Nothing, gauss_rule) where T = (mass = zero(T))
function global_mass!(mass::AbstractArray{T}, mesh::Mesh{T}, ρ::Function, gauss_rule) where {T}
    numBasis,degP = mesh.numBasis,mesh.degP[1]
    for iElem = 1:mesh.numElem
        # compute the (B-)spline basis functions and derivatives with Bezier extraction
        Jac_ref_par,N,dN,ddN = BSplineBasis(mesh, iElem)
        
        # where are we in the stiffness matrix
        I = element(iElem,degP)
        In = nodes(iElem,degP)

        # integrate on element
        for iGauss = 1:length(gauss_rule.nodes)
            #compute the rational basis
            RR = N[iGauss,:] .* mesh.weights[In]
            dR = dN[iGauss,:] .* mesh.weights[In]
            w_sum = sum(RR)
            dw_xi = sum(dR)
            dR = dR/w_sum - RR*dw_xi/w_sum^2

            #compute the derivatives w.r.t to the physical space
            dxdxi = dR' * mesh.controlPoints[1,In]
            Jac_par_phys = det(dxdxi)
            RR /= w_sum
            phys_pt = RR' * mesh.controlPoints[1,In]

            # add contribution
            mass[I] += Jac_par_phys * Jac_ref_par * ρ(phys_pt) * (RR*RR') * gauss_rule.weights[iGauss]
            mass[I.+δ(numBasis,numBasis)] += Jac_par_phys * Jac_ref_par * ρ(phys_pt) * (RR*RR') * gauss_rule.weights[iGauss]
        end
    end
end
