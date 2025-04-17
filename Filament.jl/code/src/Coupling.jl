using LinearAlgebra: norm,dot
include("QRFactorization.jl")
include("Methods.jl")

abstract type AbstractCoupling end
"""
    update!(cp::AbstractCoupling,primary,secondary,kwargs...)

Updates the coupling variable `cp` using the implemented couping scheme.
"""
function update!(cp::AbstractCoupling,primary,secondary;kwargs...)
    xᵏ=zero(cp.x); concatenate!(xᵏ,primary,secondary,cp.subs)
    converged = update!(cp,xᵏ;kwargs...)
    revert!(xᵏ,primary,secondary,cp.subs)
    return converged
end

"""
    CoupledSimulation()

A struct to hold the coupled simulation of a fluid-structure interaction problem.
"""
mutable struct CoupledSimulation <: AbstractSimulation
    U :: Number # velocity scale
    L :: Number # length scale
    ϵ :: Number # kernel width
    flow :: Flow
    body :: AbstractBody
    pois :: AbstractPoisson
    struc 
    cpl :: AbstractCoupling
    # coupling variables
    forces :: AbstractArray
    pnts :: AbstractArray
    # storage for iterations
    uˢ :: AbstractArray
    pˢ :: AbstractArray
    dˢ :: AbstractArray
    vˢ :: AbstractArray
    aˢ :: AbstractArray
    xˢ :: AbstractArray
    ẋˢ :: AbstractArray
    function CoupledSimulation(dims::NTuple{N}, u_BC, L::Number,
                               body::AbstractBody, struc, Coupling;
                               Δt=0.25, ν=0., g=nothing, U=nothing, ϵ=1, perdir=(),
                               uλ=nothing, exitBC=false, T=Float32, mem=Array, kw...) where N
        
        @assert !(isa(u_BC,Function) && isa(uλ,Function)) "`u_BC` and `uλ` cannot be both specified as Function"
        @assert !(isnothing(U) && isa(u_BC,Function)) "`U` must be specified if `u_BC` is a Function"
        isa(u_BC,Function) && @assert all(typeof.(ntuple(i->u_BC(i,T(0)),N)).==T) "`u_BC` is not type stable"
        uλ = isnothing(uλ) ? ifelse(isa(u_BC,Function),(i,x)->u_BC(i,0.),(i,x)->u_BC[i]) : uλ
        U = isnothing(U) ? √sum(abs2,u_BC) : U # default if not specified
        flow = Flow(dims,u_BC;uλ,Δt,ν,g,T,f=mem,perdir,exitBC); measure!(flow,body;ϵ)
        force = zeros((3,length(uv_integration(struc)))) ###
        Ns = size(struc.u[1]); Nn = size(body.curve.pnts)
        uˢ, pˢ = zero(flow.u) |> mem, zero(flow.p) |> mem
        dˢ, vˢ, aˢ = zeros(Ns) |> mem, zeros(Ns) |> mem, zeros(Ns) |> mem
        pnts,xˢ,ẋˢ = zeros(Nn) |> mem, zeros(Nn) |> mem, zeros(Nn) |> mem
        new(U,L,ϵ,flow,body,MultiLevelPoisson(flow.p,flow.μ₀,flow.σ;perdir),struc,
            Coupling(pnts,force;kw...),force,pnts,uˢ,pˢ,dˢ,vˢ,aˢ,xˢ,ẋˢ)
    end
end

"""
    sim_time(sim::CoupledSimulation,t_end)

    
"""
function sim_step!(sim::CoupledSimulation,t_end;verbose=true,maxStep=15,kwargs...)
    t = sum(sim.flow.Δt[1:end-1])
    # @show t
    while t < t_end*sim.L/sim.U
        store!(sim); iter=1
        # @show t
        while true
            # update structure
            solve_step!(sim.struc,sim.forces,sim.flow.Δt[end]/sim.L)
            # update body
            ParametricBodies.update!(sim.body,u⁰+L*sim.pnts,sim.flow.Δt[end])
            # update flow
            measure!(sim,t); mom_step!(sim.flow,sim.pois)
            # compute new coupling variable
            sim.forces.=force(sim.body,sim.flow); sim.pnts.=points(sim.struc)
            # check convergence and accelerate
            verbose && print("    iteration: ",iter)
            converged = update!(sim.cpl,sim.pnts,sim.forces;kwargs...)
            # revert!(xᵏ,sim.pnts,sim.forces,sim.cpl.subs)
            (converged || iter+1 > maxStep) && break
            # revert if not convergend
            revert!(sim); iter+=1
        end
        #update time
        t += sim.flow.Δt[end]
        verbose && println("tU/L=",round(t*sim.U/sim.L,digits=4),
                           ", Δt=",round(sim.flow.Δt[end],digits=3))
    end
end

"""
    store!(sim::CoupledSimulation)

Checkpoints that state of a coupled simulation for implicit coupling.
"""
function store!(sim::CoupledSimulation)
    sim.uˢ .= sim.flow.u
    sim.pˢ .= sim.flow.p
    sim.dˢ .= sim.struc.u[1]
    sim.vˢ .= sim.struc.u[2]
    sim.aˢ .= sim.struc.u[3]
    sim.xˢ .= sim.body.curve.pnts
    sim.ẋˢ .= sim.body.dotS.pnts
end

"""
    revert!(sim::CoupledSimulation)

Reverts to the previous state of a coupled simulation for implicit coupling.
"""
function revert!(sim::CoupledSimulation)
    sim.flow.u .= sim.uˢ
    sim.flow.p .= sim.pˢ
    pop!(sim.flow.Δt) 
    # pop the last two iter in the poisson solver
    pop!(sim.pois.n); pop!(sim.pois.n)
    sim.struc.u[1] .= sim.dˢ
    sim.struc.u[2] .= sim.vˢ
    sim.struc.u[3] .= sim.aˢ
    sim.body.curve.pnts .= sim.xˢ
    sim.body.dotS.pnts .= sim.ẋˢ
end