using WaterLily
using ParametricBodies
using Splines
using StaticArrays
using LinearAlgebra

include("../ext/vis.jl")
include("Coupling.jl")
include("utils.jl")

# Material properties and mesh
numElem=4
degP=3
ptLeft = 0.0
ptRight = 1.0

# parameters
EI = 0.0015         # Cauhy number
EA = 100_000.0  # make inextensible
density(ξ) = 0.5  # mass ratio

# mesh
mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

# boundary conditions
Dirichlet_BC = [
    Boundary1D("Dirichlet", ptLeft, 0.0; comp=1),
    Boundary1D("Dirichlet", ptLeft, 0.0; comp=2)
]
Neumann_BC = []

# make a structure
struc = DynamicFEOperator(mesh, gauss_rule, EI, EA, 
                          Dirichlet_BC, Neumann_BC, ρ=density; ρ∞=0.0)

## Simulation parameters
L=2^5
Re=200
U=1
ϵ=0.5
thk=2ϵ+√2

# construct from mesh, this can be tidy
u⁰ = SMatrix{2,size(mesh.controlPoints,2)}(mesh.controlPoints[1:2,:].*L.+[2L,3L].+1.5)

# flow sim
body = DynamicNurbsBody(NurbsCurve(u⁰,mesh.knots,mesh.weights);thk=(u)->thk,boundary=false)

# force function
integration_points = uv_integration(struc) ## utils.jl

# make a coupled sim
a₀ = 0.5
Ut(i,t::T) where T = i==1 ? convert(T,a₀*t+(1.0+tanh(31.4*(t-1.0/a₀)))/2.0*(1-a₀*t)) : zero(T) #?
sim = CoupledSimulation((8L,6L),Ut,L,body,struc,IQNCoupling;
                         U,ν=U*L/Re,ϵ,T=Float64,relax=0.05,maxCol=6)

# sime time
t₀ = round(sim_time(sim)); duration = 0.2; step = 0.2

# time loop
@time @gif for tᵢ in range(t₀,t₀+duration;step)
    
    # update until time tᵢ in the background
    t = sum(sim.flow.Δt[1:end-1])
    
    while t < tᵢ*sim.L/sim.U
        
        println("  tᵢ=$tᵢ, t=$(round(t,digits=2)), Δt=$(round(sim.flow.Δt[end],digits=2))")

        # save at start of iterations
        store!(sim); iter=1;

        # iterative loop
        while true

            #  integrate once in time
            solve_step!(sim.struc, sim.forces, sim.flow.Δt[end]/sim.L) # apply FEA solver
            
            # update flow, this requires scaling the displacements
            sim.body = ParametricBodies.update!(sim.body,u⁰.+L*sim.pnts,sim.flow.Δt[end]) 
            measure!(sim,t); mom_step!(sim.flow,sim.pois) # body and flow from flow solver.

            # get new coupling variable
            sim.pnts .= points(sim.struc) # use FEA solver to obtain the displacement of control points. Here is the IGA.
            sim.forces .= force(sim.body,sim.flow) # use flow solver to obtain the external force. this requires modification.

            # accelerate coupling
            print("    iteration: ",iter)
            converged = update!(sim.cpl, sim.pnts, sim.forces; atol=1e-2) # coupling.jl

            # check for convengence
            (converged || iter+1 > 50) && break

            # if we have not converged, we must revert
            revert!(sim); iter += 1
        end
        # println(" beam length: ", ParametricBodies.integrate(sim.body.curve,(0.,1.)))
        # finish the time step
        t += sim.flow.Δt[end]
    end

    # println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
    get_omega!(sim); plot_vorticity(sim.flow.σ, limit=10)
    plot!(sim.body.curve,shift=[2.0,2.0])
    plot!(title="tU/L $tᵢ")
end