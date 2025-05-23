using WaterLily
using ParametricBodies
using ParametricBodies: _pforce, _vforce
using Splines
using StaticArrays
using LinearAlgebra
using BiotSavartBCs
using JLD2
include("../ext/vis.jl")
include("Coupling.jl")
include("utils.jl")

## Simulation parameters
L=2^5
Re=200; U=1
ϵ=0.5; thk=2ϵ+√2
B = 1000       # scaled stiffness/buouancy
density(ξ) = 0.75+2.5ξ # mass for correct mass ratio
g = U^2/(∫dξ(density)-1.0)/thk # gravity force
EI(ξ) = (∫dξ(density)-1.0)*thk/B #
EI(a) = 0.001    # Cauhy number
EA(ξ) = 50_000.0  # make inextensible
@show g, EI, EA

# Mesh property
numElem=10
degP=3
ptLeft = 0.0
ptRight = 1.0
# mesh
mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

# make a structure
gravity(i,ξ::T) where T = i==2 ? convert(T,-g) : zero(T)
struc = DynamicFEOperator(mesh, gauss_rule, EI, EA, 
                          [], [], ρ=density, g=gravity; ρ∞=0.0)

# construct from mesh, this can be tidy
u⁰ = SMatrix{2,size(mesh.controlPoints,2)}(mesh.controlPoints[1:2,:].*L.+[1.5L,L].+1.5)

# flow sim
body = DynamicNurbsBody(NurbsCurve(u⁰,mesh.knots,mesh.weights);thk,boundary=false)

# force function
integration_points = uv_integration(struc)

# make a coupled sim
global U_cm = SA[0.,0.]
global X = SA[0.,0.]
Ut(i,t) = -U_cm[i]
sim = CoupledSimulation((4L,4L),Ut,L,body,struc,IQNCoupling;
                         U,ν=U*L/Re,ϵ,T=Float64,relax=0.05,maxCol=12)

# Multilevel Biot-Savart
ω = MLArray(sim.flow.σ)

# sime time
t₀ = round(sim_time(sim)); duration = 1.0; step = 0.2
iterations = []
# time loop
@time @gif for tᵢ in range(t₀,t₀+duration;step)
    
    # update until time tᵢ in the background
    while WaterLily.time(sim) < tᵢ*sim.L/sim.U
        
        println("  t=$(round(sim_time(sim),digits=2)), Δt=$(round(sim.flow.Δt[end],digits=2))")

        # save at start of iterations
        store!(sim); iter=1;

        # get position and velocity of CM of structure
        global X = SA[mean(points(sim.struc).*sim.L;dims=2)...]
        global U_cm = SA[mean(vⁿ(sim.struc);dims=2)...]
        @show X, U_cm

        # iterative loop
        while true

            # integrate once in time
            solve_step!(sim.struc, sim.forces, sim.flow.Δt[end]/sim.L)
            
           # update flow, this requires scaling the displacements
           sim.body = ParametricBodies.update!(sim.body,u⁰.+(sim.L*sim.pnts.-X),sim.flow.Δt[end])
           measure!(sim); biot_mom_step!(sim.flow,sim.pois,ω)

            # get new coupling variable
            sim.pnts .= points(sim.struc)
            sim.forces .= force(sim.body,sim.flow)

            # accelerate coupling
            print("    iteration: ",iter)
            converged = update!(sim.cpl, sim.pnts, sim.forces, 0.0)

            # check for convengence
            (converged || iter+1 > 20) && break

            # if we have not converged, we must revert
            revert!(sim); iter += 1
        end
        push!(iterations,iter)
    end

    N = size(sim.flow.σ)
    get_omega!(sim); flood(sim.flow.σ;shift=(X[1],X[2]),clims=(-10,10),dpi=300) #plot_vorticity(sim.flow.σ, limit=10)
    plot!(sim.body.curve;add_cp=false,shift=(X[1],X[2]))
    xlims!(-2N[1],3N[1]); ylims!(-6N[2],N[2])
    plot!(title="tU/L $tᵢ")
    
    # check that we are still inside the domain
    pos = mean(sim.body.curve.pnts;dims=2)
    !(all(pos.>[0.,0.]) && all(pos.<size(sim.flow.p))) && break
end