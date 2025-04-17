using WaterLily
using ParametricBodies
using Splines
using StaticArrays
using LinearAlgebra
using BiotSavartBCs
using JLD2
include("Coupling.jl")
include("utils.jl")

# Simulation parameters
L=2^5
Re=200; U=1
ϵ=0.5; thk=2ϵ+√2
Ca = 1000       # Cauchy number
ρ,mᵨ = 2.0,1.5
density(ξ) = (ξ*mᵨ - 0.5*mᵨ + ρ) # mass for correct mass ratio 2
g = U^2/(∫dξ(density)-1.0)/thk # gravity force
EI(a) = 1.0/Ca       # set stiffness
EA(ξ) = 500_000.0  # make inextensible
@show g, ∫dξ(EI), ∫dξ(EA), ∫dξ(density)

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
t₀ = round(sim_time(sim)); duration = 50.0; step = 0.2
iterations = []

# store initial state
# file = jldopen("fillament_8.jld2","w")
# count = 0

# time loop
@time for tᵢ in range(t₀,t₀+duration;step)
    
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

    # global count += 1
    # group = JLD2.Group(file,"frame_$count")
    # @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
    # group["ω"] = sim.flow.σ
    # group["X"] = X
    # group["U"] = U_cm
    # group["t"] = sim_time(sim)
    # group["pnts"] = sim.body.surf.pnts
    
    # check that we are still inside the domain
    pos = bbox(sim.body.locate)
    !(all(pos[1].>0) && all(pos[2].<size(sim.flow.p))) && break
end
# close(file)