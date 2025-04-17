using WaterLily
using ParametricBodies
using Splines
using StaticArrays
using LinearAlgebra
using WriteVTK
using CSV,Tables,DataFrames
include("../ext/vis.jl")
include("Coupling.jl")
include("utils.jl")

ptLeft = 0.0
ptRight = 1.0

# Geometry
AR = 15.4 / 0.653 # aspect ratio of cylinder #####
ARR = round(Int,AR)

thk1 = 7.556975354821782
thk2 = 0
thkk2 = round(Int,thk2)

tapper_leng = 31 / 17 # dimension_less tapper length
tapper = AR / (AR + tapper_leng) 
tl = "31_17"

L = thk1 * AR / tapper
TL = L * (ptRight - ptLeft) # total length with tapper end in numerical space
LL = round(Int,L)

# Material properties and mesh
numElem = round(Int, 12*AR/(15.4/0.653))
numE = round(Int,numElem)
degP = 3

# exp data
rho_exp = 1026 # sea water
μ_exp = 1.22 * 10^(-3) # 15 ceils, sea water

D_exp = 0.03 # m
L_exp = AR * D_exp # m
I_exp = π*D_exp^4/64

U_exp = 5*0.5144 # m/s #####
uu = round(Int,10*U_exp) # for the file name
E_exp = 12*10^6 # Pa
EI_exp = E_exp*I_exp
Re_exp = rho_exp * D_exp * U_exp / μ_exp ## Reynolds number
Ca_exp = EI_exp / I_exp / (rho_exp*U_exp^2) ## Cauchy number 
Ca_exp_expand = EI_exp / (rho_exp*U_exp^2*L_exp^4)

# simulation parameters
U = 1 * (ptRight - ptLeft)
E_sim = Ca_exp * (1*U^2)
EI_sim = E_sim * (π*(thk1)^4) / 64 
EI_sim_test = Ca_exp_expand * (1*U^2*(TL*tapper)^4) 
EI = EI_sim / TL^3 # EI for FEA solver
EA = 100_000.0  # make inextensible
density(ξ) = 1 # assume nutral buoyancy, density of solid structure
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

# make a structure
struc = DynamicFEOperator(mesh, gauss_rule, EI, EA, 
                          Dirichlet_BC, Neumann_BC, ρ=density; ρ∞=0.0)

## Simulation parameters
Re=Re_exp
# Re=Re/10 # change Re only if you need
ϵ=0.5

# construct from mesh, this can be tidy
u⁰ = SMatrix{3,size(mesh.controlPoints,2)}(mesh.controlPoints[1:3,:].*L.+[24,24,24].+0.5) #controlPoints
half_doma = 24.5

# flow sim
function thickness(u,thick1,thick2,tapper)
    u<=tapper ? thick1 : ((((thick2)/2-(thick1)/2)/(1-tapper))*u + ((thick2)/2-((thick2)/2-(thick1)/2)/(1-tapper)))*2
end
body = DynamicNurbsBody(NurbsCurve(u⁰,mesh.knots,mesh.weights);thk=(u)->thickness(u,thk1,thk2,tapper),boundary=false)

# force function
integration_points = uv_integration(struc)/(ptRight-ptLeft) ## utils.jl

# make a coupled sim
sim = CoupledSimulation((280,48,48),(U,0,0),L,body,struc,IQNCoupling;
                         U,ν=U*(thk1)/Re,ϵ,T=Float64,relax=0.05,maxCol=6)

q_poke = 0.01

# sim time
t₀ = round(sim_time(sim)); duration = 30; step = 0.2
# write para_view
velocity(a::CoupledSimulation) = a.flow.u |> Array;
pressure(a::CoupledSimulation) = a.flow.p |> Array;
_body(a::CoupledSimulation) = (measure_sdf!(a.flow.σ, a.body, WaterLily.time(a.flow)); 
                        a.flow.σ |> Array;)
custom_attrib = Dict(
    "u" => velocity, "p" => pressure, "d" => _body)

name = "ExpU$(uu)_L$(LL)_Gpnts$(numE)_thk2$(thkk2)_AR$(ARR)_CF_RealCase_truncErrTest123"*tl
wr = vtkWriter(name; attrib=custom_attrib)

disy_1 = []
disz_1 = []
disy_2 = []
disz_2 = []
disy_3 = []
disz_3 = []
disy_4 = []
disz_4 = []
disy_5 = []
disz_5 = []
disy_6 = []
disz_6 = []
disy_7 = []
disz_7 = []
disy_8 = []
disz_8 = []
forces_y = []
forces_z = []
ttime = []
# time loop
@time for tᵢ in range(t₀,t₀+duration;step)
    
    # update until time tᵢ in the background
    t = sum(sim.flow.Δt[1:end-1])
    
    while t < tᵢ*sim.L/sim.U
        
        println("  tᵢ=$tᵢ, t=$(round(t,digits=2)), Δt=$(round(sim.flow.Δt[end],digits=2))")

        # save at start of iterations
        store!(sim); iter=1;

        # iterative loop
        while true

            #  integrate once in time
            solve_step!(sim.struc, sim.forces[2:3,:], sim.flow.Δt[end]/sim.L) # apply FEA solver
            
            # update flow, this requires scaling the displacements
            sim.body = ParametricBodies.update!(sim.body,u⁰.+ L*sim.pnts,sim.flow.Δt[end]) ###!!!
            measure!(sim,t); mom_step!(sim.flow,sim.pois) # body and flow from flow solver.
            
            # apply!(x->x[2],sim.flow.p)
            # get new coupling variable
            sim.pnts[2:3,:] = points(sim.struc) # use FEA solver to obtain the displacement of control points. Here is the IGA.
            sim.forces = force_fea(sim,degP+1,(thk1)/2,(thk2)/2,numElem,sim.L,tapper)
            if tᵢ <= 0.2
                sim.forces[3,:] .+= q_poke 
            end

            # accelerate coupling
            print("    iteration: ",iter)
            converged = update!(sim.cpl, sim.pnts, sim.forces; atol=1e-2) # coupling.jl

            # check for convengence
            (converged || iter+1 > 50) && break

            # if we have not converged, we must revert
            revert!(sim); iter += 1
        end
        t += sim.flow.Δt[end]
        push!(disy_1, sim.body.curve.(0.125)[2] .- half_doma)
        push!(disz_1, sim.body.curve.(0.125)[3] .- half_doma)
        push!(disy_2, sim.body.curve.(0.25)[2] .- half_doma)
        push!(disz_2, sim.body.curve.(0.25)[3] .- half_doma)
        push!(disy_3, sim.body.curve.(0.375)[2] .- half_doma)
        push!(disz_3, sim.body.curve.(0.375)[3] .- half_doma)
        push!(disy_4, sim.body.curve.(0.5)[2] .- half_doma)
        push!(disz_4, sim.body.curve.(0.5)[3] .- half_doma)
        push!(disy_5, sim.body.curve.(0.625)[2] .- half_doma)
        push!(disz_5, sim.body.curve.(0.625)[3] .- half_doma)
        push!(disy_6, sim.body.curve.(0.75)[2] .- half_doma)
        push!(disz_6, sim.body.curve.(0.75)[3] .- half_doma)
        push!(disy_7, sim.body.curve.(0.875)[2] .- half_doma)
        push!(disz_7, sim.body.curve.(0.875)[3] .- half_doma)
        push!(disy_8, sim.body.curve.(1.0)[2] .- half_doma)
        push!(disz_8, sim.body.curve.(1.0)[3] .- half_doma)
        push!(forces_y, sim.forces[2,:])
        push!(forces_z, sim.forces[3,:])
        push!(ttime,t*sim.U/sim.L)
    end
    write!(wr, sim)
    CSV.write(name*"_displacement_"*".csv", Tables.table(hcat(ttime,disy_1,disz_1,disy_2,disz_2,disy_3,disz_3,disy_4,disz_4,disy_5,disz_5,disy_6,disz_6,disy_7,disz_7,disy_8,disz_8,forces_y,forces_z)), writeheader=false)
end
close(wr)

# uu=integration_points[48]
# thkk = (((thk2)/2-(thk1)/2)/(1-tapper))*uu + ((thk2)/2-((thk2)/2-(thk1)/2)/(1-tapper))
# α = atan(((thk1)/2-(thk2)/2)/((1-tapper)*L))
# dV = -thkk^2*π * cos(α)
# @show(sim.forces[:,48])
# @show(dV)