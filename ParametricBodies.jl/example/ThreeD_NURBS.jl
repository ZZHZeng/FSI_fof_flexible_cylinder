using WaterLily
using ParametricBodies
using StaticArrays
using CUDA
using WriteVTK
# parameters
function dynamicSpline(;L=2^6,Re=250,U =1,ϵ=4,thk=12,mem=Array)
    # define a flat plat at and angle of attack
    cps = SA[-1   0   1
             0.5 0.25 0
             0    0   0]*L .+ [2L,L,8]

    # needed if control points are moved
    curve = BSplineCurve(cps;degree=2)

    # use BDIM-σ distance function, make a body and a Simulation
    body = DynamicNurbsBody(curve;thk=(u)->thk,boundary=false)
    Simulation((6L,2L,16),(U,0,0),L;U,ν=U*L/Re,body,T=Float64,mem)
end

# make a writer with some attributes
velocity(a::Simulation) = a.flow.u |> Array;
pressure(a::Simulation) = a.flow.p |> Array;
body(a::Simulation) = (measure_sdf!(a.flow.σ, a.body); 
                       a.flow.σ |> Array;)
custom_attrib = Dict(
    "Velocity" => velocity,
    "Pressure" => pressure,
    "Body" => body
)# this maps what to write to the name in the file

# # intialize
sim = dynamicSpline(mem=Array);
t₀,duration,tstep = sim_time(sim),0.3,0.1;
wr = vtkWriter("ThreeD_nurbs"; attrib=custom_attrib)

# run
for tᵢ in range(t₀,t₀+duration;step=tstep)

    # update until time tᵢ in the background
    t = sum(sim.flow.Δt[1:end-1])
    while t < tᵢ*sim.L/sim.U
        δx = SA[-1   0   1
                0.5 0.25 0
                0 0.00sin(2π*t/sim.L) 0.0sin(2π*t/sim.L)]*sim.L .+ [2sim.L,sim.L,8]
        sim.body = update!(sim.body,δx,sim.flow.Δt[end])
        # random update
        measure!(sim,t)
        mom_step!(sim.flow,sim.pois) # evolve Flow
        t += sim.flow.Δt[end]
    end

    # print time step
    write!(wr, sim)
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
end
close(wr)