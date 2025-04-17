using WaterLily
using ParametricBodies
using StaticArrays
using CUDA
using WriteVTK

using WaterLily: @loop,inside,inside_u,nds,∇²u
function pressure_force(sim,t=WaterLily.time(sim.flow),T=promote_type(Float64,eltype(sim.flow.p)))
    sim.flow.f .= zero(eltype(sim.flow.p))
    @loop sim.flow.f[I,:] .= sim.flow.p[I]*nds(sim.body,loc(0,I,T),t) over I ∈ inside(sim.flow.p)
    sum(T,sim.flow.f,dims=ntuple(i->i,ndims(sim.flow.p)))[:] |> Array
end
function viscous_force(sim,t=WaterLily.time(sim.flow),T=promote_type(Float64,eltype(sim.flow.u)))
    sim.flow.f .= zero(eltype(sim.flow.u))
    @loop sim.flow.f[I,:] .= -sim.flow.ν*∇²u(I,sim.flow.u)*nds(sim.body,loc(0,I,T),t) over I ∈ inside_u(sim.flow.u)
    sum(T,sim.flow.f,dims=ntuple(i->i,ndims(sim.flow.u)-1))[:] |> Array
end

# parameters
function dynamicSpline(;L=2^6,Re=250,U =1,thk=4,mem=Array)
    # define a flat plat at and angle of attack
    cps = SA[-1   0   1
             0.5 0.25 0
             0    0   0]*L .+ [2L,2L,8]

    # needed if control points are moved
    curve = BSplineCurve(cps;degree=2)

    # use BDIM-σ distance function, make a body and a Simulation
    body = DynamicNurbsBody(curve;thk=thk,boundary=false)
    Simulation((6L,4L,16),(U,0,0),L;U,ν=U*L/Re,body,T=Float64,mem)
end

velocity(a::Simulation) = a.flow.u |> Array;
pressure(a::Simulation) = a.flow.p |> Array;
NDS(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds(a.body,loc(0,I),0.0) over I in inside(a.flow.p);
                     a.flow.f |> Array;)
_normal(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= measure(a.body,loc(0,I),0.0)[2] over I in inside(a.flow.p);
                     a.flow.f |> Array;) 
body(a::Simulation) = (measure_sdf!(a.flow.σ, a.body); 
                       a.flow.σ |> Array;)
custom_attrib = Dict(
    "Velocity" => velocity,
    "Pressure" => pressure,
    "NDS" => NDS, ## may be wrong
    "Body" => body,
    "normal" => _normal
)# this maps what to write to the name in the file

# # intialize
sim = dynamicSpline(mem=Array);
t₀,duration,tstep = sim_time(sim),0.1,0.1;
wr = vtkWriter("ThreeD_nurbs"; attrib=custom_attrib)

pforce,vforce,drag = [],[],[]
vol = []
vol_analy = [] 
vol1 = 0

# run
for tᵢ in range(t₀,t₀+duration;step=tstep)

    # update until time tᵢ in the background
    t = sum(sim.flow.Δt[1:end-1])
    while t < tᵢ*sim.L/sim.U
        δx = SA[-1   0   1
                0.5 0.25 0
                0    0   0]*sim.L .+ [2sim.L,2sim.L,8]
        sim.body = update!(sim.body,δx,sim.flow.Δt[end])
        # random update
        measure!(sim,t)
        WaterLily.apply!(x->x[1], sim.flow.p) 
        global vol1 = pressure_force(sim)[1] # volume measured by pressure_force
        sim.flow.p .= 0 
        mom_step!(sim.flow,sim.pois) # evolve Flow
        t += sim.flow.Δt[end]
    end
    push!(vol,vol1)

    # print time step
    write!(wr, sim)
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
end
close(wr)
thk=4
@show(vol1)
@show(2.062*sim.L*(thk/2)^2*π+4/3*π*(thk/2)^3)