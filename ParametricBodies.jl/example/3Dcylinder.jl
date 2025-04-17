using WaterLily
using ParametricBodies
using StaticArrays
using Plots
using WriteVTK
# using CUDA

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

function cylinder_sdf(L,Re,m,n;U=1,mem=Array,T=Float32)
    cps = SA_F32[1 1 0 -1 -1 -1  0  1 1
                0 1 1  1  0 -1 -1 -1 0]*L/2 
    weights = SA_F32[1.,√2/2,1.,√2/2,1.,√2/2,1.,√2/2,1.]
    knots =   SA_F32[0,0,0,1/4,1/4,1/2,1/2,3/4,3/4,1,1,1]
    # make a nurbs curve and a body for the simulation
    circle = NurbsCurve(cps,knots,weights)
    function map(x::SVector{3},t) # Map the 3D space into a 2D curve 
        y = x - SA[2L,n÷2*L,n÷2*L]
        return SA[y[2],y[3]] 
    end
    cylinder = ParametricBody(circle;map,scale=1)
    Simulation((m*L,n*L,n*L),(U,0,0),L;U,ν=U*L/Re,body=cylinder,T,mem=mem,exitBC=true) 
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
    "NDS" => NDS, 
    "Body" => body,
    "normal" => _normal
)# this maps what to write to the name in the file

pforce,vforce,drag = [],[],[]
vol = []
vol_analy = [] 
vol = 0
function run(L,Re,m,n)
    sim = cylinder_sdf(L,Re,m,n)
    duration,tstep = 0.1,0.1
    name = "3Dcylinder"
    wr = vtkWriter(name; attrib=custom_attrib)
    for tᵢ in range(0,duration;step=tstep)

        WaterLily.apply!(x->x[1], sim.flow.p)
        global vol1 = pressure_force(sim)[1]
        sim.flow.p .= 0

        # update until time tᵢ in the background
        sim_step!(sim,tᵢ,remeasure=false)
        # print time step
        println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
        write!(wr, sim)
    end
    close(wr)
end

run(2^3,250,8,6)