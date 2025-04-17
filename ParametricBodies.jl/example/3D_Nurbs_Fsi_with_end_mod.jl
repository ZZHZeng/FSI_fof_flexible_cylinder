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

function cable(L,m,n,Re;U=1,mem=Array)
    cps = SA[0   1    2
             0.0 0.0 0.0
             0.0 0.0 0.0]*L .+ [1,n÷2,n÷2]*L
    curve = BSplineCurve(cps;degree=2)
    function thickness(u;thick=1,tapper=0.5)
        u<=tapper ? thick : ((thick/2/(tapper-1))*u-thick/2/(tapper-1))*2
    end
    body = DynamicNurbsBody(curve;thk=(u)->thickness(u;thick=6,tapper=0.8),boundary=false)  

    Simulation((m*L,n*L,n*L),(U,0,0),L;U,ν=U*L/Re,body,T=Float64,mem,exitBC=true)
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
vol1 = 0

function run(L,m,n,Re)
    sim = cable(L,m,n,Re)#mem=CuArray);
    t₀,duration,tstep = sim_time(sim),0.1,0.1;
    name = "3D_cable_thk=6_horizontal"
    wr = vtkWriter(name; attrib=custom_attrib)

    for tᵢ in range(t₀,t₀+duration;step=tstep)
        # update until time tᵢ in the background
        t = sum(sim.flow.Δt[1:end-1])
        while t < tᵢ*sim.L/sim.U
            # random update
            new_pnts = (SA[0     1   2
                           0.0  0.0 0.0
                           0.0  0.0 0.0] .+ [1,n÷2,n÷2])*L
            sim.body = update!(sim.body,new_pnts,sim.flow.Δt[end])
            measure!(sim,t)
            mom_step!(sim.flow,sim.pois) # evolve Flow
            t += sim.flow.Δt[end]
        end
        println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
        write!(wr, sim)
    end
    close(wr)
end

run(32,6,4,250) #L,m,n,Re