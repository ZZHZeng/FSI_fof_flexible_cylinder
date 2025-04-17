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

degree = 0
arc_length = 0
function arc_sim(R,thk,curve_area, α = π/16, U=1, Re=100)
    arc_area = curve_area - π*(thk/2)^2
    global arc_length = arc_area/thk
    curve(θ,t) = SA[cos(θ),sin(θ)] # ξ-space: angle=0,center=0,radius=1
    Rotate = SA[cos(α) -sin(α); sin(α) cos(α)]
    center = SA[R,-2R÷5]
    scale = R
    map(x,t) = Rotate*(x-center)/scale # map from x-space to ξ-space
    global degree = 1/3*π
    arc = HashedBody(curve,(π/3,π/3+degree),thk=thk,boundary=false,map=map)
    Simulation((3R,R),(U,0),R,body=arc,ν=U*R/Re)
end

# pforce,vforce,drag = [],[],[]
area_simu = []
area_analy = [] 
vol1 = 0

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


duration,tstep = 0.1,0.1

function run(R,thk,curve_area)
    sim = arc_sim(R,thk,curve_area)
    name = "2Dark_R$(R)_thk$(thk)_curvearea$(curve_area)"
    wr = vtkWriter(name; attrib=custom_attrib)
    for tᵢ in range(0,duration;step=tstep)

        WaterLily.apply!(x->x[1], sim.flow.p)
        global vol1 = pressure_force(sim)[1]
        sim.flow.p .= 0
        push!(area_simu,vol1)
        push!(area_analy,2*π*sim.L*(degree/(2π))*thk + π*(thk/2)^2)

        # update until time tᵢ in the background
        sim_step!(sim,tᵢ,remeasure=false)
        # print time step
        println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
        write!(wr, sim)
    end
    close(wr)
    plot(range(0,duration;step=tstep),area_simu,label="area_simu",xlabel="tU/L",ylabel="arc_area_L$(R)_thk$(thk)")
    plot!(range(0,duration;step=tstep),area_analy,label="area_analy")
    savefig(name*"png")
end

run(512,4,120)