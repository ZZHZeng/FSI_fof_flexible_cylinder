###
using Pkg
Pkg.develop(path="E:/Graduation_Assignment/waterlily/WaterLily.jl")
using WaterLily
println(pathof(WaterLily))
###
using ParametricBodies
using StaticArrays
using CUDA
using WriteVTK
using Plots

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

function circle_hash(L,ar,Re;U=1,T=Float32,mem=Array)
    b = T(ar)
    ellipse(θ,t) = 0.5f0L*SA[cos(θ),b*sin(θ)]
    curve = HashedBody(ellipse,(0,2π);T)
    function map(x::SVector{3},t) # Map the 3D space into a 2D curve 
        y = x - SA[2L,2L,2L]
        r = √(y[2]^2 + y[3]^2)
        return SA[y[1],r] 
    end
    body = ParametricBody(curve; map,scale=1f0)
    Simulation((6L,4L,4L),(U,0,0),L;ν=U*L/Re,body,T,mem,exitBC=true)
end

function circle_nurbs(L,ar,Re;U=1,mem=Array,T=Float32)

    cps = SA_F32[1 1 0 -1 -1 -1  0  1 1
                0 1 1  1  0 -1 -1 -1 0]*L/2
    weights = SA_F32[1.,√2/2,1.,√2/2,1.,√2/2,1.,√2/2,1.]
    knots =   SA_F32[0,0,0,1/4,1/4,1/2,1/2,3/4,3/4,1,1,1]
    curve = NurbsCurve(cps,knots,weights)
    function map(x::SVector{3},t) # Map the 3D space into a 2D curve 
        y = x - SA[2L,2L,2L]
        r = √(y[2]^2 + y[3]^2)
        return SA[y[1],r] 
    end
    body = ParametricBody(curve; map,scale=1)
    Simulation((6L,4L,4L),(U,0,0),L;U,ν=U*L/Re,body=body,T=T,mem=mem,exitBC=true)
end

function circle_sdf(L,ar,Re;U=1,mem=Array,T=Float32)
    center,radius = [2,2,2]*L,0.5f0L
    Body = AutoBody((x,t)->√sum(abs2, x .- center) - radius)
    Simulation((6L,4L,4L),(U,0,0),L;U,ν=U*L/Re,body=Body,T=T,mem=mem,exitBC=true) 
end

velocity(a::Simulation) = a.flow.u |> Array;
pressure(a::Simulation) = a.flow.p |> Array;
#NDS(a::Simulation) = a.flow.f ./ a.flow.p |> Array;
vforce(a::Simulation) = a.flow.f |> Array;
body(a::Simulation) = (measure_sdf!(a.flow.σ, a.body); 
                       a.flow.σ |> Array;)
custom_attrib = Dict(
    "Velocity" => velocity,
    "Pressure" => pressure,
    #"NDS" => NDS,
    "vforce" => vforce,
    "Body" => body
)# this maps what to write to the name in the file


function run(L,ar,Re) 

    A = 4/3*π*(L/2)^3
    duration,tstep = 5, 0.1
    sim = circle_nurbs(L,ar,Re)

    name = "3Dsphere_$(L)_100_v4_nurbs"
    wr = vtkWriter(name; attrib=custom_attrib,dir="E:/Graduation_Assignment/waterlily/ParametricBodies.jl/vtk_data")
    pforce,vforce,drag = [],[],[]

    for tᵢ in range(0,duration;step=tstep)
        sim_step!(sim,tᵢ,remeasure=false)
        cp = 2*pressure_force(sim)[1]/A
        cv = 2*viscous_force(sim)[1]/A
        push!(pforce,cp)
        push!(vforce,cv)
        push!(drag,cp+cv)
        println("3D: tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
        write!(wr, sim)
    end
    close(wr)

    @show(drag)
    @show(pforce)
    @show(vforce)
    plot(range(0,duration;step=tstep),drag,label="Cd",xlabel="tU/L",ylabel="C_force")
    plot!(range(0,duration;step=tstep),vforce,label="Cv")
    plot!(range(0,duration;step=tstep),pforce,label="Cp")
    savefig(name*".png")
end 

run(2^6,1,10000)
