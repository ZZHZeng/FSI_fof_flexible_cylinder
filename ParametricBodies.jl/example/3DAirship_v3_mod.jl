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

function characteristic_length(L) # f<9pi/(6*400)
        f = 0.00391/(27.953*0.0254)^3
        @assert f<9pi/(6*400)
    return f^(1/3)*L
end

function airship(L; Re_exp=108500,U=1,T=Float32,mem=Array)
    length = characteristic_length(L)
    @assert L%32==0 # ensures your L is sufficiently big pow of 2
    D = 3L÷16 # safe over estimate
    cps = SA{T}[1.0  0.819232  0.38  0.15   0.0    0.0  
                0.0  0.054     0.092 0.079  0.03   0.0 ] * L
    curve = BSplineCurve(cps; degree=4)
    function map(x::SVector{3},t) # Map the 3D space into a 2D curve 
        y = x - SA[L÷2,6D,6D]
        r = √(y[2]^2 + y[3]^2)
        return SA[y[1],r] 
    end
    Body = ParametricBody(curve; map,scale=1)
    Simulation((5L÷2,12D,12D),(U,0,0),L; ν=U*length/Re_exp,body=Body,T=T,mem=mem,exitBC=true)
end

velocity(a::Simulation) = a.flow.u |> Array;
pressure(a::Simulation) = a.flow.p |> Array;
NDS(a::Simulation) = a.flow.f ./ a.flow.p |> Array;
body(a::Simulation) = (measure_sdf!(a.flow.σ, a.body); 
                       a.flow.σ |> Array;)
custom_attrib = Dict(
    "Velocity" => velocity,
    "Pressure" => pressure,
    "NDS" => NDS,
    "Body" => body
)# this maps what to write to the name in the file


function run(L,duration,tstep) 

    A = characteristic_length(L)^2
    sim = airship(L)

    WaterLily.apply!(x->x[1], sim.flow.p)
    V = pressure_force(sim)[1]
    #@assert abs(A-V^(2/3))<10 
    sim.flow.p .= 0

    name = "3DAirshp_$(L)_108500_nds"
    wr = vtkWriter(name; attrib=custom_attrib,dir="E:/Graduation_Assignment/waterlily/ParametricBodies.jl/vtk_data")
    pforce,vforce,drag = [],[],[]

    for tᵢ in range(0,duration;step=tstep)
        sim_step!(sim,tᵢ,remeasure=false)
        cf,cp = 2*viscous_force(sim)[1]/A, 2*pressure_force(sim)[1]/A
        push!(pforce,cp)
        push!(vforce,cf)
        push!(drag,cp+cf)
        println("3D: tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
        write!(wr, sim)
    end
    close(wr)

    @show(drag)
    @show(pforce)
    @show(vforce)
    @show(A)
    @show(V^(2/3))
    plot(range(0,duration;step=tstep),drag,label="Cd",xlabel="tU/L",ylabel="C_force")
    plot!(range(0,duration;step=tstep),vforce,label="Cv")
    plot!(range(0,duration;step=tstep),pforce,label="Cp")
    savefig(name*".png")
end 

run(2^6,8,0.1)

#for L in [2^5,2^6,3*2^5,2^7,3*2^6]
    #run(L)
#end