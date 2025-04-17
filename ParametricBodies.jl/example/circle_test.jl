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
using CSV,Tables,DataFrames

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

function circle_hash(L,ar,Re,n,m;U=1,T=Float32,mem=Array)
    # Map from simulation coordinate x to surface coordinate ξ
    b = T(ar)
    ellipse(θ,t) = 0.5f0L*SA[cos(θ),b*sin(θ)].+[2L,m÷2*L] # define parametric curve
    body = HashedBody(ellipse,(0,2π);T)  # automatically finds closest point
    Simulation((n*L,m*L),(U,0),L;ν=U*L/Re,body,T,mem,exitBC=true)
end

function circle_nurbs(L,ar,Re,n,m;U=1,mem=Array,T=Float32)

    cps = SA_F32[1 1 0 -1 -1 -1  0  1 1
                0 1 1  1  0 -1 -1 -1 0]*L/2 .+ [2L,m÷2*L]
    weights = SA_F32[1.,√2/2,1.,√2/2,1.,√2/2,1.,√2/2,1.]
    knots =   SA_F32[0,0,0,1/4,1/4,1/2,1/2,3/4,3/4,1,1,1]

    Body = ParametricBody(NurbsCurve(cps,knots,weights))
    Simulation((n*L,m*L),(U,0),L;U,ν=U*L/Re,body=Body,T=T,mem=mem,exitBC=true)
end

function circle_sdf(L,ar,Re,n,m;U=1,mem=Array,T=Float32)
    center,radius = [2,m÷2]*L,0.5f0L
    Body = AutoBody((x,t)->√sum(abs2, x .- center) - radius)
    Simulation((n*L,m*L),(U,0),L;U,ν=U*L/Re,body=Body,T=T,mem=mem,exitBC=true) 
end

velocity(a::Simulation) = a.flow.u |> Array;
pressure(a::Simulation) = a.flow.p |> Array;
NDS(a::Simulation) = a.flow.f ./ a.flow.p #(@loop sim.flow.f[I,:] .= measure(sim.body,loc(0,I),0)[2] over I ∈ inside(sim.flow.p));
                     #sim.flow.f |> Array;
body(a::Simulation) = (measure_sdf!(a.flow.σ, a.body); 
                       a.flow.σ |> Array;)
custom_attrib = Dict(
    "Velocity" => velocity,
    "Pressure" => pressure,
    "NDS" => NDS,
    "Body" => body
)# this maps what to write to the name in the file


pforce,vforce,drag = [],[],[]
pforce_para,vforce_para,drag_para = [],[],[]
ratio = []
ratio_para = []

function run(L,ar,Re,t,n,m,func)

    if func == "hash"
        sim = circle_hash(L,ar,Re,n,m)
    elseif func == "sdf"
        sim = circle_sdf(L,ar,Re,n,m)
    else sim = circle_nurbs(L,ar,Re,n,m)
    end 

    duration,tstep = t, 0.1
    arr = round(Int,10ar)

    name = "2Dcircle_test_$(L)_$(func)_Re$(Re)_dur$(t)_ar0$(arr)_$(n)_$(m)_20241009"
    wr = vtkWriter(name; attrib=custom_attrib)#,dir="E:/Graduation_Assignment/waterlily/ParametricBodies.jl/vtk_data")

    for tᵢ in range(0,duration;step=tstep)
        sim_step!(sim,tᵢ,remeasure=false)
        cv,cp = 2viscous_force(sim)[1]/(ar*L), 2pressure_force(sim)[1]/(ar*L)
        # cv_para,cp_para = 2ParametricBodies.viscous_force(sim)[1]/L, 2ParametricBodies.pressure_force(sim)[1]/L

        push!(pforce,cp)
        push!(vforce,cv)
        push!(drag,cv+cp)

        # push!(pforce_para,cp_para)
        # push!(vforce_para,cv_para)
        # push!(drag_para,cp_para+cv_para)

        # push!(ratio,cv/cp)
        # push!(ratio_para,cv_para/cp_para)

        println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
        write!(wr, sim)
    end
    close(wr)

    plot(range(0,duration;step=tstep),-drag,label="Cd",xlabel="tU/L",ylabel="C_force")
    plot!(range(0,duration;step=tstep),-vforce,label="Cv")
    plot!(range(0,duration;step=tstep),-pforce,label="Cp")
    savefig(name*"_water_force"*".png")

    # plot(range(0,duration;step=tstep),drag_para,label="Cd_para",xlabel="tU/L",ylabel="C_force")
    # plot!(range(0,duration;step=tstep),vforce_para,label="Cv_para")
    # plot!(range(0,duration;step=tstep),pforce_para,label="Cp_para")
    # savefig(name*"_para_force"*".png")

    # plot(range(0,duration;step=tstep),ratio,label="water_force",xlabel="tU/L",ylabel="Cv/Cp")
    # plot!(range(0,duration;step=tstep),ratio_para,label="para_force")
    # savefig(name*"_ratio_vp"*".png")

    # @show(-pforce)
    # @show(-vforce)
    # @show(-drag)
    # @show(pforce_para)
    # @show(vforce_para)
    # @show(drag_para)
    # @show(ratio)
    # @show(ratio_para)
    CSV.write(name*".csv", Tables.table(hcat(-pforce,-vforce,-drag)), writeheader=false)

end

run(2^9,0.5,100000,24,10,8,"hash") # L, aspect ratio, Re, duration, length of domain, width, function; only circle_hash can change ar.

