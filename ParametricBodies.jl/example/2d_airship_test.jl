using WaterLily
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

function circle(L,ar;Re=500,U=1,mem=Array,T=Float32)
    theta = atan(ar) # ar=aspect ratio
    k1 = theta/π
    k2 = (π-theta)/π
    # NURBS points, weights and knot vector for a circle
    cps = SA_F32[1 1  0  -1 -1 -1  0   1   1
                 0 1 1 1 0  -1 -1 -1 0]*L/2 .+ [2L,3L]
    weights = SA_F32[1.,√2/2,1.,√2/2,1.,√2/2,1.,√2/2,1.]
    knots =   SA_F32[0,0,0,1/4,1/4,1/2,1/2,3/4,3/4,1,1,1]
    curve = NurbsCurve(cps,knots,weights)
    # make a nurbs curve and a body for the simulation
    Body = ParametricBody(curve)
    Simulation((8L,6L),(U,0),L;U,ν=U*L/Re,body=Body,T=T,mem=mem,exitBC=true)
end

function ellipse(L,ar;Re=100,U=1,T=Float32,mem=Array)
    # Map from simulation coordinate x to surface coordinate ξ
    b = T(ar)
    ellipse(θ,t) = 0.5f0L*SA[1*cos(θ),b*sin(θ)] .+[2L,3L] # define parametric curve
    body = HashedBody(ellipse,(0,2π))
    # make a sim
    Simulation((8L,6L),(U,0),L;ν=U*L/Re,body,T,mem,exitBC=true)
end

function airship(L,ar,Re;U=1,T=Float32,mem=Array)
    length = L
    D = 3L÷16 # safe over estimate
    cps = SA_F32[1.0  0.819232  0.39  0.15   0.0    0.0  0.0    0.15      0.39   0.819232  1.0
                 0.0ar  0.054ar 0.088ar  0.07ar   0.03ar   0.0ar  -0.03ar -0.07ar    -0.088ar  -0.054ar    0.0ar]*L .+[L÷2,8D]
    curve = BSplineCurve(cps; degree=4)
    Body = ParametricBody(curve)
    Simulation((5L÷2,16D),(U,0),L; ν=U*length/Re,body=Body,T=T,mem=mem,exitBC=true)
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

function run(L,ar,Re)

    sim = airship(L,ar,Re)
    duration,tstep = 4, 0.1
    ar1 = round(Int,10ar)

    name = "2Dairship_test_$(L)_$(ar1)_Re$(Re)"
    wr = vtkWriter(name; attrib=custom_attrib)#,dir="E:/Graduation_Assignment/waterlily/ParametricBodies.jl/vtk_data")

    for tᵢ in range(0,duration;step=tstep)

        sim_step!(sim,tᵢ,remeasure=false)
        cf,cp = 2viscous_force(sim)[1]/L, 2pressure_force(sim)[1]/L
        #cf_para,cp_para = 2ParametricBodies.viscous_force(sim)[1]/L, 2ParametricBodies.pressure_force(sim)[1]/L

        push!(pforce,cp)
        push!(vforce,cf)
        push!(drag,cf+cp)
        #push!(pforce_para,cp_para)
        #push!(vforce_para,cf_para)
        #push!(drag_para,cf_para+cp_para)

        println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
        write!(wr, sim)

    end
    close(wr)

    #plot(range(0,duration;step=tstep),drag,label="Cd",xlabel="tU/L",ylabel="C_force")
    plot(range(0,duration;step=tstep),pforce,label="Cp")
    plot!(range(0,duration;step=tstep),vforce,label="Cv")
    #plot!(range(0,duration;step=tstep),pforce_para,label="Cp_para")
    #plot!(range(0,duration;step=tstep),vforce_para,label="Cv_para")
    #plot!(range(0,duration;step=tstep),drag_para,label="Cd_para")
    savefig(name*".png")
    #@show(drag)
    #@show(drag_para)
    @show sim.pois.n
end

run(2^7,1,100)

duration,tstep = 4, 0.1
#plot(range(0,duration;step=tstep),drag,label="Cd",xlabel="tU/L",ylabel="C_force")
plot(range(0,duration;step=tstep),pforce,label="Cp")
plot!(range(0,duration;step=tstep),vforce,label="Cv")
#plot!(range(0,duration;step=tstep),pforce_para,label="Cp_para")
#plot!(range(0,duration;step=tstep),vforce_para,label="Cv_para")
#plot!(range(0,duration;step=tstep),drag_para,label="Cd_para")

