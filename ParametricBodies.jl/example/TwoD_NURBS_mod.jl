###
using Pkg
Pkg.develop(path="E:/Graduation_Assignment/waterlily/WaterLily.jl")
using WaterLily
println(pathof(WaterLily))
###
using ParametricBodies
using StaticArrays
using Plots
using WriteVTK
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

# parameters
function dynamicSpline(L,Re,m,n,thk;U=1,mem=Array)
    # define a flat plat at and angle of attack
    cps = SA[-1   0   1
              0.2 0.1 0.0]*L .+ [2,n÷2]*L

    # needed if control points are moved
    weights,knots = SA[1.,1.,1.],SA[0,0,0,1,1,1.]

    # a non-boundary curve of thinckness thk
    body = DynamicNurbsBody(NurbsCurve(cps,knots,weights);thk=thk,boundary=false,exitBC=true)

    # make sim
    Simulation((m*L,n*L),(U,0),L;U,ν=U*L/Re,body,T=Float64,mem)
end

velocity(a::Simulation) = a.flow.u |> Array;
pressure(a::Simulation) = a.flow.p |> Array;
NDS(a::Simulation) = a.flow.f ./ a.flow.p 
body(a::Simulation) = (measure_sdf!(a.flow.σ, a.body); 
                       a.flow.σ |> Array;)
custom_attrib = Dict(
    "Velocity" => velocity,
    "Pressure" => pressure,
    "NDS" => NDS, ## may be wrong
    "Body" => body
)# this maps what to write to the name in the file

pforce,vforce,drag = [],[],[]
vol = []
vol_analy = [] 
vol1 = 0

function run(L,Re,m,n,thk,t)

    sim = dynamicSpline(L,Re,m,n,thk)#mem=CuArray);
    t₀,tstep = sim_time(sim),0.1
    duration = t
    t1 = round(Int,10t)
    name = "2Dnurbs_mod_L$(L)_dur0$(t1)_m$(m)_n$(n)_thk$(thk)"
    wr = vtkWriter(name; attrib=custom_attrib)

    for tᵢ in range(t₀,t₀+duration;step=tstep)

       # update until time tᵢ in the background
       t = sum(sim.flow.Δt[1:end-1])
       while t < tᵢ*sim.L/sim.U
           # random update
           new_pnts = (SA[-1     0   1
                           0.2+0*sin(π/4*t/sim.L)   0.1   0.0-0*sin(π/4*t/sim.L)] .+ [2,n÷2])*sim.L
           sim.body = update!(sim.body,new_pnts,sim.flow.Δt[end])
           measure!(sim,t)
           WaterLily.apply!(x->x[1], sim.flow.p) 
           global vol1 = pressure_force(sim)[1] # volume measured by pressure_force
           sim.flow.p .= 0 
           mom_step!(sim.flow,sim.pois) # evolve Flow 
        t += sim.flow.Δt[end]
       end
       push!(vol,vol1)

       # print time step
       println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))

       cv,cp = 2viscous_force(sim,thk)[1]/L, 2pressure_force(sim,thk)[1]/L
       push!(pforce,-cp)
       push!(vforce,-cv)
       push!(drag,-cv-cp)

       # analytical result of volume
       vol2 = thk*2*L + π*(thk/2)^2
       push!(vol_analy,vol2)

       write!(wr, sim)
   end
   close(wr)
   
#    plot(range(0,duration;step=tstep),drag,label="Cd",xlabel="tU/L",ylabel="C_force")
#    plot!(range(0,duration;step=tstep),vforce,label="Cv")
#    plot!(range(0,duration;step=tstep),pforce,label="Cp")
#    savefig(name*"_water_force"*".png")
   plot(range(0,duration;step=tstep),vol,label="vol",xlabel="tU/L",ylabel="vol")
   plot!(range(0,duration;step=tstep),vol_analy,label="vol_analy")
#    savefig(name*"_volume"*".png")

#    CSV.write(name*".csv", Tables.table(hcat(pforce,vforce,drag,vol_analy)), writeheader=false)
end
run(2^6,500,6,4,6,0.5) #L,Re,domain length,domain width,thickness,duration