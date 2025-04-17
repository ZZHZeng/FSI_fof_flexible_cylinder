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
function obtain_nds(sim,t=WaterLily.time(sim.flow),T=promote_type(Float64,eltype(sim.flow.p)))
    sim.flow.f .= zero(eltype(sim.flow.p))
    @loop sim.flow.f[I,:] .= 1*nds(sim.body,loc(0,I,T),t)[1] over I ∈ inside(sim.flow.p)
    sim.flow.f |> Array
end

function beam_segment(L,Re,m,n;U=1,T=Float32,mem=Array)
    cps = SA[1.0 -1.0 -1.0 1.0  1.0
             1.0  1.0 -1.0 -1.0 1.0]*L .+ [3,n÷2]*L
    curve = BSplineCurve(cps; degree=1)
    body = ParametricBody(curve;exitBC=true)
    Simulation((m*L,n*L),(U,0),L;U,ν=U*L/Re,body,T=Float64,mem)
end

velocity(a::Simulation) = a.flow.u |> Array;
pressure(a::Simulation) = a.flow.p |> Array;
# NDS(a::Simulation) = a.flow.f ./ a.flow.p 
body(a::Simulation) = (measure_sdf!(a.flow.σ, a.body); 
                       a.flow.σ |> Array;)
custom_attrib = Dict(
    "Velocity" => velocity,
    "Pressure" => pressure,
    # "NDS" => NDS, ## may be wrong
    "Body" => body
)# this maps what to write to the name in the file

pforce,vforce,drag = [],[],[]
vol = []
vol_analy = []
nds1 = []

function run(L,Re,m,n,t)
    sim = beam_segment(L,Re,m,n)#mem=CuArray);
    t₀,tstep = sim_time(sim),0.1
    duration = t
    name = "beam_segment_Re$(Re)_L$(L)_dur$(t)_m$(m)_n$(n)_"
    wr = vtkWriter(name; attrib=custom_attrib)

    for tᵢ in range(t₀,t₀+duration;step=tstep)

       # update until time tᵢ in the background
       t = sum(sim.flow.Δt[1:end-1])
       while t < tᵢ*sim.L/sim.U
           # random update
           new_pnts = SA[1.0              -1.0 -1.0  1.0              1.0
                         1.0+sin(π/4*t/L)  1.0 -1.0 -1.0-sin(π/4*t/L) 1.0+sin(π/4*t/L)]*L .+ [3,n÷2]*L
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

       # analytical result of volume
       vol2 = L^2*(4+2sin(π/4/L*tᵢ*sim.L/sim.U))
       push!(vol_analy,vol2)
       push!(nds1,obtain_nds(sim))

       write!(wr, sim)
   end
   close(wr)
   
   plot(range(0,duration;step=tstep),vol,label="vol",xlabel="tU/L",ylabel="vol")
   plot!(range(0,duration;step=tstep),vol_analy,label="vol_analy")
   savefig(name*"_volume"*".png")
#    CSV.write(name*".csv", Tables.table(nds1), writeheader=false)
end
run(2^4,250,8,6,4) # L,Re,domain length,domain width,duration
