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
function nds_obtain(sim,t=WaterLily.time(sim.flow),T=promote_type(Float64,eltype(sim.flow.u)))
    sim.flow.f .= zero(eltype(sim.flow.u))
    @loop sim.flow.f[I,:] .= 1*nds(sim.body,loc(0,I,T),t) over I ∈ inside_u(sim.flow.u)
    vcat([sim.flow.f]...)
end

# parameters
function dynamicSpline(;L=2^4,Re=250,U =1,ϵ=0.5,mem=Array)
    # define a flat plat at and angle of attack
    cps = SA[-1   0   1
             0.5 0.25 0]*L .+ [2L,3L]

    # needed if control points are moved
    weights,knots = SA[1.,1.,1.],SA[0,0,0,1,1,1.]

    # a non-boundary curve of thinckness √2/2+1
    body = DynamicNurbsBody(NurbsCurve(cps,knots,weights);thk=6,boundary=false)

    # make sim
    Simulation((8L,6L),(U,0),L;U,ν=U*L/Re,body,T=Float64,mem)
end

function pressure_distribution(sim;N=64)
    lims = ParametricBodies.lims(sim.body)
    # integrate NURBS curve to compute integral
    uv_, w_ = ParametricBodies._gausslegendre(N,typeof(first(lims)))
    # map onto the (uv) interval, need a weight scalling
    scale=(last(lims)-first(lims))/2; uv_=scale*(uv_.+1); w_=scale*w_
    @show(uv_)
    @show(lims)
    [-ParametricBodies._pforce(sim.body.curve,sim.flow.p,uv,0.0,ParametricBodies.open(sim.body);δ=6) for uv in uv_]
end

# intialize
sim = dynamicSpline()#mem=CuArray);
t₀,duration,tstep = sim_time(sim),0.1,0.1;

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

name = "2Dnurbs"
wr = vtkWriter(name; attrib=custom_attrib)
nds2 = []
f = []

# run
for tᵢ in range(t₀,t₀+duration;step=tstep)

    # update until time tᵢ in the background
    t = sum(sim.flow.Δt[1:end-1])
    while t < tᵢ*sim.L/sim.U
        # random update
        new_pnts = (SA[-1     0   1
                        0.5 0.25+0.5*sin(π/4*t/sim.L) 0] .+ [2,3])*sim.L
        sim.body = update!(sim.body,new_pnts,sim.flow.Δt[end])
        measure!(sim,t)
        mom_step!(sim.flow,sim.pois) # evolve Flow
        t += sim.flow.Δt[end]
    end

    global f = pressure_distribution(sim;N=16)
    @show f#[1][2]

    global nds2 = nds_obtain(sim)
    @show(nds2[:,:,1][1,1])

    # print time step
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
    write!(wr, sim)

end
close(wr)