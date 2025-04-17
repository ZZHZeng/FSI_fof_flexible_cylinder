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

# parameters
function dynamicSpline(;L=2^5,Re=250,U=1,mem=Array)
    # define a flat plat at and angle of attack
    cps = SA[-1   0   1
             0.5 0.25 0]*L .+ [2L,3L]

    # needed if control points are moved
    weights,knots = SA[1.,1.,1.],SA[0,0,0,1,1,1.]

    # a non-boundary curve of thinckness √2/2+1
    body = DynamicNurbsBody(NurbsCurve(cps,knots,weights);thk=6,boundary=false)

    # cps = SA_F32[1 1 0 -1 -1 -1  0  1 1
    # 0 1 1  1  0 -1 -1 -1 0]*L/2 .+ [2L,3L]
    # weights = SA_F32[1.,√2/2,1.,√2/2,1.,√2/2,1.,√2/2,1.]
    # knots =   SA_F32[0,0,0,1/4,1/4,1/2,1/2,3/4,3/4,1,1,1]

    # # make a nurbs curve and a body for the simulation
    # body = DynamicNurbsBody(NurbsCurve(cps,knots,weights))

    # make sim
    Simulation((8L,6L),(U,0),L;U,ν=U*L/Re,body,T=Float64,mem)
end

# function pressure_distribution(sim;N=64)
#     lims = ParametricBodies.lims(sim.body)
#     # integrate NURBS curve to compute integral
#     uv_, w_ = ParametricBodies._gausslegendre(N,typeof(first(lims)))
#     # map onto the (uv) interval, need a weight scalling
#     scale=(last(lims)-first(lims))/2; uv_=scale*(uv_.+1); w_=scale*w_

#     hcat([-ParametricBodies._pforce(sim.body.curve,sim.flow.p,uv,0.0,ParametricBodies.open(sim.body);δ=6) for uv in uv_]...)
# end

# intialize
sim = dynamicSpline()#mem=CuArray);
t₀,duration,tstep = sim_time(sim),1,0.1;

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

pforce,vforce,drag = [],[],[]
vol = []
vol_analy = [] 
vol1 = 0

name = "2Dnurbs"
wr = vtkWriter(name; attrib=custom_attrib)

# run
@gif for tᵢ in range(t₀,t₀+duration;step=tstep)

    # update until time tᵢ in the background
    t = sum(sim.flow.Δt[1:end-1])
    while t < tᵢ*sim.L/sim.U
        # random update
        new_pnts = (SA[-1     0   1
                        0.4+0*sin(π/4*t/sim.L) 0.2+0*sin(π/4*t/sim.L) 0-0*sin(π/4*t/sim.L)] .+ [2,3])*sim.L
        sim.body = update!(sim.body,new_pnts,sim.flow.Δt[end])
        measure!(sim,t)
        WaterLily.apply!(x->x[1], sim.flow.p) 
        global vol1 = pressure_force(sim)[1] # volume measured by pressure_force
        sim.flow.p .= 0 
        mom_step!(sim.flow,sim.pois) # evolve Flow
        t += sim.flow.Δt[end]
    end
    push!(vol,vol1)

    # f = pressure_distribution(sim;N=8)
    # @show f

    # flood plot
    @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u) * sim.L / sim.U
    contourf(clamp.(sim.flow.σ,-10,10)',dpi=300,
             color=palette(:RdBu_11), clims=(-10,10), linewidth=0,
             aspect_ratio=:equal, legend=false, border=:none)
    plot!(sim.body.curve;shift=(0.5,0.5),add_cp=true)

    # print time step
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
    write!(wr, sim)
    thk=6
    vol2 = thk*sim.L*(2^2+(0.4)^2)^0.5 + π*(thk/2)^2
    push!(vol_analy,vol2)
end
close(wr)
plot(range(0,duration;step=tstep),vol,label="area_simu",xlabel="tU/L",ylabel="curve_area")
plot!(range(0,duration;step=tstep),vol_analy,label="area_analy")
