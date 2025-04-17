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

# parameters
function dynamicSpline(L,thk;Re=250,U=1,mem=Array)
    # define a flat plat at and angle of attack
    cps = SA[-1   0   1
             2.0 0.0 -2.0]*L .+ [2L,2L]
    # needed if control points are moved
    weights,knots = SA[1.,1.,1.],SA[0,0,0,1,1,1.]
    # a non-boundary curve of thinckness √2/2+1
    body = DynamicNurbsBody(NurbsCurve(cps,knots,weights);thk=thk,boundary=false)
    # make sim
    Simulation((8L,6L),(U,0),L;U,ν=U*L/Re,body,T=Float64,mem)
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
    "NDS" => NDS, ## may be wrong
    "Body" => body,
    "normal" => _normal
)# this maps what to write to the name in the file

pforce,vforce,drag = [],[],[]
vol = []
vol_analy = [] 
vol1 = 0

function run(L,thk)
    sim = dynamicSpline(L,thk)#mem=CuArray);
    t₀,duration,tstep = sim_time(sim),0.2,0.1;
    name = "2Dnurbs_test_thk$(thk)_large_slope"
    wr = vtkWriter(name; attrib=custom_attrib)

    for tᵢ in range(t₀,t₀+duration;step=tstep)
        # update until time tᵢ in the background
        t = sum(sim.flow.Δt[1:end-1])
        while t < tᵢ*sim.L/sim.U
            # random update
            new_pnts = (SA[-1     0   1
                            2+0*sin(π/4*t/sim.L) 0.0+0*sin(π/4*t/sim.L) -2-0*sin(π/4*t/sim.L)] .+ [2,2])*sim.L
            sim.body = update!(sim.body,new_pnts,sim.flow.Δt[end])
            measure!(sim,t)
            WaterLily.apply!(x->x[1], sim.flow.p) 
            global vol1 = pressure_force(sim)[1] # volume measured by pressure_force
            sim.flow.p .= 0 
            mom_step!(sim.flow,sim.pois) # evolve Flow
            t += sim.flow.Δt[end]
        end
        push!(vol,vol1)

        println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
        
        vol2 = thk*sim.L*(2^2+(0.5)^2)^0.5 + π*(thk/2)^2 # analytical result
        push!(vol_analy,vol2)
        write!(wr, sim)
    end
    close(wr)
    plot(range(0,duration;step=tstep),vol,label="area_simu",xlabel="tU/L",ylabel="curve_area_L$(L)_thk$(thk)")
    plot!(range(0,duration;step=tstep),vol_analy,label="area_analy")
    savefig(name*".png")
    plot(range(0,duration;step=tstep),vol,label="area_simu",xlabel="tU/L",ylabel="curve_area_L$(L)_thk$(thk)")
    plot!(range(0,duration;step=tstep),vol_analy,label="area_analy")
end
run(64,4) # L,thk; dont do L=96
