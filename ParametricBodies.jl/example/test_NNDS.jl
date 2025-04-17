using WaterLily
using StaticArrays
using WriteVTK
using ParametricBodies

using WaterLily: @loop,inside,inside_u,nds,∇²u
function pressure_force(sim::Simulation;T=promote_type(Float64,eltype(sim.flow.p)))
    sim.flow.f .= zero(eltype(sim.flow.p))
    WaterLily.@loop sim.flow.f[I,:] .= sim.flow.p[I]*WaterLily.nds(sim.body,loc(0,I,T),0.0) over I ∈ inside(sim.flow.p)
    sum(T,sim.flow.f,dims=ntuple(i->i,ndims(sim.flow.p)))[:] |> Array
end

function NNDS(R;t=0,L=64,U=1,Re=250)
    cps = SA[1/2 1
             1/2 1/2
             1/4 1/4]*L
    curve = BSplineCurve(cps,degree=1)
    body = DynamicNurbsBody(curve;thk=(u)->R,boundary=false)
    Simulation((4L,2L,32),(1,0,0),L;U,ν=U*L/Re,body,T=Float64,mem=Array)
end

sim = NNDS(6)
apply!(x->x[2],sim.flow.p)
@show(pressure_force(sim)[2])
@show(sim.L*9π/2+4/3*π*3^3)


# d,n,dotS = ParametricBodies.measure(body,x,t)
# return n*WaterLily.kern(clamp(d,-1,1))#.*WaterLily.μ₀(sdf(body,x,0)+0.5,0)