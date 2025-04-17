using WaterLily
using StaticArrays
using WriteVTK
using ParametricBodies

function pressure_force(sim::Simulation;T=promote_type(Float64,eltype(sim.flow.p)))
    sim.flow.f .= zero(eltype(sim.flow.p))
    WaterLily.@loop sim.flow.f[I,:] .= sim.flow.p[I]*WaterLily.nds(sim.body,loc(0,I,T),0.0) over I ∈ inside(sim.flow.p)
    sum(T,sim.flow.f,dims=ntuple(i->i,ndims(sim.flow.p)))[:] |> Array
end

function pressure_force1(sim::Simulation,n,R;T=promote_type(Float64,eltype(sim.flow.p)))
    forces = []; limits = ParametricBodies.lims(sim.body)
    ξ = limits[1]:1/n:limits[2] # get segments
    segmts = [sim.body.curve.(ξᵢ,0) for ξᵢ in zip(ξ[1:end-1],ξ[2:end])]
    for (p₁,p₂) in segmts
        sim.flow.f .= zero(eltype(sim.flow.p))
        WaterLily.@loop sim.flow.f[I,:] .= sim.flow.p[I].*nds(sim.body,loc(0,I,T),p₁,p₂,R) over I ∈ inside(sim.flow.p)
        push!(forces,sum(T,sim.flow.f,dims=ntuple(i->i,ndims(sim.flow.p)))[:])
    end
    return forces
end

using LinearAlgebra: dot,norm
function cylinder(x::SVector{N,T},p₁,p₂,R) where {N,T}
    ba = p₂ - p₁
    pa = x - p₁
    baba = dot(ba,ba)
    paba = dot(pa,ba)
    x = norm(pa*baba-ba*paba) - R*baba
    y = abs(paba-baba*0.5f0)-baba*0.5f0
    x2 = x*x
    y2 = y*y*baba
    d = (max(x,y)<0) ? -min(x2,y2) : (((x>0) ? x2 : 0)+((y>0) ? y2 : 0))
    return convert(T,sign(d)*sqrt(abs(d))/baba)
end
using ForwardDiff
function nds(body,x,p1,p2,R)
    d = cylinder(x,p1,p2,R)
    n = ForwardDiff.gradient(x->cylinder(x,p1,p2,R), x)
    m = √sum(abs2,n); d /= m; n /= m
    any(isnan.(n)) && return zero(x)##
    dsdf = WaterLily.sdf(body,x,0)+0.5
    return n*WaterLily.kern(clamp(d,-1,1)).*WaterLily.μ₀(clamp(dsdf,-1,1),0)
end

function make_sim(L;Re=250,U =1,ϵ=0.5,thk=6,mem=Array)
    # define a flat plat at and angle of attack
    cps = SA[0   1    2
             0.0 0.2  0.4
             0    0   0]*L .+ [2L,L,L]

    # needed if control points are moved
    curve = BSplineCurve(cps;degree=2)

    # use BDIM-σ distance function, make a body and a Simulation
    body = DynamicNurbsBody(curve;thk=(u)->thk,boundary=false)
    Simulation((6L,2L,2L),(U,0,0),L;U,ν=U*L/Re,body,T=Float64,mem)
end


# make the sim
sim = make_sim(16)

# make a writer with some attributes
velocity(a::Simulation) = a.flow.u |> Array;
pressure(a::Simulation) = a.flow.p |> Array;
_body(a::Simulation) = (measure_sdf!(a.flow.σ, a.body, WaterLily.time(a.flow)); 
                        a.flow.σ |> Array;)

n = 4
limits = ParametricBodies.lims(sim.body)
ξ = limits[1]:1/n:limits[2] # get segments
segmts = [sim.body.curve.(ξᵢ,0) for ξᵢ in zip(ξ[1:end-1],ξ[2:end])]
p₁,p₂ = segmts[1]

_sdf(a::Simulation) = (WaterLily.@loop a.flow.σ[I] = cylinder(loc(0,I),p₁,p₂,4) over I ∈ inside(a.flow.p); 
                        a.flow.σ |> Array;)
_nds(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds(a.body,loc(0,I),p₁,p₂,4) over I ∈ inside(a.flow.p); 
                        a.flow.f |> Array;)
_nds2(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= WaterLily.nds(a.body,loc(0,I),0.0) over I ∈ inside(a.flow.p); 
                        a.flow.f |> Array;)

custom_attrib = Dict(
    "u" => velocity, "p" => pressure, "d" => _body, "nds" => _nds, "sdf_2" => _sdf, "nds2" => _nds2
)# this

# make a vtk writer
wr = vtkWriter("ThreeD_cylinder_no_tapper"; attrib=custom_attrib)

# intialize
t₀ = 0.0; duration = 0.1; tstep = 0.1
pforce,vforce = [],[]

# step and write for a longer time
for tᵢ in range(t₀,t₀+duration;step=tstep)
    sim_step!(sim,tᵢ,remeasure=false)
    write!(wr, sim)
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
end
close(wr)

apply!(x->x[2],sim.flow.p)
@time p = pressure_force1(sim,16,3)

forces = []; limits = ParametricBodies.lims(sim.body)
ξ = limits[1]:1/n:limits[2] # get segments
segmts = [sim.body.curve.(ξᵢ,0) for ξᵢ in zip(ξ[1:end-1],ξ[2:end])]
T = Float64
R=4
p₁,p₂ =  segmts[1]
sim.flow.f .= zero(eltype(sim.flow.p));
WaterLily.@loop sim.flow.f[I,:] .= sim.flow.p[I].*nds(sim.body,loc(0,I,T),p₁,p₂,R) over I ∈ inside(sim.flow.p)
sum(T,sim.flow.f,dims=ntuple(i->i,ndims(sim.flow.p)))[:]

@show p
@show(sum(getindex.(p,2))+4/3*π*(6/2)^3) # new force function
@show pressure_force(sim) # old force function
@show((2^2+0.4^2)^0.5*16*(6/2)^2*π+4/3*π*(6/2)^3)