using WaterLily
using StaticArrays
using WriteVTK
using ParametricBodies

function pressure_force(sim::Simulation,n::Integer=16,T=promote_type(Float64,eltype(sim.flow.p)))
    forces = []; limits = ParametricBodies.lims(sim.body)
    ξ = limits[1]:1/n:limits[2] # get segments
    segmts = [sim.body.curve.(ξᵢ,0) for ξᵢ in zip(ξ[1:end-1],ξ[2:end])]
    for (p₁,p₂) in segmts
        sim.flow.f .= zero(eltype(sim.flow.p))
        WaterLily.@loop sim.flow.f[I,:] .= sim.flow.p[I]*sim.flow.μ₀[I,:].*nds(loc(0,I,T),p₁,p₂) over I ∈ inside(sim.flow.p)
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
function nds(x,p1,p2,R=3)
    d = cylinder(x,p1,p2,R)
    n = ForwardDiff.gradient(x->cylinder(x,p1,p2,R), x)
    m = √sum(abs2,n); d /= m; n /= m
    return n*WaterLily.kern(clamp(d,-1,1))
end
# function make_sim(L;Re=250,U=1,mem=Array,T=Float32)
#     a = SA[L/2,L/2,L/2]
#     b = SA[3L/2,L/2,L/2]
#     body = AutoBody((x,t)->cylinder(x,a,b,L/4))
#     Simulation((2L,L,L),(U,0,0),L;ν=U*L/Re,body,mem,T)
# end

function make_sim(L=2^5;Re=250,U =1,ϵ=0.5,thk=6,mem=Array)
    # define a flat plat at and angle of attack
    cps = SA[-1   0   1
             0.0 0.0 0.0
             0    0   0]*L .+ [2L,3L,8]

    # needed if control points are moved
    curve = BSplineCurve(cps;degree=2)

    # use BDIM-σ distance function, make a body and a Simulation
    body = DynamicNurbsBody(curve;thk=thk,boundary=false)
    Simulation((8L,6L,16),(U,0,0),L;U,ν=U*L/Re,body,T=Float64,mem)
end


# make the sim
sim = make_sim(32)

# make a writer with some attributes
velocity(a::Simulation) = a.flow.u |> Array;
pressure(a::Simulation) = a.flow.p |> Array;
_body(a::Simulation) = (measure_sdf!(a.flow.σ, a.body, WaterLily.time(a.flow)); 
                        a.flow.σ |> Array;)

# measures only on the first 1/2 of the curve
p₁ = sim.body.curve(0.f0,0.f0)
p₂ = sim.body.curve(0.5f0,0.f0)

_nds(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds(loc(0,I),p₁,p₂,2) over I ∈ inside(a.flow.p); 
                        a.flow.f |> Array;)

custom_attrib = Dict(
    "u" => velocity, "p" => pressure, "d" => _body, "nds" => _nds
)# this

# make a vtk writer
wr = vtkWriter("ThreeD_cylinder_origin"; attrib=custom_attrib)

# intialize
t₀ = 0.0; duration = 4; tstep = 0.1
pforce,vforce = [],[]
write!(wr,sim)
close(wr)

WaterLily.apply!(x->x[2], sim.flow.p)
p = pressure_force(sim,4)
@show sum(getindex.(p,2))
@show pressure_force(sim,4) # new force function
@show(2*32*(6/2)^2*π)

# # step and write for a longer time
# @time for tᵢ in range(t₀,t₀+duration;step=tstep)
#     sim_step!(sim,tᵢ,remeasure=false)
#     cp = 2*pressure_force(sim)[1]/sim.L^2
#     cf = 2*viscous_force(sim)[1]/sim.L^2
#     push!(pforce,cp)
#     push!(vforce,cf)
#     write!(wr, sim)
#     println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
# end
# close(wr)