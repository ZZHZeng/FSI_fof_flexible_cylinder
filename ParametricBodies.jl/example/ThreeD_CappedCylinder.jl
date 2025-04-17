using WaterLily
using StaticArrays
using WriteVTK
using ParametricBodies

function pressure_force(sim::Simulation;T=promote_type(Float64,eltype(sim.flow.p)))
    sim.flow.f .= zero(eltype(sim.flow.p))
    WaterLily.@loop sim.flow.f[I,:] .= sim.flow.p[I]*WaterLily.nds(sim.body,loc(0,I,T),0.0) over I ∈ inside(sim.flow.p)
    sum(T,sim.flow.f,dims=ntuple(i->i,ndims(sim.flow.p)))[:] |> Array
end

function pressure_force(sim::Simulation,n,R;T=promote_type(Float64,eltype(sim.flow.p)))
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
    RR = R-1   
    ba = p₂ - p₁
    pa = x - p₁
    baba = dot(ba,ba)
    paba = dot(pa,ba)
    x = norm(pa*baba-ba*paba) - RR*baba
    y = abs(paba-baba*0.5f0)-baba*0.5f0
    x2 = x*x
    y2 = y*y*baba
    d = (max(x,y)<0) ? -min(x2,y2) : (((x>0) ? x2 : 0)+((y>0) ? y2 : 0))
    return convert(T,sign(d)*sqrt(abs(d))/baba-1)
end
using ForwardDiff
function nds(body,x,p1,p2,R)
    d = cylinder(x,p1,p2,R)
    n = ForwardDiff.gradient(x->cylinder(x,p1,p2,R), x)
    m = √sum(abs2,n); d /= m; n /= m
    any(isnan.(n)) && return zero(x) ## remove the singular points. think about the rectangular, the normal vectors around corners are nasty 
    return n*WaterLily.kern(clamp(d,-1,1)).*WaterLily.μ₀(WaterLily.sdf(body,x,0)+0.5,0)
end

function make_sim(L,thk,h;Re=250,U =1,ϵ=0.5,mem=Array)
    # define a flat plat at and angle of attack
    cps = SA[0   1   2
             0.0 h/2 h
             0    0   0]*L .+ [L,2L,16]

    # needed if control points are moved
    curve = BSplineCurve(cps;degree=2)

    # use BDIM-σ distance function, make a body and a Simulation
    body = DynamicNurbsBody(curve;thk=(u)->thk,boundary=false)
    Simulation((4L,4L,32),(U,0,0),L;U,ν=U*L/Re,body,T=Float64,mem)
end

function run(L,h,thk,N)

    sim = make_sim(L,thk,h)

    # make a writer with some attributes
    velocity(a::Simulation) = a.flow.u |> Array;
    pressure(a::Simulation) = a.flow.p |> Array;
    _body(a::Simulation) = (measure_sdf!(a.flow.σ, a.body, WaterLily.time(a.flow)); 
                            a.flow.σ |> Array;)

    n = N
    limits = ParametricBodies.lims(sim.body)
    ξ = limits[1]:1/n:limits[2] # get segments
    segmts = [sim.body.curve.(ξᵢ,0) for ξᵢ in zip(ξ[1:end-1],ξ[2:end])]
    p₁,p₂ = segmts[1]

    _sdf(a::Simulation) = (WaterLily.@loop a.flow.σ[I] = cylinder(loc(0,I),p₁,p₂,thk/2) over I ∈ inside(a.flow.p); 
                            a.flow.σ |> Array;)
    _nds(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds(a.body,loc(0,I),p₁,p₂,thk/2) over I ∈ inside(a.flow.p); 
                            a.flow.f |> Array;)
    _nds2(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= WaterLily.nds(a.body,loc(0,I),0.0) over I ∈ inside(a.flow.p); 
                            a.flow.f |> Array;)

    custom_attrib = Dict(
        "u" => velocity, "p" => pressure, "d" => _body, "nds" => _nds, "sdf_2" => _sdf, "nds2" => _nds2
    )

    wr = vtkWriter("ThreeD_cylinder"; attrib=custom_attrib)

    write!(wr,sim)
    close(wr)

    apply!(x->x[2],sim.flow.p)
    @time global p = pressure_force(sim,N,thk/2)
    @show p
    @show sum(getindex.(p,2)) # new force function
    @show pressure_force(sim)[2] # old force function
    @show((thk/2)^2*π*(2^2+h^2)^0.5*64)

end
run(64,0.2,12,32)