using WaterLily
using StaticArrays
using WriteVTK
using ParametricBodies

function pressure_force(sim::Simulation;T=promote_type(Float64,eltype(sim.flow.p)))
    sim.flow.f .= zero(eltype(sim.flow.p))
    WaterLily.@loop sim.flow.f[I,:] .= sim.flow.p[I]*WaterLily.nds(sim.body,loc(0,I,T),0.0) over I ∈ inside(sim.flow.p)
    sum(T,sim.flow.f,dims=ntuple(i->i,ndims(sim.flow.p)))[:] |> Array
end

function pressure_force1(sim::Simulation,n,R,L;T=promote_type(Float64,eltype(sim.flow.p)))
    forces = []; limits = ParametricBodies.lims(sim.body)
    ξ = limits[1]:1/n:limits[2] # get segments
    segmts = [sim.body.curve.(ξᵢ,0) for ξᵢ in zip(ξ[1:end-1],ξ[2:end])]
    for (p₁,p₂) in segmts
        sim.flow.f .= zero(eltype(sim.flow.p))
        WaterLily.@loop sim.flow.f[I,:] .= sim.flow.p[I].*nds1(loc(0,I,T),p₁,p₂,R,segmts,n,L) over I ∈ inside(sim.flow.p)
        push!(forces,sum(T,sim.flow.f,dims=ntuple(i->i,ndims(sim.flow.p)))[:])
        # @show(p₁)
        # @show(p₂)
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
function nds1(x,p1,p2,R,segmts,n,L)
    d = cylinder(x,p1,p2,R) 
    p11 = p1 
    p22 = p2 
    aa = x - p11
    bb = p22 - p11
    angn = dot(aa,bb)/(norm(aa)*norm(bb))
    cc = x - p22
    dd = p11 - p22
    ange = dot(cc,dd)/(norm(cc)*norm(dd)) 

    nn = angn*√sum(abs2,aa)
    mm = ange*√sum(abs2,cc)

    gg = (abs(1-angn^2))^0.5*√sum(abs2,aa)
    hh = (abs(1-ange^2))^0.5*√sum(abs2,cc)

    nose=convert(Float64,p1[1])
    beam_nose=convert(Float64,segmts[1][1][1])
    beam_end=convert(Float64,segmts[n][1][1])

    if (p22[1]-p11[1]) == norm(bb)
        std_ang = 0
    else std_ang = -1/(n*R*L) ##?
    end

    # if angn>=0 && ange>0 && nn>1 && mm>1 && nose != beam_nose && nose != beam_end
    #     n = ForwardDiff.gradient(x->cylinder(x,p1,p2,R), x)  
    #     m = √sum(abs2,n); d /= m; n /= m 
    #     any(isnan.(n)) && return zero(x)
    #     return n*WaterLily.kern(clamp(d,-1,1))
    if angn>=std_ang && ange>std_ang && gg>=R-1 && hh>=R-1 && nose != beam_nose && nose != beam_end
        n = ForwardDiff.gradient(x->cylinder(x,p1,p2,R), x)  
        m = √sum(abs2,n); d /= m; n /= m 
        any(isnan.(n)) && return zero(x)
        return n*WaterLily.kern(clamp(d,-1,1))
    elseif nose == beam_nose && (mm>=1 || (gg>=R-1 && hh>=R-1)) && ange>0
        n = ForwardDiff.gradient(x->cylinder(x,p1,p2,R), x)  
        m = √sum(abs2,n); d /= m; n /= m 
        any(isnan.(n)) && return zero(x)
        return n*WaterLily.kern(clamp(d,-1,1))
    elseif nose == beam_end && (nn>=1 || (gg>=R-1 && hh>=R-1)) && angn>=0
        n = ForwardDiff.gradient(x->cylinder(x,p1,p2,R), x)  
        m = √sum(abs2,n); d /= m; n /= m 
        any(isnan.(n)) && return zero(x)
        return n*WaterLily.kern(clamp(d,-1,1))
    else
        return zero(x)
    end
end

function make_sim(L,h,thk;Re=250,U =1,ϵ=0.5,mem=Array)
    # define a flat plat at and angle of attack
    cps = SA[0   1    2
             0.0 h/2  h
             0    0   0]*L .+ [L,2L,16]
    # needed if control points are moved
    curve = BSplineCurve(cps;degree=2)
    # use BDIM-σ distance function, make a body and a Simulation
    body = DynamicNurbsBody(curve;thk=(u)->thk,boundary=false)
    Simulation((4L,4L,32),(U,0,0),L;U,ν=U*L/Re,body,T=Float64,mem)
end

function run(L,h,thk,N)
    sim = make_sim(L,h,thk) # don't do 48, 96

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
    p₃,p₄ = segmts[N-2]
    p₅,p₆ = segmts[N-1]
    p₇,p₈ = segmts[N]

    _sdf(a::Simulation) = (WaterLily.@loop a.flow.σ[I] = cylinder(loc(0,I),p₁,p₂,4) over I ∈ inside(a.flow.p); 
                            a.flow.σ |> Array;)
    _nds_1(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds1(loc(0,I),p₁,p₂,thk/2,segmts,N,L) over I ∈ inside(a.flow.p); 
                            a.flow.f |> Array;)
    _nds_2(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds1(loc(0,I),p₃,p₄,thk/2,segmts,N,L) over I ∈ inside(a.flow.p); 
                            a.flow.f |> Array;)
    _nds_3(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds1(loc(0,I),p₅,p₆,thk/2,segmts,N,L) over I ∈ inside(a.flow.p); 
                            a.flow.f |> Array;)
    _nds_4(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds1(loc(0,I),p₇,p₈,thk/2,segmts,N,L) over I ∈ inside(a.flow.p); 
                            a.flow.f |> Array;)
    _nds2(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= WaterLily.nds(a.body,loc(0,I),0.0) over I ∈ inside(a.flow.p); 
                            a.flow.f |> Array;)

    custom_attrib = Dict(
        "u" => velocity, "p" => pressure, "d" => _body, "nds11" => _nds_1, "nds12" => _nds_2, "nds13" => _nds_3, "nds14" => _nds_4,"sdf_4" => _sdf, "nds2" => _nds2
    )

    wr = vtkWriter("ThreeD_cylinder_no_tapper"; attrib=custom_attrib)

    apply!(x->x[2],sim.flow.p)
    write!(wr, sim)
    close(wr)

    p = pressure_force1(sim,N,thk/2,L)

    @show p
    @show(sum(getindex.(p,2))) # new force function
    @show((2^2+h^2)^0.5*L*(thk/2)^2*π)
end

run(128,0.8,12,16)