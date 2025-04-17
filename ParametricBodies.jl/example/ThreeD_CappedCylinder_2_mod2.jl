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
    @show(segmts[1][1])
    for (p₁,p₂) in segmts
        sim.flow.f .= zero(eltype(sim.flow.p))
        WaterLily.@loop sim.flow.f[I,:] .= sim.flow.p[I].*nds(sim.body,loc(0,I,T),p₁,p₂,R,segmts,n) over I ∈ inside(sim.flow.p)
        push!(forces,sum(T,sim.flow.f,dims=ntuple(i->i,ndims(sim.flow.p)))[:])
        # @show(p₁)
        # @show(p₂)
    end
    return forces
end

# function pressure_force2(sim::Simulation,n,R,t;T=promote_type(Float64,eltype(sim.flow.p)))
#     t = 0.0
#     forces = []; limits = ParametricBodies.lims(sim.body)
#     ξ = limits[1]:1/n:limits[2] # get segments
#     segmts = [sim.body.curve.(ξᵢ,0) for ξᵢ in zip(ξ[1:end-1],ξ[2:end])]
#     for (p₁,p₂) in segmts
#         sim.flow.f .= zero(eltype(sim.flow.p))
#         WaterLily.@loop sim.flow.f[I,:] .= sim.flow.p[I].*NNDS(loc(0,I,T),p₁,p₂,R,t) over I ∈ inside(sim.flow.p)
#         push!(forces,sum(T,sim.flow.f,dims=ntuple(i->i,ndims(sim.flow.p)))[:])
#         @show(p₁)
#         @show(p₂)
#     end
#     return forces
# end

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
function nds(body,x,p1,p2,R,segmts,n)
    d = cylinder(x,p1,p2,R) 
    aa = x-p1
    bb = p2 - p1
    angn = dot(aa,bb)/(norm(aa)*norm(bb))
    cc = x - p2
    dd = p1 - p2
    ange = dot(cc,dd)/(norm(cc)*norm(dd)) 

    nn = angn*norm(aa)
    mm = ange*norm(cc)

    gg = (abs(1-angn^2))^0.5*norm(aa)
    hh = (abs(1-ange^2))^0.5*norm(cc)

    nose=convert(Float64,p1[1])
    beam_nose=convert(Float64,segmts[1][1][1])
    beam_end=convert(Float64,segmts[n][1][1])

    if angn>=0 && ange>0 && nn>1 && mm>1 && nose != beam_nose && nose != beam_end
        n = ForwardDiff.gradient(x->cylinder(x,p1,p2,R), x)  
        m = √sum(abs2,n); d /= m; n /= m 
        any(isnan.(n)) && return zero(x)
        return n*WaterLily.kern(clamp(d,-1,1))
    elseif angn>=0 && ange>0 && gg>=R-1 && hh>=R-1 && nose != beam_nose && nose != beam_end
        n = ForwardDiff.gradient(x->cylinder(x,p1,p2,R), x)  
        m = √sum(abs2,n); d /= m; n /= m 
        any(isnan.(n)) && return zero(x)
        return n*WaterLily.kern(clamp(d,-1,1))
    elseif nose == beam_nose && mm>1 
        n = ForwardDiff.gradient(x->cylinder(x,p1,p2,R), x)  
        m = √sum(abs2,n); d /= m; n /= m 
        any(isnan.(n)) && return zero(x)
        return n*WaterLily.kern(clamp(d,-1,1))
    elseif nose == beam_end && nn>1
        n = ForwardDiff.gradient(x->cylinder(x,p1,p2,R), x)  
        m = √sum(abs2,n); d /= m; n /= m 
        any(isnan.(n)) && return zero(x)
        return n*WaterLily.kern(clamp(d,-1,1))
    else
        return zero(x)
    end
end
    # if angn<0
    #     return zero(x)
    # elseif ange<0
    #     return zero(x)
    # else
    #     n = ForwardDiff.gradient(x->cylinder(x,p1,p2,R), x)  
    #     m = √sum(abs2,n); d /= m; n /= m # normalize the normal vector
    #     any(isnan.(n)) && return zero(x) 
    #     return n*WaterLily.kern(clamp(d,-1,1))
    # end  
#.*WaterLily.μ₀(WaterLily.sdf(body,x,0)+0.5,0) ##!!


# function NNDS(x,p1,p2,R,t)
#     cps = hcat(p1,p2)
#     curve = BSplineCurve(cps,degree=1)
#     function thickness(u,R)
#         if u == 0.0
#             return 0
#         elseif u==1
#             return 0.0
#         else
#             return 2R
#         end
#     end
#     body = DynamicNurbsBody(curve;thk=(u)->6,boundary=false)
#     d,n,dotS = ParametricBodies.measure(body,x,t)
#     return n*WaterLily.kern(clamp(d,-1,1))#.*WaterLily.μ₀(sdf(body,x,0)+0.5,0) ##!!
# end

function make_sim(L;Re=250,U =1,ϵ=0.5,thk=12,mem=Array)
    # define a flat plat at and angle of attack
    cps = SA[0   1    2
             0.0 0.0 0.0
             0    0   0]*L .+ [L,2L,16]
    # needed if control points are moved
    curve = BSplineCurve(cps;degree=2)
    # use BDIM-σ distance function, make a body and a Simulation
    body = DynamicNurbsBody(curve;thk=(u)->thk,boundary=false)
    Simulation((4L,4L,32),(U,0,0),L;U,ν=U*L/Re,body,T=Float64,mem)
end


# make the sim
sim = make_sim(64) # don't do 48, 96

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
p₃,p₄ = segmts[2]
p₅,p₆ = segmts[3]
p₇,p₈ = segmts[4]
xn = segmts[1][1]
xe = segmts[4][1]

_sdf(a::Simulation) = (WaterLily.@loop a.flow.σ[I] = cylinder(loc(0,I),p₁,p₂,4) over I ∈ inside(a.flow.p); 
                        a.flow.σ |> Array;)
_nds_1(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds(a.body,loc(0,I),p₁,p₂,4,segmts,4) over I ∈ inside(a.flow.p); 
                        a.flow.f |> Array;)
_nds_2(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds(a.body,loc(0,I),p₃,p₄,4,segmts,4) over I ∈ inside(a.flow.p); 
                        a.flow.f |> Array;)
_nds_3(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds(a.body,loc(0,I),p₅,p₆,4,segmts,4) over I ∈ inside(a.flow.p); 
                        a.flow.f |> Array;)
_nds_4(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds(a.body,loc(0,I),p₇,p₈,4,segmts,4) over I ∈ inside(a.flow.p); 
                        a.flow.f |> Array;)
_nds2(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= WaterLily.nds(a.body,loc(0,I),0.0) over I ∈ inside(a.flow.p); 
                        a.flow.f |> Array;)

custom_attrib = Dict(
    "u" => velocity, "p" => pressure, "d" => _body, "nds11" => _nds_1, "nds12" => _nds_2, "nds13" => _nds_3, "nds14" => _nds_4,"sdf_4" => _sdf, "nds2" => _nds2
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
p = pressure_force1(sim,4,6)
p2 = 0
global p2 = pressure_force(sim)[2]

@show p
@show(sum(getindex.(p,2))+4/3*π*6^3) # new force function
@show((2^2+0.0^2)^0.5*64*36π+4/3*π*6^3)
@show(p2) # old force function