using WaterLily
using StaticArrays
using WriteVTK
using ParametricBodies



function pressure_force1(sim::Simulation,n,R,L;T=promote_type(Float64,eltype(sim.flow.p)))
    forces = []; a=[]; b=[]
    ds = 1/n
    for i in 1:n-1
        s = i*ds 
        tan = ForwardDiff.derivative(s->sim.body.curve(s),s)
        tann = tan/(norm(tan))
        p1 = sim.body.curve.(s,0) -12*tann
        p2 = sim.body.curve.(s,0) +12*tann
        push!(a,p1)
        push!(b,p2)
    end
    for i in 1:n-1
        sim.flow.f .= zero(eltype(sim.flow.p))
        p1 = a[i]
        p2 = b[i]
        WaterLily.@loop sim.flow.f[I,:] .= sim.flow.p[I].*(-nds1(loc(0,I,T),p1,p2,R)) over I ∈ inside(sim.flow.p)
        push!(forces,sum(T,sim.flow.f,dims=ntuple(i->i,ndims(sim.flow.p)))[:])
        @show(p1)
        @show(p2)
    end
    return forces
end

using LinearAlgebra: dot,norm
function cylinder(x::SVector{N,T},p1,p2,R) where {N,T}
    RR = R-1
    ba = p2 - p1
    pa = x - p1
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
function nds1(x,p1,p2,R)
    d = cylinder(x,p1,p2,R) 
    aa = x - p1
    bb = p2 - p1
    angn = dot(aa,bb)/(norm(aa)*norm(bb))
    cc = x - p2
    dd = p1 - p2
    ange = dot(cc,dd)/(norm(cc)*norm(dd)) 

    gg = (abs(1-angn^2))^0.5*√sum(abs2,aa)
    hh = (abs(1-ange^2))^0.5*√sum(abs2,cc)

    if angn>=0 && ange>0 && gg>=R-1 && hh>=R-1
        n = ForwardDiff.gradient(x->cylinder(x,p1,p2,R), x)  
        m = √sum(abs2,n); d /= m; n /= m 
        any(isnan.(n)) && return zero(x)
        return n*WaterLily.kern(clamp(d,-1,1))
    else
        return zero(x)
    end
end

function make_sim(L,h,thk;Re=250,U =1,mem=Array)
    # define a flat plat at and angle of attack
    cps = SA_F32[-1 -1 0 1 1 
                 0 1 1  1  0
                 0 0 0  0  0]*L .+ [2L,64,48]
    weights = SA_F32[1,√2/2,1,√2/2,1]
    knots =   SA_F32[0,0,0,1/2,1/2,1,1,1] # build by only one curve
    # needed if control points are moved
    curve = NurbsCurve(cps,knots,weights)
    # use BDIM-σ distance function, make a body and a Simulation
    body = DynamicNurbsBody(curve;thk=(u)->thk,boundary=false)
    Simulation((4L,2L,96),(U,0,0),L;U,ν=U*L/Re,body,T=Float64,mem)
end

function run(L,h,thk,N)
    sim = make_sim(L,h,thk) # don't do 48, 96
    apply!(x->x[2],sim.flow.p)

    # make a writer with some attributes
    velocity(a::Simulation) = a.flow.u |> Array;
    pressure(a::Simulation) = a.flow.p |> Array;
    _body(a::Simulation) = (measure_sdf!(a.flow.σ, a.body, WaterLily.time(a.flow)); 
                            a.flow.σ |> Array;)
    _sdf(a::Simulation) = (WaterLily.@loop a.flow.σ[I] = cylinder(loc(0,I),p₇,p₈,thk/2) over I ∈ inside(a.flow.p); 
                            a.flow.σ |> Array;)
    _sdf2(a::Simulation) = (WaterLily.@loop a.flow.σ[I] = cylinder(loc(0,I),p₅,p₆,thk/2) over I ∈ inside(a.flow.p); 
                            a.flow.σ |> Array;)
    a=[];b=[]
    for i in 1:N-1
        ds = 1/N
        s = i*ds 
        tan = ForwardDiff.derivative(s->sim.body.curve(s),s)
        tann = tan/(norm(tan))
        p2 = sim.body.curve.(s,0) +12*tann
        p1 = sim.body.curve.(s,0) -12*tann
        push!(a,p1)
        push!(b,p2)
    end
    p₁,p₂ = a[1],b[1]
    p₃,p₄ = a[2],b[2]
    p₅,p₆ = a[3],b[3]
    p₇,p₈ = a[4],b[4]
    p₉,p10 = a[5],b[5]

    _nds_1(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds1(loc(0,I),p₁,p₂,thk/2) over I ∈ inside(a.flow.p); 
                            a.flow.f |> Array;)
    _nds_2(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds1(loc(0,I),p₃,p₄,thk/2) over I ∈ inside(a.flow.p); 
                            a.flow.f |> Array;)
    _nds_3(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds1(loc(0,I),p₅,p₆,thk/2) over I ∈ inside(a.flow.p); 
                            a.flow.f |> Array;)
    _nds_4(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds1(loc(0,I),p₇,p₈,thk/2) over I ∈ inside(a.flow.p); 
                            a.flow.f |> Array;)
    _nds_5(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds1(loc(0,I),p₉,p10,thk/2) over I ∈ inside(a.flow.p); 
                            a.flow.f |> Array;)
    _nds2(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= WaterLily.nds(a.body,loc(0,I),0.0) over I ∈ inside(a.flow.p); 
                            a.flow.f |> Array;)

    custom_attrib = Dict(
        "cy" => _sdf,"cy2" => _sdf2, "u" => velocity, "p" => pressure, "d" => _body, "nds2" => _nds2, "nds11" => _nds_1, "nds12" => _nds_2, "nds13" => _nds_3, "nds14" => _nds_4, "nds15" => _nds_5
    )

    wr = vtkWriter("ThreeD_cylinder_validation"; attrib=custom_attrib)
    write!(wr, sim)
    close(wr)


    @time global p = pressure_force1(sim,N,thk/2,L)

    @show(p)
    @show(5*thk*π)
end

run(256,0.0,64,6)