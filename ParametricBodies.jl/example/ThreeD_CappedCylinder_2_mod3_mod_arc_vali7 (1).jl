using WaterLily
using StaticArrays
using WriteVTK
using ParametricBodies



function pressure_force1(sim::Simulation,n,R,L;T=promote_type(Float64,eltype(sim.flow.p)))
    forces = []; a=[]; b=[]; c=[]; d=[];
    ds = 1/n
    dl = 2 # half length of the cylinder
    for i in 1:n-1
        s = i*ds 
        tan = ForwardDiff.derivative(s->sim.body.curve(s),s)
        tann = tan/(norm(tan))
        p1 = sim.body.curve.(s,0) -dl*tann # nose of the cylinder
        p2 = p1 + 2dl*tann # end of the cylinder
        p3 = p1 # p3,p4 has no use
        p4 = p2
        push!(a,p1)
        push!(b,p2)
        push!(c,p3)
        push!(d,p4)
    end
    for i in 1:n-1
        sim.flow.f .= zero(eltype(sim.flow.p))
        p1 = a[i]
        p2 = b[i]
        p3 = c[i]
        p4 = d[i]
        WaterLily.@loop sim.flow.f[I,:] .= sim.flow.p[I].*(-nds1(loc(0,I,T),p1,p2,p3,p4,R)) over I ∈ inside(sim.flow.p)
        push!(forces,sum(T,sim.flow.f,dims=ntuple(i->i,ndims(sim.flow.p)))[:])
    end
    return forces
end

function surface_area(sim::Simulation,n,R,L;T=promote_type(Float64,eltype(sim.flow.p)))
    forces2 = []; a=[]; b=[]; c=[]; d=[];
    ds = 1/n
    dl = 2 # half length of the cylinder
    for i in 1:n-1
        s = i*ds 
        tan = ForwardDiff.derivative(s->sim.body.curve(s),s)
        tann = tan/(norm(tan))
        p1 = sim.body.curve.(s,0) -dl*tann
        p2 = p1 + 2dl*tann
        p3 = p1 # p3,p4 has no use
        p4 = p2
        push!(a,p1)
        push!(b,p2)
        push!(c,p3)
        push!(d,p4)
    end
    for i in 1:n-1
        sim.flow.f .= zero(eltype(sim.flow.p))
        p1 = a[i]
        p2 = b[i]
        p3 = c[i]
        p4 = d[i]
        WaterLily.@loop sim.flow.f[I,:] .= 1*norm(nds1(loc(0,I,T),p1,p2,p3,p4,R)) over I ∈ inside(sim.flow.p)
        push!(forces2,sum(T,sim.flow.f,dims=ntuple(i->i,ndims(sim.flow.p)))[:])
    end
    return forces2
end
# sim.flow.p[I].

using LinearAlgebra: dot,norm
function cylinder(x::SVector{N,T},p1,p2,R) where {N,T}
    RR = R
    ba = p2 - p1
    pa = x - p1
    baba = dot(ba,ba)
    paba = dot(pa,ba)
    x = norm(pa*baba-ba*paba) - RR*baba
    y = abs(paba-baba*0.5f0)-baba*0.5f0
    x2 = x*x
    y2 = y*y*baba
    d = (max(x,y)<0) ? -min(x2,y2) : (((x>0) ? x2 : 0)+((y>0) ? y2 : 0))
    return convert(T,sign(d)*sqrt(abs(d))/baba)
end

using ForwardDiff
function nds1(x,p1,p2,p3,p4,R) # return nds
    d = cylinder(x,p1,p2,R) 
    aa = x - p1 # vector x_p1
    bb = x - p2 # vector x_p2
    cc = p2 - p1 # vector p2_p1 
    dd = p1 - p2 # vector p1_p2
    angn = dot(aa,cc)/(√sum(abs2,aa)*√sum(abs2,cc)) # cos x_p1_p2
    ange = dot(bb,dd)/(√sum(abs2,bb)*√sum(abs2,dd)) # cos x_p2_p1
    sin_angn = √(abs(1-angn^2)) # sin x_p1_p2
    sin_ange = √(abs(1-ange^2)) # sin x_p2_p1

    # if angn>0 && ange>0 # angle method
    if  norm(aa)*sin_angn>=R-1 && norm(bb)*sin_ange>=R-1 # measure the distance from the point to cylinder axis, don't need to round the cylinder.
        n = ForwardDiff.gradient(x->cylinder(x,p1,p2,R), x)  
        m = √sum(abs2,n); d /= m; n /= m 
        any(isnan.(n)) && return zero(x)
        return n*WaterLily.kern(clamp(d,-1,1))
    else
        return zero(x)
    end
end

function nds2(x,p1,p2,p3,p4,R) # return scalar
    d = cylinder(x,p1,p2,R) 
    aa = x - p1
    bb = x - p2
    cc = p2 - p1
    dd = p1 - p2
    angn = dot(aa,cc)/(√sum(abs2,aa)*√sum(abs2,cc))
    ange = dot(bb,dd)/(√sum(abs2,bb)*√sum(abs2,dd))
    sin_angn = √(abs(1-angn^2))
    sin_ange = √(abs(1-ange^2))

    # if angn>=0 && ange>=0 # angle method
    if  norm(aa)*sin_angn>=R-1 && norm(bb)*sin_ange>=R-1 # measure the distance from the point to cylinder axis, don't need to round the cylinder.
        n = ForwardDiff.gradient(x->cylinder(x,p1,p2,R), x)  
        m = √sum(abs2,n); d /= m; n /= m 
        any(isnan.(n)) && return zero(x)
        return 1
    else
        return zero(x)
    end
end

function make_sim(L,thk;Re=250,U =1,mem=Array)
    # define a flat plat at and angle of attack
    cps = SA_F32[-1 -1 0 1 1 
                 0 1 1  1  0
                 0 0 0  0  0]*L .+ [2L,32,16]
    weights = SA_F32[1,√2/2,1,√2/2,1]
    knots =   SA_F32[0,0,0,1/2,1/2,1,1,1] # build by only one curve
    # needed if control points are moved
    curve = NurbsCurve(cps,knots,weights)
    # use BDIM-σ distance function, make a body and a Simulation
    body = DynamicNurbsBody(curve;thk=(u)->thk,boundary=false)
    Simulation((4L,2L,32),(U,0,0),L;U,ν=U*L/Re,body,T=Float64,mem)
end

function run(L,thk,N)
    sim = make_sim(L,thk) # don't do 48, 96
    apply!(x->x[2],sim.flow.p)

    # make a writer with some attributes
    velocity(a::Simulation) = a.flow.u |> Array;
    pressure(a::Simulation) = a.flow.p |> Array;
    _body(a::Simulation) = (measure_sdf!(a.flow.σ, a.body, WaterLily.time(a.flow)); 
                            a.flow.σ |> Array;)
    forces = []; a=[]; b=[]; c=[]; d=[]
    ds = 1/N
    dl = 2 # half length of the cylinder
    for i in 1:N-1
        s = i*ds 
        tan = ForwardDiff.derivative(s->sim.body.curve(s),s)
        tann = tan/(norm(tan))
        p1 = sim.body.curve.(s,0) -dl*tann
        p2 = p1 + 2dl*tann
        p3 = p1
        p4 = p2
        push!(a,p1)
        push!(b,p2)
        push!(c,p3)
        push!(d,p4)
    end

    p1,p2,p3,p4 = a[1],b[1],c[1],d[1]
    p5,p6,p7,p8 = a[2],b[2],c[2],d[2]
    p9,p10,p11,p12 = a[3],b[3],c[3],d[3]
    p13,p14,p15,p16 = a[4],b[4],c[4],d[4]
    p17,p18,p19,p20 = a[5],b[5],c[5],d[5]

    _sdf(a::Simulation) = (WaterLily.@loop a.flow.σ[I] = cylinder(loc(0,I),p1,p2,thk/2) over I ∈ inside(a.flow.p); 
                            a.flow.σ |> Array;)
    _sdf2(a::Simulation) = (WaterLily.@loop a.flow.σ[I] = cylinder(loc(0,I),p5,p6,thk/2) over I ∈ inside(a.flow.p); 
                            a.flow.σ |> Array;)
    _sdf3(a::Simulation) = (WaterLily.@loop a.flow.σ[I] = cylinder(loc(0,I),p9,p10,thk/2) over I ∈ inside(a.flow.p); 
                            a.flow.σ |> Array;)
    _sdf4(a::Simulation) = (WaterLily.@loop a.flow.σ[I] = cylinder(loc(0,I),p13,p14,thk/2) over I ∈ inside(a.flow.p); 
                            a.flow.σ |> Array;)
    _nds_1(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds2(loc(0,I),p1,p2,p3,p4,thk/2) over I ∈ inside(a.flow.p); # use nds1 to obtain the nds view
                            a.flow.f |> Array;)
    _nds_2(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds2(loc(0,I),p5,p6,p7,p8,thk/2) over I ∈ inside(a.flow.p); 
                            a.flow.f |> Array;)
    _nds_3(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds2(loc(0,I),p9,p10,p11,p12,thk/2) over I ∈ inside(a.flow.p); 
                            a.flow.f |> Array;)
    _nds_4(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds2(loc(0,I),p13,p14,p15,p16,thk/2) over I ∈ inside(a.flow.p); 
                            a.flow.f |> Array;)
    _nds_5(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds2(loc(0,I),p17,p18,p19,p20,thk/2) over I ∈ inside(a.flow.p); 
                            a.flow.f |> Array;)
    _nds2(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= WaterLily.nds(a.body,loc(0,I),0.0) over I ∈ inside(a.flow.p); 
                            a.flow.f |> Array;)

    custom_attrib = Dict(
        "u" => velocity, "p" => pressure, "d" => _body, "nds2" => _nds2, "nds11" => _nds_1, "nds12" => _nds_2, "nds13" => _nds_3, "nds14" => _nds_4, "nds15" => _nds_5, "cy1" => _sdf, "cy2" => _sdf2, "cy3" => _sdf3, "cy4" => _sdf4
    )

    wr = vtkWriter("ThreeD_cylinder_validation"; attrib=custom_attrib)
    write!(wr, sim)
    close(wr)

    @time global area = surface_area(sim,N,thk/2,L)
    @time global p = pressure_force1(sim,N,thk/2,L)

    @show(p)
    @show(area)
    println("analytical:")
    @show(-(sin(π/8))^2*π*(thk/2)^2/(thk*π))
    @show(-(sin(2π/8))^2*π*(thk/2)^2/(thk*π))
    @show(-(sin(3π/8))^2*π*(thk/2)^2/(thk*π))
    @show(-1.0*π*(thk/2)^2/(thk*π))
    println("simulation:")
    for i in 1:4
        @show(p[i]./area[i])
    end
end

run(96,16,8) # run(L,thk,number of segments)