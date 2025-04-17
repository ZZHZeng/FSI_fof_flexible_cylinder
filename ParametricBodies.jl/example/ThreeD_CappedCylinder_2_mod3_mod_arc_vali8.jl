using WaterLily
using StaticArrays
using WriteVTK
using ParametricBodies



function pressure_force1(sim::Simulation,n,R,L;T=promote_type(Float64,eltype(sim.flow.p)))
    forces = []; a=[]; b=[]; c=[]; d=[];
    ds = 1/n
    dl = 2/(2π*L) # half length of the cylinder
    for i in 1:n-1
        s = i*ds 
        s1 = s - dl # nose of the cylinder
        s2 = s + dl # end of the cylinder
        p1 = sim.body.curve.(s1,0)
        p2 = sim.body.curve.(s2,0)
        tan1 = ForwardDiff.derivative(s1->sim.body.curve(s1),s1)
        p3 = p1 + tan1/norm(tan1) 
        tan2 = ForwardDiff.derivative(s2->sim.body.curve(s2),s2)
        p4 = p2 - tan2/norm(tan2) 
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
        WaterLily.@loop sim.flow.f[I,:] .= sim.flow.p[I].*(-nds1(sim.body,loc(0,I,T),p1,p2,p3,p4,R)) over I ∈ inside(sim.flow.p)
        push!(forces,sum(T,sim.flow.f,dims=ntuple(i->i,ndims(sim.flow.p)))[:])
    end
    return forces
end

function surface_area(sim::Simulation,n,R,L;T=promote_type(Float64,eltype(sim.flow.p)))
    forces2 = []; a=[]; b=[]; c=[]; d=[];
    ds = 1/n
    dl = 2/(2π*L) # half length of the cylinder
    for i in 1:n-1
        s = i*ds 
        s1 = s - dl # nose of the cylinder
        s2 = s + dl # end of the cylinder
        p1 = sim.body.curve.(s1,0)
        p2 = sim.body.curve.(s2,0)
        tan1 = ForwardDiff.derivative(s1->sim.body.curve(s1),s1)
        p3 = p1 + tan1/norm(tan1) 
        tan2 = ForwardDiff.derivative(s2->sim.body.curve(s2),s2)
        p4 = p2 - tan2/norm(tan2) 
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
        WaterLily.@loop sim.flow.f[I,:] .= 1*norm(nds1(sim.body,loc(0,I,T),p1,p2,p3,p4,R)) over I ∈ inside(sim.flow.p)
        push!(forces2,sum(T,sim.flow.f,dims=ntuple(i->i,ndims(sim.flow.p)))[:])
    end
    return forces2
end

using LinearAlgebra: dot,norm
using ForwardDiff
function nds1(body,x,p1,p2,p3,p4,R) # return nds
    aa = x - p1 # vector x_p1
    bb = x - p2 # vector x_p2
    cc = p3 - p1
    dd = p4 - p2
    angn = dot(aa,cc)/(√sum(abs2,aa)*√sum(abs2,cc)) # cos x_p1_p2
    ange = dot(bb,dd)/(√sum(abs2,bb)*√sum(abs2,dd)) # cos x_p2_p1
    sinn = √abs(1-angn^2)
    if angn>=0 && ange>=0 # angle method
        return WaterLily.nds(body,x,0.0)
    else
        return zero(x)
    end
end

function nds2(body,x,p1,p2,p3,p4,R) # return scalar
    aa = x - p1 # vector x_p1
    bb = x - p2 # vector x_p2
    cc = p3 - p1
    dd = p4 - p2
    angn = dot(aa,cc)/(√sum(abs2,aa)*√sum(abs2,cc)) # cos x_p1_p2
    ange = dot(bb,dd)/(√sum(abs2,bb)*√sum(abs2,dd)) # cos x_p2_p1
    if angn>=0 && ange>=0 # angle method
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
    dl = 2/(2π*L) # half length of the cylinder
    for i in 1:N-1
        s = i*ds 
        s1 = s - dl # nose of the cylinder
        s2 = s + dl # end of the cylinder
        p1 = sim.body.curve.(s1,0)
        p2 = sim.body.curve.(s2,0)
        tan1 = ForwardDiff.derivative(s1->sim.body.curve(s1),s1)
        p3 = p1 + tan1/norm(tan1) 
        tan2 = ForwardDiff.derivative(s2->sim.body.curve(s2),s2)
        p4 = p2 - tan2/norm(tan2) 
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

    _nds_1(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds1(sim.body,loc(0,I),p1,p2,p3,p4,thk/2) over I ∈ inside(a.flow.p); # use nds1 to obtain the nds view
                            a.flow.f |> Array;)
    _nds_2(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds1(sim.body,loc(0,I),p5,p6,p7,p8,thk/2) over I ∈ inside(a.flow.p); 
                            a.flow.f |> Array;)
    _nds_3(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds1(sim.body,loc(0,I), p9,p10,p11,p12,thk/2) over I ∈ inside(a.flow.p); 
                            a.flow.f |> Array;)
    _nds_4(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds1(sim.body,loc(0,I),p13,p14,p15,p16,thk/2) over I ∈ inside(a.flow.p); 
                            a.flow.f |> Array;)
    _nds_5(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds1(sim.body,loc(0,I),p17,p18,p19,p20,thk/2) over I ∈ inside(a.flow.p); 
                            a.flow.f |> Array;)
    _nds2(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= WaterLily.nds(a.body,loc(0,I),0.0) over I ∈ inside(a.flow.p); 
                            a.flow.f |> Array;)

    custom_attrib = Dict(
        "u" => velocity, "p" => pressure, "d" => _body, "nds2" => _nds2, "nds11" => _nds_1, "nds12" => _nds_2, "nds13" => _nds_3, "nds14" => _nds_4, "nds15" => _nds_5
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