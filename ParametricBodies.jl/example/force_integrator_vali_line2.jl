using WaterLily
using StaticArrays
using WriteVTK
using ParametricBodies

function pressure_force(sim::Simulation,n,R,L,dl;T=promote_type(Float64,eltype(sim.flow.p)))
    forces = []
    ds = 1/n
    for i in 1:n-1
        s = i*ds
        tan = ForwardDiff.derivative(s->sim.body.curve(s),s)
        tann = tan/(norm(tan))
        p1 = sim.body.curve.(s,0) - 2dl*tann
        p2 = p1 + 4dl*tann
        p3 = sim.body.curve.(s,0) - dl*tann
        p4 = p3 + 2dl*tann
        sim.flow.f .= zero(eltype(sim.flow.p))
        WaterLily.@loop sim.flow.f[I,:] .= sim.flow.p[I].*(-nds_cy(loc(0,I,T),p1,p2,p3,p4,R)) over I ∈ inside(sim.flow.p)
        push!(forces,sum(T,sim.flow.f,dims=ntuple(i->i,ndims(sim.flow.p)))[:])
    end
    return forces
end

function surface_area(sim::Simulation,n,R,L,dl;T=promote_type(Float64,eltype(sim.flow.p)))
    area = []
    ds = 1/n
    for i in 1:n-1
        s = i*ds
        tan = ForwardDiff.derivative(s->sim.body.curve(s),s)
        tann = tan/(norm(tan))
        p1 = sim.body.curve.(s,0) - 2dl*tann
        p2 = p1 + 4dl*tann
        p3 = sim.body.curve.(s,0) - dl*tann
        p4 = p3 + 2dl*tann
        @show(norm(p2-p1))
        sim.flow.f .= zero(eltype(sim.flow.p))
        WaterLily.@loop sim.flow.f[I,:] .= 1*norm(nds_cy(loc(0,I,T),p1,p2,p3,p4,R)) over I ∈ inside(sim.flow.p)
        push!(area,sum(T,sim.flow.f,dims=ntuple(i->i,ndims(sim.flow.p)))[:])
    end
    return area
end

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
function nds_cy(x,p1,p2,p3,p4,R)
    d = cylinder(x,p1,p2,R) 
    x_p3 = x - p3
    x_p4 = x - p4
    p4_p3 = p4 - p3
    p3_p4 = p3 - p4
    cos_nose = dot(x_p3,p4_p3)/(norm(x_p3)*norm(p4_p3))
    cos_eend = dot(x_p4,p3_p4)/(norm(x_p4)*norm(p3_p4))
    # sin_nose = √abs(1-cos_nose^2)

    if cos_nose>0 && cos_eend>0
        p = norm(x_p3)*cos_nose*(p4_p3/norm(p4_p3))
        p33 = p1 + p
        n = (x - p33)/(norm(x - p33))   ####
        any(isnan.(n)) && return zero(x)
        return n*WaterLily.kern(clamp(d,-1,1))
    else
        return zero(x)
    end 
end

function make_sim(L,thk,h;Re=250,U=1,mem=Array)
    cps = SA_F32[-1   0 1 
                 -h   0 h 
                 0    0 0]*L .+ [2L,L,16]
    weights = SA_F32[1,1,1]
    knots = SA_F32[0,0,0,1,1,1]
    curve = NurbsCurve(cps,knots,weights)
    body = DynamicNurbsBody(curve;thk=(u)->thk,boundary=false)
    Simulation((4L,2L,32),(U,0,0),L;U,ν=U/Re,body,T=Float64,mem)
end

function run(L,thk,n,h,dl)
    sim = make_sim(L,thk,h)
    apply!(x->x[2],sim.flow.p)

    velocity(a::Simulation) = a.flow.u |> Array;
    pressure(a::Simulation) = a.flow.p |> Array;
    _body(a::Simulation) = (measure_sdf!(a.flow.σ, a.body, WaterLily.time(a.flow)); 
                            a.flow.σ |> Array;)
    a = []; b = []; c=[]; d=[]
    ds = 1/n
    for i in 1:n-1
        s = i*ds
        tan = ForwardDiff.derivative(s->sim.body.curve(s),s)
        tann = tan/(norm(tan))
        p111 = sim.body.curve.(s,0) - 2dl*tann
        p222 = p111 + 4dl*tann
        p333 = sim.body.curve.(s,0) - dl*tann
        p444 = p333 + 2dl*tann
        push!(a,p111)
        push!(b,p222)
        push!(c,p333)
        push!(d,p444)
    end
    @show(a)
    @show(b)

    p111,p222,p333,p444 = a[1],b[1],c[1],d[1]
    p555,p666,p777,p888 = a[2],b[2],c[2],d[2]
    p999,p1010,p1111,p1212 = a[3],b[3],c[3],d[3]
    p1313,p1414,p1515,p1616 = a[4],b[4],c[4],d[4]

    _sdf1(a::Simulation) = (WaterLily.@loop a.flow.σ[I] = cylinder(loc(0,I),p111,p222,thk/2) over I ∈ inside(a.flow.p); 
    a.flow.σ |> Array;)
    _sdf2(a::Simulation) = (WaterLily.@loop a.flow.σ[I] = cylinder(loc(0,I),p555,p666,thk/2) over I ∈ inside(a.flow.p); 
    a.flow.σ |> Array;)
    _sdf3(a::Simulation) = (WaterLily.@loop a.flow.σ[I] = cylinder(loc(0,I),p999,p1010,thk/2) over I ∈ inside(a.flow.p); 
    a.flow.σ |> Array;)
    _sdf4(a::Simulation) = (WaterLily.@loop a.flow.σ[I] = cylinder(loc(0,I),p1313,p1414,thk/2) over I ∈ inside(a.flow.p); 
    a.flow.σ |> Array;)

    _nds1(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds_cy(loc(0,I),p111,p222,p333,p444,thk/2) over I ∈ inside(a.flow.p); 
                            a.flow.f |> Array;)
    _nds2(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds_cy(loc(0,I),p444,p555,p666,p777,thk/2) over I ∈ inside(a.flow.p); 
                            a.flow.f |> Array;)
    _nds3(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds_cy(loc(0,I),p999,p1010,p1111,p1212,thk/2) over I ∈ inside(a.flow.p); 
                            a.flow.f |> Array;)
    _nds4(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds_cy(loc(0,I),p1313,p1414,p1515,p1616,thk/2) over I ∈ inside(a.flow.p); 
                            a.flow.f |> Array;)
    _nds0(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= WaterLily.nds(a.body,loc(0,I),0.0) over I ∈ inside(a.flow.p); 
                            a.flow.f |> Array;)

    custom_attrib = Dict(
        "u" => velocity, "p" => pressure, "d" => _body, "nds0" => _nds0, "nds1" => _nds1, "nds2" => _nds2, "nds3" => _nds3, "nds4" => _nds4, "cy1" => _sdf1, "cy2" => _sdf2, "cy3" => _sdf3, "cy4" => _sdf4
    )
    wr = vtkWriter("ThreeD_cylinder_validation"; attrib=custom_attrib)
    write!(wr, sim)
    close(wr)

    @time global area = surface_area(sim,n,thk/2,L,dl)
    @time global p = pressure_force(sim,n,thk/2,L,dl)
    @show(area)
    @show(p)

    println("analytical:")
    # @show(-(sin(π/8))^2*π*(thk/2)^2/(thk*π))
    # @show(-(sin(2π/8))^2*π*(thk/2)^2/(thk*π))
    @show(-(2/√(4+4h^2))^2*π*(thk/2)^2/(thk*π))
    # @show(-1.0*π*(thk/2)^2/(thk*π))
    println("simulation:")
    for i in 1:4
        @show(p[i]./area[i])
    end
end

run(96,16,8,0.4,1)