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
        p1 = sim.body.curve.(s,0) - dl*tann
        p2 = p1 + 2dl*tann
        sim.flow.f .= zero(eltype(sim.flow.p))
        WaterLily.@loop sim.flow.f[I,:] .= sim.flow.p[I].*(-nds_cy(loc(0,I,T),p1,p2,R)) over I ∈ inside(sim.flow.p)
        push!(forces,sum(T,sim.flow.f,dims=ntuple(i->i,ndims(sim.flow.p)))[:])
    end
    return forces
end

function pressure_force2(sim::Simulation,n,R,L,dl;T=promote_type(Float64,eltype(sim.flow.p)))
    forces = []; t_forces = []; 
    ds = 1/n
    global beta = zeros(n-1,2)
    for i in 1:n-1
        xp = zeros(3)
        s = i*ds
        p = sim.body.curve.(s,0)
        @show(p)
        tan = ForwardDiff.derivative(s->sim.body.curve(s),s)
        tann = tan/(norm(tan))
        α = atan(tann[3]/tann[1]) # rotation angle along y axis
        β = atan(tann[2]/tann[1]) # rotation angle along z axis
        global beta[i,1] = cos(β)
        global beta[i,2] = sin(β)
        A = [cos(α) 0 sin(α);0 1 0;-sin(α) 0 cos(α)]
        B = [cos(β) -sin(β) 0;sin(β) cos(β) 0;0 0 1]
    
        nn = 32
        dθ = 2π/nn
        for j in 1:nn

            θ = (j-1)*dθ 
            xp[1] = p[1]
            xp[2] = R*cos(θ)
            xp[3] = R*sin(θ)
 
            xpp = (A*B)*xp .+1.5
            c = (xpp-p)/norm(xpp-p)#*WaterLily.interp(SA[xpp...],sim.flow.p)
            push!(t_forces,c)
        end
    end
    return t_forces
end

function surface_area(sim::Simulation,n,R,L,dl;T=promote_type(Float64,eltype(sim.flow.p)))
    area = []
    ds = 1/n
    for i in 1:n-1
        s = i*ds
        tan = ForwardDiff.derivative(s->sim.body.curve(s),s)
        tann = tan/(norm(tan))
        p1 = sim.body.curve.(s,0) - dl*tann
        p2 = p1 + 2dl*tann
        @show(norm(p2-p1))
        sim.flow.f .= zero(eltype(sim.flow.p))
        WaterLily.@loop sim.flow.f[I,:] .= 1*norm(nds_cy(loc(0,I,T),p1,p2,R)) over I ∈ inside(sim.flow.p)
        push!(area,sum(T,sim.flow.f,dims=ntuple(i->i,ndims(sim.flow.p)))[:])
    end
    return area
end

using LinearAlgebra: dot,norm
function cylinder(x::SVector{N,T},p1,p2,R) where {N,T}
    RR = R -1
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
function nds_cy(x,p1,p2,R)
    d = cylinder(x,p1,p2,R) 
    x_p1 = x - p1
    x_p2 = x - p2
    p2_p1 = p2 - p1
    p1_p2 = p1 - p2
    cos_nose = dot(x_p1,p2_p1)/(norm(x_p1)*norm(p2_p1))
    cos_eend = dot(x_p2,p1_p2)/(norm(x_p2)*norm(p1_p2))
    sin_nose = √abs(1-cos_nose^2)
    dd = norm(x-p1)*sin_nose - R

    if cos_nose>0 && cos_eend>0
        p = norm(x_p1)*cos_nose*(p2_p1/norm(p2_p1))
        p3 = p1 + p
        n = (x - p3)/(norm(x - p3))  
        any(isnan.(n)) && return zero(x)
        return n*WaterLily.kern(clamp(dd,-1,1))
    else
        return zero(x)
    end 
end

function make_sim(L,thk,h;Re=250,U=1,mem=Array)
    cps = SA_F32[-1   0 1 
                 -h   0 h 
                 -1   0 1]*L .+ [2L,L,2L]
    weights = SA_F32[1,1,1]
    knots = SA_F32[0,0,0,1,1,1]
    curve = NurbsCurve(cps,knots,weights)
    body = DynamicNurbsBody(curve;thk=(u)->thk,boundary=false)
    Simulation((4L,2L,4L),(U,0,0),L;U,ν=U/Re,body,T=Float64,mem)
end

function run(L,thk,n,h,dl)
    sim = make_sim(L,thk,h)
    apply!(x->x[2],sim.flow.p)

    # velocity(a::Simulation) = a.flow.u |> Array;
    # pressure(a::Simulation) = a.flow.p |> Array;
    # _body(a::Simulation) = (measure_sdf!(a.flow.σ, a.body, WaterLily.time(a.flow)); 
    #                         a.flow.σ |> Array;)
    # a = []; b = []
    # ds = 1/n
    # for i in 1:n-1
    #     s = i*ds
    #     tan = ForwardDiff.derivative(s->sim.body.curve(s),s)
    #     tann = tan/(norm(tan))
    #     p11 = sim.body.curve.(s,0) - dl*tann
    #     p22 = p11 + 2dl*tann
    #     push!(a,p11)
    #     push!(b,p22)
    # end
    # @show(a)
    # @show(b)

    # p11,p22 = a[1],b[1]
    # p33,p44 = a[2],b[2]
    # p55,p66 = a[3],b[3]
    # p77,p88 = a[4],b[4]

    # _sdf1(a::Simulation) = (WaterLily.@loop a.flow.σ[I] = cylinder(loc(0,I),p11,p22,thk/2) over I ∈ inside(a.flow.p); 
    # a.flow.σ |> Array;)
    # _sdf2(a::Simulation) = (WaterLily.@loop a.flow.σ[I] = cylinder(loc(0,I),p33,p44,thk/2) over I ∈ inside(a.flow.p); 
    # a.flow.σ |> Array;)
    # _sdf3(a::Simulation) = (WaterLily.@loop a.flow.σ[I] = cylinder(loc(0,I),p55,p66,thk/2) over I ∈ inside(a.flow.p); 
    # a.flow.σ |> Array;)
    # _sdf4(a::Simulation) = (WaterLily.@loop a.flow.σ[I] = cylinder(loc(0,I),p77,p88,thk/2) over I ∈ inside(a.flow.p); 
    # a.flow.σ |> Array;)

    # _nds1(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds_cy(loc(0,I),p11,p22,thk/2) over I ∈ inside(a.flow.p); 
    #                         a.flow.f |> Array;)
    # _nds2(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds_cy(loc(0,I),p33,p44,thk/2) over I ∈ inside(a.flow.p); 
    #                         a.flow.f |> Array;)
    # _nds3(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds_cy(loc(0,I),p55,p66,thk/2) over I ∈ inside(a.flow.p); 
    #                         a.flow.f |> Array;)
    # _nds4(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds_cy(loc(0,I),p77,p88,thk/2) over I ∈ inside(a.flow.p); 
    #                         a.flow.f |> Array;)
    # _nds0(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= WaterLily.nds(a.body,loc(0,I),0.0) over I ∈ inside(a.flow.p); 
    #                         a.flow.f |> Array;)

    # custom_attrib = Dict(
    #     "u" => velocity, "p" => pressure, "d" => _body, "nds0" => _nds0, "nds1" => _nds1, "nds2" => _nds2, "nds3" => _nds3, "nds4" => _nds4, "cy1" => _sdf1, "cy2" => _sdf2, "cy3" => _sdf3, "cy4" => _sdf4
    # )
    # wr = vtkWriter("ThreeD_cylinder_validation"; attrib=custom_attrib)
    # write!(wr, sim)
    # close(wr)

    # @time global area = surface_area(sim,n,thk/2,L,dl)
    @time global p = pressure_force2(sim,n,thk/2,L,dl)
    # @show(area)
    # @show(p)


    println("analytical:")
    # @show(-(sin(π/8))^2*π*(thk/2)^2/(thk*π))
    # @show(-(sin(2π/8))^2*π*(thk/2)^2/(thk*π))
    @show(-(2/√(4+4h^2))^2*π*(thk/2)^2/(thk*π))
    # @show(-1.0*π*(thk/2)^2/(thk*π))
    println("simulation:")
    for i in 1:32
        @show(p[i])
    end
end

run(64,16,8,0.4,1)