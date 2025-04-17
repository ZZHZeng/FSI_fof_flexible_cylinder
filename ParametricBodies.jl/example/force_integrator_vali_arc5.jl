using WaterLily
using StaticArrays
using WriteVTK
using ParametricBodies
using LinearAlgebra: cross,dot,norm
using ForwardDiff
import WaterLily: interp
using WaterLily: @loop,inside,inside_u,nds,∇²u
function pressure_force(sim,t=WaterLily.time(sim.flow),T=promote_type(Float64,eltype(sim.flow.p)))
    sim.flow.f .= zero(eltype(sim.flow.p))
    @loop sim.flow.f[I,:] .= sim.flow.p[I]*nds(sim.body,loc(0,I,T),t) over I ∈ inside(sim.flow.p)
    sum(T,sim.flow.f,dims=ntuple(i->i,ndims(sim.flow.p)))[:] |> Array
end
function pressure_force2(sim::Simulation,n,R,L;T=promote_type(Float64,eltype(sim.flow.p)))
    forces = zeros(n-1,3)
    ds = 1/n
    global beta = zeros(n-1,2) # for validation
    for i in 1:n-1
        xp = zeros(3)
        s = i*ds
        p = sim.body.curve.(s,0)
        tan = ForwardDiff.derivative(s->sim.body.curve(s),s)
        tann = tan/(norm(tan))

        rp = p - [2L,32,16]
        nn0 = [-1,0,0]
        slope = acos(dot(rp,nn0)/(norm(rp)*norm(nn0)))
        global beta[i,1] = sin(π-slope)
        global beta[i,2] = cos(π-slope)

        n0 = [1f0,0,0]
        if norm(cross(n0,tann)) == 0
            k = cross(n0,tann)
        else k = cross(n0,tann)/norm(cross(n0,tann))
        end
        γ = acos(dot(n0,tann)/(norm(n0)*norm(tann)))

        nn = 64
        dθ = 2π/nn
        dl = dθ*R
        pf1,pf2,pf3 = 0,0,0
        for j in 1:nn
            θ = (j-1)*dθ 
            xp[1] = 0
            xp[2] = (R)*cos(θ)
            xp[3] = (R)*sin(θ)
            
            xpp = cos(γ)*xp + (1-cos(γ))*(dot(xp,k))*k + cross((sin(γ)*k),xp)
            Nds = -xpp/(norm(xpp)) #.* [sinα,cosα,cosα]  # normal direction, consider the tapper end
            xppp = (xpp.+p.+1.5) .+ 0.00*Nds 
            pres = interp(SA[xppp...],sim.flow.p)
            c = pres.*Nds*dl

            pf1 += c[1]
            pf2 += c[2]
            pf3 += c[3]
        end
        forces[i,1] = pf1
        forces[i,2] = pf2
        forces[i,3] = pf3
    end
    return forces
end
function make_sim(L,thk;Re=250,U=1,mem=Array)
    cps = SA_F32[-1 -1 0  1  1 
                 0   1 1  1  0
                 0   0 0  0  0]*L .+ [2L,32,16]
    weights = SA_F32[1,√2/2,1,√2/2,1]
    knots =   SA_F32[0,0,0,1/2,1/2,1,1,1]
    curve = NurbsCurve(cps,knots,weights)
    body = DynamicNurbsBody(curve;thk=(u)->thk,boundary=false)
    Simulation((4L,2L,32),(U,0,0),L;U,ν=U/Re,body,T=Float64,mem)
end
function run(L,thk,n)
    sim = make_sim(L,thk)
    # @inside sim.flow.p[I] = WaterLily.μ₀(sdf(sim.body,loc(0,I),0).+2.0,1)*loc(0,I)[2] ###
    # apply!(x->x[3],sim.flow.p)
    @inside sim.pois.z[I] = WaterLily.∂(2,I,sim.pois.L)
    WaterLily.update!(sim.pois)
    solver!(sim.pois)

    velocity(a::Simulation) = a.flow.u |> Array;
    pressure(a::Simulation) = a.flow.p |> Array;
    _body(a::Simulation) = (measure_sdf!(a.flow.σ, a.body, WaterLily.time(a.flow)); 
                            a.flow.σ |> Array;)
    _nds0(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= WaterLily.nds(a.body,loc(0,I),0.0) over I ∈ inside(a.flow.p); 
                            a.flow.f |> Array;)

    custom_attrib = Dict(
        "u" => velocity, "p" => pressure, "d" => _body, "nds0" => _nds0
    )
    wr = vtkWriter("ThreeD_cylinder_validation"; attrib=custom_attrib)
    write!(wr, sim)
    close(wr)

    @time global p = pressure_force2(sim,n,thk/2,L)
    @show(pressure_force(sim)[2])
    @show((thk/2)^2*π*L*π+4/3*(thk/2)^3*π)

    println("analytical_x:")
    for k in 1:n-1
        println(-beta[k,1]*beta[k,2]*π*((thk)/2)^2)
    end
    println("analytical_y:")
    for k in 1:n-1
        println(-beta[k,1]*beta[k,1]*π*((thk)/2)^2)
    end
    println("simulation-y:")
    for i in 1:n-1
        println(p[i,2])
    end
    println("simulation-z:")
    for i in 1:n-1
        println(p[i,3])
    end
end
run(64,6,16)