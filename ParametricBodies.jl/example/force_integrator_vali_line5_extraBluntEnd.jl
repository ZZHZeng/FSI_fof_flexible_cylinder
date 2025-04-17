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
function pressure_force2(sim::Simulation,n,R1,R2,h,L,tapper;T=promote_type(Float64,eltype(sim.flow.p)))
    forces = zeros(n-1,3)
    ds = 1/n
    global beta = zeros(n-1,2) # for validation
    for i in 1:n-1
        xp = zeros(3)
        s = i*ds
        p = sim.body.curve.(s,0)
        tan = ForwardDiff.derivative(s->sim.body.curve(s),s)
        tann = tan/(norm(tan))

        ### for validation
        rp = p - [L,L,16]
        nn0 = [1,0,0]
        slope = acos(dot(rp,nn0)/(norm(rp)*norm(nn0)))
        slope = π/2-slope
        global beta[i,1] = sin(π-atan(1/h))
        global beta[i,2] = cos(π-slope)
        ### for validation

        n0 = [1f0,0,0]
        k = cross(n0,tann)/norm(cross(n0,tann))
        any(isnan.(k)) && (k=[0,0,0])

        γ = acos(dot(n0,tann)/(norm(n0)*norm(tann)))
        # any(isnan.(γ)) && (γ=0)

        if s > tapper
            RR = ((R2-R1)/(1-tapper))*s + (R2-(R2-R1)/(1-tapper))
            α = atan((R1-R2)/((1-tapper)*L))
            cosα = cos(α)
            sinα = sin(α) 

            # println("analytical_x:")
            # println(-(RR^2*(beta[1,1]*beta[1,2])*sinα*π))
            # println("analytical_y:")
            println(-(RR^2*(beta[1,1])^2*cosα*π)) # for validation, analytical solution
            # println(-RR^2*π)
        else RR = R1; cosα = 1; sinα=1; println(-RR^2*π*(beta[1,1])^2); #println(-RR^2*π*(beta[1,1]*beta[1,2]))
        end
        nn = 64
        dθ = 2π/nn
        dl = dθ*RR
        pf1,pf2,pf3 = 0,0,0
        for j in 1:nn
            θ = (j-1)*dθ 
            xp[1] = 0
            xp[2] = (RR)*cos(θ) 
            xp[3] = (RR)*sin(θ) 
            xpp = cos(γ)*xp + (1-cos(γ))*(dot(k,xp))*k + sin(γ)*cross(k,xp) # Rodrigues' rotation formula
            Nds = -xpp/(norm(xpp)) .* [sinα,cosα,cosα]  # normal direction, consider the tapper end
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
function make_sim(L,thk1,thk2,h,tapper;Re=250,U=1,mem=Array)
    cps = SA_F32[0  0.5 1 
                 0  h/2 h 
                 0  0.0 0]*L .+ [L,L,16]
    weights = SA_F32[1,1,1]
    knots = SA_F32[0,0,0,1,1,1]
    curve = NurbsCurve(cps,knots,weights)
    function thickness(u,thick1,thick2,tapper)
        u<=tapper ? thick1 : (((thick2/2-thick1/2)/(1-tapper))*u + (thick2/2-(thick2/2-thick1/2)/(1-tapper)))*2
    end
    body = DynamicNurbsBody(curve;thk=(u)->thickness(u,thk1,thk2,tapper),boundary=false)
    Simulation((4L,2L,32),(U,0,0),L;U,ν=U/Re,body,T=Float64,mem)
end
function run(L,thk1,thk2,n,h,tapper)
    sim = make_sim(L,thk1,thk2,h,tapper)
    # @inside sim.flow.p[I] = WaterLily.μ₀(sdf(sim.body,loc(0,I),0).+1.0,1)*loc(0,I)[2]
    @inside sim.pois.z[I] = WaterLily.∂(2,I,sim.pois.L) ### 
    WaterLily.update!(sim.pois)
    solver!(sim.pois)

    velocity(a::Simulation) = a.flow.u |> Array;
    pressure(a::Simulation) = a.flow.p |> Array;
    _body(a::Simulation) = (measure_sdf!(a.flow.σ, a.body, WaterLily.time(a.flow)); 
                            a.flow.σ |> Array;)
    _nds0(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= WaterLily.nds(a.body,loc(0,I),0.0) over I ∈ inside(a.flow.p); 
                            a.flow.f |> Array;)

    custom_attrib = Dict(
        "u" => velocity, "p" => pressure, "d" => _body, "nds0" => _nds0#, "nds1" => _nds1, "nds2" => _nds2, "nds3" => _nds3, "nds4" => _nds4, "cy1" => _sdf1, "cy2" => _sdf2, "cy3" => _sdf3, "cy4" => _sdf4
    )
    wr = vtkWriter("ThreeD_cylinder_validation_straight_extraBlunt"; attrib=custom_attrib)
    write!(wr, sim)
    close(wr)
    global p = pressure_force2(sim,n,(thk1)/2,(thk2)/2,h,L,tapper)
    println("simulation-y:")
    for i in 1:n-1
        println(p[i,2])
    end
    println("simulation-z:")
    for i in 1:n-1
        println(p[i,3])
    end
    println("coordinate")
    intx=L/n
    inty=h*L/n
    for i in 1:n-1
        println("(",i*intx,",",round(i*inty,digits=3),")")
    end
end
@time run(64,6,12,32,0.2,0.8)