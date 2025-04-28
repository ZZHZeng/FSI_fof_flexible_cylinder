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
function pressure_force2(sim::Simulation,n,R1,R2,R3,L,tapper,rope,parachute;T=promote_type(Float64,eltype(sim.flow.p)))
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
        rp = p - [L,16,L]
        nn0 = [1,0,0]
        slope = acos(dot(rp,nn0)/(norm(rp)*norm(nn0)))
        slope = π/2-slope
        global beta[i,1] = sin(π-slope)
        global beta[i,2] = cos(π-slope)
        ### for validation

        n0 = [1f0,0,0]
        k = cross(n0,tann)/norm(cross(n0,tann))
        any(isnan.(k)) && (k=[0,0,0])

        γ = acos(dot(n0,tann)/(norm(n0)*norm(tann)))
        # any(isnan.(γ)) && (γ=0)

        if s > tapper && s<=rope
            RR = ((R2-R1)/(1-tapper))*s + (R2-(R2-R1)/(1-tapper))
            α = atan((R1-R2)/((1-tapper)*L))
            cosα = cos(α)
            sinα = sin(α) 
            println(-(RR^2*(beta[1,1])^2*cosα*π)) # for validation, analytical solution
        elseif s>rope && s<=parachute
            RR = R2; cosα = 1; sinα=1; println(-RR^2*π*(beta[1,1])^2 )
        elseif s>parachute
            RR = R3; cosα = 1; sinα=1; println(-RR^2*π*(beta[1,1])^2 )
        else
            RR = R1; cosα = 1; sinα=1; println(-RR^2*π*(beta[1,1])^2)
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

            pf1 += c[1] # on the taper end, force in x-direction is not accurate.
            pf2 += c[2]
            pf3 += c[3]
        end
        if s > rope
            forces[i,1] = pf1 #* (R1/R2)^4 # compensate the reuduced EI
            forces[i,2] = pf2 #* (R1/R2)^4
            forces[i,3] = pf3 #* (R1/R2)^4
        else
            forces[i,1] = pf1
            forces[i,2] = pf2
            forces[i,3] = pf3
        end
    end
    return forces
end
function make_sim(L,thk1,thk2,thk3,h,tapper,rope,parachute;Re=250,U=1,mem=Array)
    cps = SA_F32[0  0.5 1 
                 0  0.0 0 
                 0  h/2 h]*L .+ [L,16,L]
    weights = SA_F32[1,1,1]
    knots = SA_F32[0,0,0,1,1,1]
    curve = NurbsCurve(cps,knots,weights)
    function thickness(u,thick1,thick2,thick3,tapper,rope,parachute)
        # u<=tapper ? thick1 : ((((thick2)/2-(thick1)/2)/(1-tapper))*u + ((thick2)/2-((thick2)/2-(thick1)/2)/(1-tapper)))*2
        if u <= tapper
            return thick1
        elseif u > tapper && u <= rope
            return ((((thick2)/2-(thick1)/2)/(1-tapper))*u + ((thick2)/2-((thick2)/2-(thick1)/2)/(1-tapper)))*2
        elseif u > rope && u <= parachute
            return thick2
        else return thick3
        end
    end
    body = DynamicNurbsBody(curve;thk=(u)->thickness(u,thk1,thk2,thk3,tapper,rope,parachute),boundary=false)
    Simulation((4L,32,2L),(U,0,0),L;U,ν=U/Re,body,T=Float64,mem)
end
function run(L,thk1,thk2,thk3,n,h,tapper,rope,parachute)
    sim = make_sim(L,thk1,thk2,thk3,h,tapper,rope,parachute)
    # @inside sim.flow.p[I] = WaterLily.μ₀(sdf(sim.body,loc(0,I),0).+1.0,1)*loc(0,I)[2]
    @inside sim.pois.z[I] = WaterLily.∂(3,I,sim.pois.L) ### 
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
    wr = vtkWriter("ThreeD_cylinder_validation_straight_parachute"; attrib=custom_attrib)
    write!(wr, sim)
    close(wr)

    println("Analytical force in z direcytion:")
    global p = pressure_force2(sim,n,(thk1)/2,(thk2)/2,(thk3)/2,L,tapper,rope,parachute)
    println("Force in y direction measure by the integrator:")
    for i in 1:n-1
        println(p[i,2])
    end
    println("Force in z direction measure by the integrator:")
    for i in 1:n-1
        println(p[i,3])
    end
    println("Coordinate of Gauss point:")
    intx=L/n
    inty=h*L/n
    for i in 1:n-1
        println("(",i*intx,",",round(i*inty,digits=3),")")
    end
end
@time run(64,8,2,12,32,0.3,0.7,0.8,0.9)