using WaterLily
using StaticArrays
using WriteVTK
using ParametricBodies


using WaterLily: @loop,inside,inside_u,nds,∇²u
function pressure_force(sim,t=WaterLily.time(sim.flow),T=promote_type(Float64,eltype(sim.flow.p)))
    sim.flow.f .= zero(eltype(sim.flow.p))
    @loop sim.flow.f[I,:] .= sim.flow.p[I]*nds(sim.body,loc(0,I,T),t) over I ∈ inside(sim.flow.p)
    sum(T,sim.flow.f,dims=ntuple(i->i,ndims(sim.flow.p)))[:] |> Array
end
# sim.flow.p[I]
using LinearAlgebra: cross
function pressure_force2(sim::Simulation,n,R,L,dl;T=promote_type(Float64,eltype(sim.flow.p)))
    t_forces = []; 
    forces = zeros(n-1,3)
    ds = 1/n
    global beta = zeros(n-1,2)
    for i in 1:n-1
        xp = zeros(3)
        s = i*ds
        p = sim.body.curve.(s,0)
        tan = ForwardDiff.derivative(s->sim.body.curve(s),s)
        tann = tan/(norm(tan))

        n0 = [1,0,0]
        k = cross(n0,tann)/norm(cross(n0,tann))
        γ = acos(dot(n0,tann)/(norm(n0)*norm(tann)))

        nn = 128
        dθ = 2π/nn
        pf1,pf2,pf3 = 0,0,0
        for j in 1:nn
            θ = (j-1)*dθ 
            xp[1] = 0
            xp[2] = (R)*cos(θ)
            xp[3] = (R)*sin(θ)
 
            xpp = cos(γ)*xp + (1-cos(γ))*(dot(xp,k))*k + cross((sin(γ)*k),xp)
            xppp = xpp .+ p .+1.5
            Nds = xpp/norm(xpp) 

            ii = trunc.(Int,xppp)
            I = CartesianIndex(ii...)
            pres = sim.flow.p[I] 
            c = (Nds)*pres

            pf1 += c[1]
            pf2 += c[2]
            pf3 += c[3]
        end
        forces[i,1] = pf1/nn
        forces[i,2] = pf2/nn
        forces[i,3] = pf3/nn
    end
    return forces
end


using LinearAlgebra: dot,norm
using ForwardDiff
function make_sim(L,thk,h;Re=250,U=1,mem=Array)
    cps = SA_F32[0  0.5 1 
                 0  h/2 h 
                 0  0.0 0]*L .+ [L,L,16]
    weights = SA_F32[1,1,1]
    knots = SA_F32[0,0,0,1,1,1]
    curve = NurbsCurve(cps,knots,weights)
    body = DynamicNurbsBody(curve;thk=(u)->thk,boundary=false)
    Simulation((4L,2L,32),(U,0,0),L;U,ν=U/Re,body,T=Float64,mem)
end

function run(L,thk,n,h,dl)
    sim = make_sim(L,thk,h)
    t₀,duration,tstep = 0,0.1,0.1;
    # apply!(x->x[2],sim.flow.p)
    # @inside sim.flow.p[I] = WaterLily.μ₀(sdf(sim.body,loc(0,I),0),1)*loc(0,I)[2]
    velocity(a::Simulation) = a.flow.u |> Array;
    pressure(a::Simulation) = a.flow.p |> Array;
    _body(a::Simulation) = (measure_sdf!(a.flow.σ, a.body, WaterLily.time(a.flow)); 
                            a.flow.σ |> Array;)
    _nds0(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= WaterLily.nds(a.body,loc(0,I),0.0) over I ∈ inside(a.flow.p); 
                            a.flow.f |> Array;)

    custom_attrib = Dict(
        "u" => velocity, "p" => pressure, "d" => _body, "nds0" => _nds0#, "nds1" => _nds1, "nds2" => _nds2, "nds3" => _nds3, "nds4" => _nds4, "cy1" => _sdf1, "cy2" => _sdf2, "cy3" => _sdf3, "cy4" => _sdf4
    )
    wr = vtkWriter("ThreeD_cylinder_validation_straight"; attrib=custom_attrib)

    for tᵢ in range(t₀,t₀+duration;step=tstep)
        sim_step!(sim,tᵢ,remeasure=false)
        # apply!(x->x[2],sim.flow.p)
        @inside sim.flow.p[I] = WaterLily.μ₀(sdf(sim.body,loc(0,I),0).+1.0,1)*loc(0,I)[2]
        write!(wr, sim)
        println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
    end
    close(wr)

    # write!(wr, sim)
    # close(wr)

    @time global p = pressure_force2(sim,n,thk/2,L,dl)
    @show(pressure_force(sim)[2])
    @show((1+h^2)^0.5*L*((thk+0)/2)^2*π + 4/3*((thk+0)/2)^3*π)

    println("analytical:")
    @show(-(2/√(4+4h^2))^2*π*((thk+0)/2)^2/((thk+0)*π))
    println("simulation:")
    for i in 1:7
        @show(-p[i,:])
    end

end

run(64,16,8,0.2,1)