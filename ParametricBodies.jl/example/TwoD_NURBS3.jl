using WaterLily
using ParametricBodies
using StaticArrays
using Plots
using WriteVTK
# using CUDA

using WaterLily: @loop,inside,inside_u,nds,∇²u
function pressure_force(sim,t=WaterLily.time(sim.flow),T=promote_type(Float64,eltype(sim.flow.p)))
    sim.flow.f .= zero(eltype(sim.flow.p))
    @loop sim.flow.f[I,:] .= sim.flow.p[I]*nds(sim.body,loc(0,I,T),t) over I ∈ inside(sim.flow.p)
    sum(T,sim.flow.f,dims=ntuple(i->i,ndims(sim.flow.p)))[:] |> Array
end
function viscous_force(sim,t=WaterLily.time(sim.flow),T=promote_type(Float64,eltype(sim.flow.u)))
    sim.flow.f .= zero(eltype(sim.flow.u))
    @loop sim.flow.f[I,:] .= -sim.flow.ν*∇²u(I,sim.flow.u)*nds(sim.body,loc(0,I,T),t) over I ∈ inside_u(sim.flow.u)
    sum(T,sim.flow.f,dims=ntuple(i->i,ndims(sim.flow.u)-1))[:] |> Array
end
function nds_obtain(sim,t=WaterLily.time(sim.flow),T=promote_type(Float64,eltype(sim.flow.u)))
    sim.flow.f .= zero(eltype(sim.flow.u))
    @loop sim.flow.f[I,:] .= 1*nds(sim.body,loc(0,I,T),t) over I ∈ inside_u(sim.flow.u)
    sim.flow.f |> Array
end
function p_obtain(sim,t=WaterLily.time(sim.flow),T=promote_type(Float64,eltype(sim.flow.u)))
    sim.flow.f .= zero(eltype(sim.flow.u))
    @loop sim.flow.f[I,:] .= sim.flow.p[I] over I ∈ inside_u(sim.flow.u)
    sim.flow.f |> Array
end

# parameters
function dynamicSpline(L,m,n,thk;Re=250,U=1,mem=Array)
    # define a flat plat at and angle of attack
    cps = SA[0   1   2
             0.0 0.2 0.4]*L .+ [m÷4,n÷2]*L # use the nose of beam to locate

    # needed if control points are moved
    weights,knots = SA[1.,1.,1.],SA[0,0,0,1,1,1.]

    # a non-boundary curve of thinckness √2/2+1
    body = DynamicNurbsBody(NurbsCurve(cps,knots,weights);thk=thk,boundary=false)

    # make sim
    Simulation((m*L,n*L),(0,U),L;U,ν=U*L/Re,body,T=Float64,mem)
end

function pressure_distribution(sim;N=64)
    lims = ParametricBodies.lims(sim.body)
    # integrate NURBS curve to compute integral
    uv_, w_ = ParametricBodies._gausslegendre(N,typeof(first(lims)))
    # map onto the (uv) interval, need a weight scalling
    scale=(last(lims)-first(lims))/2; uv_=scale*(uv_.+1); w_=scale*w_
    @show(uv_)
    @show(lims)
    [-ParametricBodies._pforce(sim.body.curve,sim.flow.p,uv,0.0,ParametricBodies.open(sim.body);δ=6) for uv in uv_]
end

velocity(a::Simulation) = a.flow.u |> Array;
pressure(a::Simulation) = a.flow.p |> Array;
NDS(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds(a.body,loc(0,I),0.0) over I in inside(a.flow.p);
                     a.flow.f |> Array;)
_normal(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= measure(a.body,loc(0,I),0.0)[2] over I in inside(a.flow.p);
                     a.flow.f |> Array;) 
body(a::Simulation) = (measure_sdf!(a.flow.σ, a.body); 
                       a.flow.σ |> Array;)
custom_attrib = Dict(
    "Velocity" => velocity,
    "Pressure" => pressure,
    "NDS" => NDS,
    "normal" => _normal,
    "Body" => body
)# this maps what to write to the name in the file

name = "2Dnurbs"

Normal_D = [] # combined nds vector
P_f = [] # combined pressure vector
F_x = [] # total force of a segment in x direction
F_y = [] # total force of a segment in y direction
f = []
area = 0
a = 0

function run(L,m,n,thk)
    sim = dynamicSpline(L,m,n,thk)#mem=CuArray);
    t₀,duration,tstep = sim_time(sim),0.1,0.1;
    wr = vtkWriter(name; attrib=custom_attrib)

    N = round(Int, 4) # number of segment, better be even, make sure length of segment be Int
    T_length = round(Int, 2L + thk) # total length of beam at t=0
    length = round(Int, 2L ÷ N) # length of each segment at t=0
    begin_x = round(Int, m÷4*L - thk/2 +1) # x coordinate of the beam nose
    curve_begin = round(Int, m÷4*L +1)
    curve_end = round(Int, curve_begin + 2L +1)
    steps = round(Int,duration/tstep)
    F_xx = zeros(steps+1,N+2)
    F_yy = zeros(steps+1,N+2) 

    for tᵢ in range(t₀,t₀+duration;step=tstep)

        # update until time tᵢ in the background
        t = sum(sim.flow.Δt[1:end-1])
        while t < tᵢ*sim.L/sim.U
            # random update
            new_pnts = (SA[0   1   2
                           0.0 0.2 0.4] .+ [m÷4,n÷2])*sim.L
            sim.body = update!(sim.body,new_pnts,sim.flow.Δt[end])
            measure!(sim,t)
            mom_step!(sim.flow,sim.pois) # evolve Flow
            t += sim.flow.Δt[end]
        end

        global Normal_D = nds_obtain(sim)
        WaterLily.apply!(x->x[2], sim.flow.p) ## for testing
        global area = pressure_force(sim)[2]
        global P_f = p_obtain(sim)

        f_x = 0
        f_y = 0
        d_t = round(Int,10*tᵢ+1)

        for k in 1:N # nth segment
                for i in round(Int, curve_begin + (k-1)*length +1) : round(Int, curve_begin + k*length) # x coordinate
                    for j in 1 : round(Int, n*L+2) # y coordinate
                        f_x += Normal_D[:,:,1][i,j] * P_f[:,:,1][i,j] 
                        f_y += Normal_D[:,:,2][i,j] * P_f[:,:,2][i,j]
                    end
                end
                F_xx[d_t,k+1] = f_x
                F_yy[d_t,k+1] = f_y
                f_x = 0
                f_y = 0
        end

        @assert round(Int, curve_begin -thk/2) == begin_x
        @assert f_y == 0
        for g in round(Int, curve_begin -thk/2) : curve_begin
            for j in 2 : round(Int, n*L+1) # y coordinate
                f_x += Normal_D[:,:,1][g,j] * P_f[:,:,1][g,j] 
                f_y += Normal_D[:,:,2][g,j] * P_f[:,:,2][g,j]
            end
        end
        F_xx[d_t,1] = f_x
        F_yy[d_t,1] = f_y
        f_x = 0
        f_y = 0

        @assert round(Int, curve_end +thk/2) == round(Int, begin_x + T_length +1)
        @assert f_y == 0
        for h in curve_end : round(Int, curve_end +thk/2)
            for j in 2 : round(Int, n*L+1) # y coordinate
                f_x += Normal_D[:,:,1][h,j] * P_f[:,:,1][h,j] 
                f_y += Normal_D[:,:,2][h,j] * P_f[:,:,2][h,j]
            end
        end
        F_xx[d_t,N+2] = f_x
        F_yy[d_t,N+2] = f_y
        f_x = 0
        f_y = 0

        # F_x = F_xx
        # F_y = F_yy

        # print time step
        println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
        write!(wr, sim)
    end
    close(wr)

    for i in 1:6
        global a += F_yy[2,i]
    end

    @show(F_yy)
    # @show(a)
    # @show(area)
    # @show(thk*(2^2+0.4^2)^0.5*L+π*(thk/2)^2)

end
run(2^5,8,6,6)

