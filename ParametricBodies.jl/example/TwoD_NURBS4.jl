using WaterLily
using ParametricBodies
using StaticArrays
using Plots
using WriteVTK
using Distributed
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
             0.0 1.0 2.0]*L .+ [m÷4,n÷2]*L # use the nose of beam to locate

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
area = 0

function run(L,m,n,thk)
    sim = dynamicSpline(L,m,n,thk)#mem=CuArray);
    t₀,duration,tstep = sim_time(sim),0.1,0.1;
    wr = vtkWriter(name; attrib=custom_attrib)

    N = round(Int, 4) # number of segment, better be even, make sure length of segment be Int
    steps = round(Int,duration/tstep)
    F_xx = zeros(steps+1,N+2)
    F_yy = zeros(steps+1,N+2)
    seg_point = zeros(N+1)
    norm_ij_nose = 0
    norm_ij_end = 0
    norm_curve = 0
    θ_nose = 0
    θ_end = 0

    for tᵢ in range(t₀,t₀+duration;step=tstep)

        # update until time tᵢ in the background
        t = sum(sim.flow.Δt[1:end-1])
        while t < tᵢ*sim.L/sim.U
            # random update
            new_pnts = (SA[0   1   2
                           0.0 1.0 2.0] .+ [m÷4,n÷2])*sim.L
            sim.body = update!(sim.body,new_pnts,sim.flow.Δt[end])
            measure!(sim,t)
            mom_step!(sim.flow,sim.pois) # evolve Flow
            t += sim.flow.Δt[end]
        end

        seg_point = sim.body.curve.(0:1/N:1,tᵢ*sim.L/sim.U)

        Normal_D = nds_obtain(sim)
        WaterLily.apply!(x->x[2], sim.flow.p) ## for testing
        area = pressure_force(sim)[2]
        P_f = p_obtain(sim)

        f_x = 0
        f_y = 0
        d_t = round(Int,10*tᵢ+1)

        for i in 2 : m*L+1 
            for j in 2 : n*L+1 

                f_x = Normal_D[:,:,1][i,j] * P_f[:,:,1][i,j] # this code, f_x and f_y save the force of a singel cell
                f_y = Normal_D[:,:,2][i,j] * P_f[:,:,2][i,j]
                x_ij = loc(0,CartesianIndex(i,j),promote_type(Float64,eltype(sim.flow.u)))[1]
                y_ij = loc(0,CartesianIndex(i,j),promote_type(Float64,eltype(sim.flow.u)))[2]
                # println("(",i, "," ,j,")")
                # println("x_ij=",x_ij," y_ij=",y_ij)

                for k in 1 : N

                    x_nose = seg_point[k][1] 
                    y_nose = seg_point[k][2] 
                    x_end = seg_point[k+1][1]
                    y_end = seg_point[k+1][2]

                    norm_ij_nose = ((x_ij - x_nose)^2+(y_ij - y_nose)^2)^0.5
                    norm_ij_end = ((x_ij - x_end)^2+(y_ij - y_end)^2)^0.5
                    norm_curve = ((x_end - x_nose)^2+(y_end - y_nose)^2)^0.5

                    θ_nose = ((x_ij - x_nose)*(x_end - x_nose)+(y_ij - y_nose)*(y_end - y_nose))/(norm_ij_nose * norm_curve)
                    θ_end = ((x_ij - x_end)*(x_nose - x_end)+(y_ij - y_end)*(y_nose - y_end))/(norm_ij_end * norm_curve)
                    # println("(",i, "," ,j,")")
                    # println("nose=",θ_nose," end=",θ_end)

                    if θ_nose<0 && 0<θ_end<=1 && k == 1
                        F_xx[d_t,1] += f_x
                        F_yy[d_t,1] += f_y
                    elseif  0<=θ_nose<=1 && θ_end<0 && k == N
                        F_xx[d_t,N+2] += f_x
                        F_yy[d_t,N+2] += f_y
                    else
                        F_xx[d_t,k+1] += 0
                        F_yy[d_t,k+1] += 0
                    end

                    if 0<=θ_nose<=1 && 0<θ_end<=1
                        F_xx[d_t,k+1] += f_x
                        F_yy[d_t,k+1] += f_y
                    else
                        F_xx[d_t,k+1] += 0
                        F_yy[d_t,k+1] += 0
                    end
                end
            end
        end 

        # print time step
        println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
        write!(wr, sim)
    end
    close(wr)

    a=0
    for i in 1:N+2
        a += F_yy[2,i]
    end

    @show(F_yy)
    @show(a)
    @show(thk*2*2^0.5*L + π*(thk/2)^2)
    # @show(seg_point)
end
run(2^3,8,6,8)

# (63,18)
# nose=0.8162046775741121 end=-0.7818334011725454
# @show(seg_point_temp)
# seg_point[1][2] = 0
# I = inside(sim.flow.p)
# m = zeros(2,2)
# for i in I
#     @show(i)
#     # a = loc(0,i,promote_type(Float64,eltype(sim.flow.u)))
#     # @show(a)
#     # m[1,1] = a[1]
# end
# @show(m)
# # @show(loc(0,I,0))
# @show(loc(0,CartesianIndex(63,18),promote_type(Float64,eltype(sim.flow.u))))