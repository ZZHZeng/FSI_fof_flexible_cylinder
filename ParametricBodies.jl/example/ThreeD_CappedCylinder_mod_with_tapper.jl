using WaterLily
using StaticArrays
using WriteVTK
using ParametricBodies
# using CUDA

using WaterLily: @loop,inside,inside_u,nds,∇²u
function pressure_force1(sim,t=WaterLily.time(sim.flow),T=promote_type(Float64,eltype(sim.flow.p)))
    forces1 = []
    sim.flow.f .= zero(eltype(sim.flow.p))
    @loop sim.flow.f[I,:] .= sim.flow.p[I]*nds(sim.body,loc(0,I,T),t) over I ∈ inside(sim.flow.p)
    forces1 = sim.flow.f
    return forces1
end
function pressure_force(sim,t=WaterLily.time(sim.flow),T=promote_type(Float64,eltype(sim.flow.p)))
    sim.flow.f .= zero(eltype(sim.flow.p))
    @loop sim.flow.f[I,:] .= sim.flow.p[I]*nds(sim.body,loc(0,I,T),t) over I ∈ inside(sim.flow.p)
    sum(T,sim.flow.f,dims=ntuple(i->i,ndims(sim.flow.p)))[:] |> Array
end
function NDS11(sim,t=WaterLily.time(sim.flow),T=promote_type(Float64,eltype(sim.flow.p)))
    abc = []
    sim.flow.f .= zero(eltype(sim.flow.p))
    @loop sim.flow.f[I,:] .= 1*nds(sim.body,loc(0,I,T),t) over I ∈ inside(sim.flow.p)
    abc = sim.flow.f
    return abc
end

function make_sim(L,m,n,Re,thk;U =1,ϵ=0.5,mem=Array)
    # define a flat plat at and angle of attack
    cps = SA[0   1   2
             0.0 0.1 0.2
             0    0  0]*L .+ [1,n÷2,n÷2]*L

    # needed if control points are moved
    curve = BSplineCurve(cps;degree=2)
    function thickness(u,thick,tapper)
        u<=tapper ? thick : ((thick/2/(tapper-1))*u-thick/2/(tapper-1))*2
    end
    # use BDIM-σ distance function, make a body and a Simulation
    body = DynamicNurbsBody(curve;thk=(u)->thickness(u,6,0.8),boundary=false)
    Simulation((m*L,n*L,n*L),(U,0,0),L;U,ν=U*L/Re,body,T=Float64,mem)
end

vol=0
# make the sim
sim = make_sim(32,4,2,250,6)

# make a writer with some attributes
velocity(a::Simulation) = a.flow.u |> Array;
pressure(a::Simulation) = a.flow.p |> Array;
_body(a::Simulation) = (measure_sdf!(a.flow.σ, a.body, WaterLily.time(a.flow)); 
                        a.flow.σ |> Array;);
NDS(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= nds(a.body,loc(0,I),0.0) over I in inside(a.flow.p);
                        a.flow.f |> Array;);
_normal(a::Simulation) = (WaterLily.@loop a.flow.f[I,:] .= measure(a.body,loc(0,I),0.0)[2] over I in inside(a.flow.p);
                        a.flow.f |> Array;) 

custom_attrib = Dict(
    "u" => velocity, "p" => pressure, "d" => _body, "nds" => NDS, "normal" => _normal
)# this

# make a vtk writer
wr = vtkWriter("ThreeD_cylinder_with_tapper"; attrib=custom_attrib)

# intialize
t₀ = 0.0; duration = 0.1; tstep = 0.1
pforce,vforce = [],[]

for tᵢ in range(t₀,t₀+duration;step=tstep)
    sim_step!(sim,tᵢ,remeasure=false)
    apply!(x->x[2],sim.flow.p)
    write!(wr, sim)
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
end
close(wr)

function pforce_inter(sim,n,N,thk)

    pforce = pressure_force1(sim)
    seg_point = []
    seg_point = sim.body.curve.(0:1/N:1,0f0)
    fea = zeros(N+2,3)
    @show(seg_point)
    for i in 1:N
        for x in round(Int, seg_point[i][1]+2) : round(Int, seg_point[i+1][1]+1)
            for y in 2:(n*sim.L+1)
                for z in 2:(n*sim.L+1)
                    fea[i+1,1] += pforce[x,y,z,1]
                    fea[i+1,2] += pforce[x,y,z,2]
                    fea[i+1,3] += pforce[x,y,z,3]
                end
            end
        end
    end
    for x in round(Int, seg_point[1][1]-thk) : round(Int, seg_point[1][1]+1)
        for y in 2:(n*sim.L+1)
            for z in 2:(n*sim.L+1)
                fea[1,1] += pforce[x,y,z,1]
                fea[1,2] += pforce[x,y,z,2]
                fea[1,3] += pforce[x,y,z,3]
            end
        end
    end
    for x in round(Int, seg_point[N+1][1]+2) : round(Int, seg_point[N+1][1]+thk)
        for y in 2:(n*sim.L+1)
            for z in 2:(n*sim.L+1)
                fea[N+2,1] += pforce[x,y,z,1]
                fea[N+2,2] += pforce[x,y,z,2]
                fea[N+2,3] += pforce[x,y,z,3]
            end
        end
    end
    return fea
end

# println("whole body pforce=")
# @show(pressure_force(sim)[2])
# println("analytical volume=")
# @show(9π*(2^2+0.2^2)^0.5*32*0.8 + 2/3*π*3^3 + 1/3*9π*(2^2+0.2^2)^0.5*32*0.2)

# println("FEA method=")
# vol1=0
# @time vol11 = pforce_inter(sim,2,4,6)
# for i in 1:6
#    global  vol1 += vol11[i,2]
# end
# @show(vol1)
# @show(vol11)

# bcd = []
# global bcd = NDS11(sim)
# bcd[:,:,:,1]