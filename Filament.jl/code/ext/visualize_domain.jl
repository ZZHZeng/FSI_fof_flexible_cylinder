using WaterLily
using ParametricBodies
using JLD2
using Plots
include("../ext/vis.jl")

# data = jldopen("~/Workspace/Filaments.jl/code/data/fillament_5.jld2","r")
data = jldopen("data/fillament_domain.jld2","r")
plot_trace = false
L = 2^5
xlim,ylim = data["frame_$(length(data))"]["X"]
pos_x = []
pos_y = []
velocity = []
x0 = sum(data["frame_1"]["pnts"];dims=2)/length(data["frame_1"]["pnts"])/L
@gif for i in 1:length(data)
    # get data
    frame = data["frame_$i"]
    mod(i,10)==0 && @show i, frame["U"]
    ω = frame["ω"]
    X = frame["X"]/L
    pnts = frame["pnts"]/L
    nurbs = BSplineCurve(pnts;degree=3)
    t = frame["t"]
    N = size(ω)

    plot(;dpi=300)
    if plot_trace
        push!(pos_x, pnts[1,[1,end]].+X[1])
        push!(pos_y, pnts[2,[1,end]].+X[2])
        push!(velocity, norm(frame["U"]))

        # plot the trajectory and velocity
        plot!(getindex.(pos_x,1),getindex.(pos_y,1),linez=velocity,label=:none,colorbar=true)
        plot!(getindex.(pos_x,2),getindex.(pos_y,2),linez=velocity,label=:none,colorbar=true)
    end
    # plot the vorticity and the domain
    clims=(-10,10)
    contourf!(axes(ω,1)/L,axes(ω,2)/L,
              clamp.(ω',clims[1],clims[2]),linewidth=0,levels=10,color=palette(RdBu_alpha,256),
              clims=clims,aspect_ratio=:equal)
    
    # plot the spline
    plot!(nurbs;add_cp=false)
    
    # plot limits
    xlabel!("x/L"); ylabel!("y/L")
    plot!(title="tU/L $(round(t,digits=2))",aspect_ratio=:equal)
end