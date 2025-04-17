using WaterLily
using ParametricBodies
using JLD2
using Plots
include("vis.jl")

norm(x) = √sum(abs2,x)

# get max arrow length
function get_δmax(nurbs)
    # get straight line between start and en d
    s,e = nurbs(0.,0.),nurbs(1.,0.); d=e-s
    # get a bunch of points
    xs = [nurbs(ξ,0.0).-s for ξ ∈ 0:0.01:1]
    # get the distance to the line
    δ = (d[1]*getindex.(xs,2)-d[2]*getindex.(xs,1))/norm(d)
    return maximum(abs.(δ)) #,argmax(abs.(δ))
end

# file and corresponding density
fnames = ["fillament_$i" for i in 5:13]
densities = [0.,2.5,1.0,1.5,2.1,0.5,3,3.5,4]


# filament trajectory and speed
Uτ = []
plt = plot(dpi=300)
for fname in fnames
    data = jldopen("data/"*fname*".jld2","r")
    L = 2^5
    pos_x = []
    pos_y = []
    velocity = []
    for i in 1:length(data)
        frame = data["frame_$i"]
        X = frame["X"]/L
        pnts = frame["pnts"]/L
        push!(pos_x, pnts[1,[1,end]].+X[1])
        push!(pos_y, pnts[2,[1,end]].+X[2])
        push!(velocity, norm(frame["U"]))
    end
    # average sedimentation speed
    push!(Uτ,sum(velocity)/length(velocity))
    # plot
    plot!(getindex.(pos_x,1),getindex.(pos_y,1),linez=velocity)
    # plot!(getindex.(pos_x,2),getindex.(pos_y,2),linez=velocity,label=:none,colorbar=true)
    plot!(frame=:none)
end
plot!(plt,aspect_ratio=:equal)
savefig(plt,"fillament.svg")
plt


# filament trajectory and speed
let
    plt = plot(dpi=300)
    for fname in fnames
        data = jldopen("data/"*fname*".jld2","r")
        L = 2^5
        pos_x = []
        pos_y = []
        velocity = []
        for i in 1:5:length(data)
            frame = data["frame_$i"]
            X = frame["X"]/L
            pnts = frame["pnts"]/L
            N = size(pnts,2)÷2
            nurbs = BSplineCurve(pnts;degree=3)
            # plot the spline
            plot!(nurbs;add_cp=false,shift=(X[1],X[2]),label=:none)
            push!(pos_x, pnts[1,N].+X[1])
            push!(pos_y, pnts[2,N].+X[2])
            push!(velocity, norm(frame["U"]))
        end
        # average sedimentation speed
        push!(Uτ,sum(velocity)/length(velocity))
        # plot
        plot!(getindex.(pos_x,1),getindex.(pos_y,1),linez=velocity,label=:none)
        # plot
        plot!(frame=:none)
    end
    plot!(plt,aspect_ratio=:equal)
    savefig(plt,"fillament_2.svg")
    savefig(plt,"fillament_2.png")
    plt
end

# plot the deflection
plt = plot(dpi=300)
for fname in fnames
    data = jldopen("data/"*fname*".jld2","r")
    L = 2^5
    deformation = []
    for i in 1:length(data)
        frame = data["frame_$i"]
        X = frame["X"]/L
        pnts = frame["pnts"]/L
        pnts = frame["pnts"]/L
        nurbs = BSplineCurve(pnts;degree=3)
        push!(deformation,get_δmax(nurbs))
    end
    plot!(deformation,label=fname)
end
plot!(plt)
savefig(plt,"deflection.png")

# plot glidding and characteristic flight time
using CurveFit
let
    # gliding distance and characteristic flight time
    plt = plot(dpi=300)
    gliding = Float64[]
    for (fname,density) in zip(fnames,densities)
        data = jldopen("data/"*fname*".jld2","r")
        L = 2^5
        frame = data["frame_1"]
        X0 = frame["X"]/L
        frame = data["frame_$(length(data))"]
        X = X0 - frame["X"]/L
        push!(gliding,abs(X[1]/X[2]))
        cm = 1.5-0.5*(density/3-density/2+2)
        @show density, cm
        scatter!([cm],[abs(X[1]/X[2])],color=:black,marker=:o,markersize =6,label=:none)
    end
    # fit a curve through the points
    xs = @. 1.5-0.5*(densities/3-densities/2+2)
    fit = curve_fit(Polynomial, xs[1:end-3], gliding[1:end-3], 2)
    plot!(plt,0.5:0.001:0.71,fit.(0.5:0.001:0.71), lw=2 ,line=:dash,label=:none)
    xlabel!("Cₘ/l"); ylabel!("Glide ratio ≡ ΔX/ΔY")
    ylims!(0,1)

    # twin axis
    # ax = twinx(plt)
    # scatter!(ax,densities,Uτ,marker=:o,label=:none)
    # ylabel!(ax,"characteristic flight time τ")
    # 
    plot!(plt,xtickfont=font(18), 
        ytickfont=font(18), 
        guidefont=font(18), 
        legendfont=font(18))
    savefig(plt,"glid_vs_density.png")
    savefig(plt,"glid_vs_density.svg")
    plt
    # plt = plot(dpi=300)
    # scatter!(plt,Uτ,gliding,marker=:o,label=:none)
    # ylabel!("Glide ratio ≡ ΔX/ΔY"); xlabel!("characteristic flight time τ")
    # # ylims!(0.75,1.25)
    # savefig(plt,"glid_vs_density.png")
    # plt
end

# plot the flow for a certain case
let    
    data = jldopen("data/fillament_6x6_2.jld2","r")
    L = 2^5
    # get data
    frame = data["frame_231"]
    ω = frame["ω"]
    X = frame["X"]/L
    pnts = frame["pnts"]/L
    nurbs = BSplineCurve(pnts;degree=3)
    t = frame["t"]
    N = size(ω)
    plt = plot(dpi=600)
    # plot the vorticity and the domain
    clims=(-10,10)
    contourf!(plt,axes(ω,1)/L,axes(ω,2)/L,
            clamp.(ω',clims[1],clims[2]),linewidth=0,levels=10,color=palette(RdBu_alpha,256),
            clims=clims,aspect_ratio=:equal;dpi=300)

    # plot the spline
    plot!(nurbs;add_cp=false,shift=(0.5/L,0.5/L))

    plot!(title="tU/L $(round(t,digits=2))",aspect_ratio=:equal)
    savefig(plt,"fillament_flow.svg")
    plt
end


# data = jldopen("data/fillament_5.jld2","r")
# L = 2^5
# pos_x = []
# pos_y = []
# velocity = []
# @gif for i in 1:length(data)
#     frame = data["frame_$i"]
#     X = frame["X"]/L
#     pnts = frame["pnts"]/L
#     nurbs = BSplineCurve(pnts;degree=3)
#     plot(nurbs,aspect_ratio=:equal,xlims=(1,3),ylims=(0,2))
#     plot!(title="tU/L $(round(frame["t"],digits=2)), δ=$(round(get_δ(nurbs),digits=2))")
# end