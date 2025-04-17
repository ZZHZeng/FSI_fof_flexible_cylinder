using WaterLily
using StaticArrays
using ParametricBodies
import WaterLily: interp
import ParametricBodies: integrate
using Plots
N = 2^7
center,radius = N÷2,N÷4
p = zeros(N,N)

apply!(x->WaterLily.μ₀(√sum(abs2,x.-center)-radius,1)*x[1],p) # true hydrostatic pressure field
# apply!(x->x[2],p) # true hydrostatic pressure field
interp(SA[2,1],p) # interp is in index-space 
interp(SA[2,1].+1.5,p) # interp is in index-space
# circle NURBS
cps = SA_F32[1 1 0 -1 -1 -1  0  1 1
                0 1 1  1  0 -1 -1 -1 0]*radius .+ center
weights = SA_F32[1.,√2/2,1.,√2/2,1.,√2/2,1.,√2/2,1.]
knots =   SA_F32[0,0,0,1/4,1/4,1/2,1/2,3/4,3/4,1,1,1]
circle = NurbsCurve(cps,knots,weights)
# check a few points along the curve
# flood(p); plot!(circle,shift=(1.5,1.5))
for ps in circle.(0:0.1:1,0.0)
    println(interp(ps.+1.5,p)," ",ps[1]) # we need to add the shift to the point
end
# integrate the old function around the cylinder
integrate(s->ParametricBodies._pforce(circle,p,s,0.0,Val(false)),
          circle,0.0,(0.0,1.0);N=64)[1]/(π*(radius)^2)
# new one with correct shift
function pforce(crv,p::AbstractArray,s,t,::Val{false};δ=1)
    xᵢ = crv(s,t); nᵢ = ParametricBodies.perp(crv,s,t); nᵢ /= √(nᵢ'*nᵢ)
    return interp((xᵢ.+1.5)+δ*nᵢ,p).*nᵢ # shift for index-coordinates
end
integrate(s->pforce(circle,p,s,0.0,Val(false)),
          circle,0.0,(0.0,1.0);N=16)[1]/(π*(radius)^2)

# circular integral around the curve a coordinates s
import ParametricBodies: _gausslegendre
using LinearAlgebra: cross
@inline perp(s::SVector{3,T}) where T = (w=@SVector(rand(3)); w-=s'*w*s; w/=√(w'*w)) #https://math.stackexchange.com/a/3385346
@inline rot(α,n,t) = cos(α)*n + sin(α)*cross(n,t) # https://math.stackexchange.com/a/3130840
# 3D curve
cps = SA_F32[-1 -.5 0 .5 1
             0   0  0  0 0
             0   0  0  0 0]*radius .+ center
curve3D = BSplineCurve(cps;degree=3)
x = curve3D(0.5,0.)
tᵢ = ParametricBodies.tangent(curve3D,0.5,0.)
nᵢ = perp(tᵢ) # this should be perp to tᵢ
@assert nᵢ'*tᵢ ≈ 0
@assert rot(π,nᵢ,tᵢ)'*tᵢ ≈ 0 
@assert rot(π/4,nᵢ,tᵢ)'*tᵢ ≈ 0 
@assert rot(2π*rand(),nᵢ,tᵢ)'*tᵢ ≈ 0
# new pforce
function pforce(crv::NurbsCurve{3},p::AbstractArray,s;t=0,R=0,δ=0.05,N=16) ##
    # integrate NURBS curve to compute integral
    uv_, w_ = _gausslegendre(N,eltype(p))
    # map onto the (uv) interval, need a weight scaling
    lims = (0,2π)
    scale=(last(lims)-first(lims))/2; uv_=scale*(uv_.+1); w_=scale*w_ 
    # find the physical point and the normal there
    xᵢ = crv(s,t); tᵢ = ParametricBodies.tangent(crv,s,t); # tangent vector
    nᵢ = perp(tᵢ) # a random normal vector
    # circular integral around this points
    @inline dF(x,n) = interp((x.+1.5).+(R+δ)*n,p).*n # add the index to cartesian offset
    sum([dF(xᵢ,rot(uv,nᵢ,tᵢ))*w/R for (uv,w) in zip(uv_,w_)])
end
Radius = 16 # radius if the 3D curve
p = zeros(N,N,N);
apply!(x->WaterLily.μ₀(√sum(abs2,SA[x[2],x[3]].-center)-Radius .+ 1,1)*x[3],p) ##
# flood(p[10,inside(p[:,:,1])]); plot!(curve3D)
x_int = curve3D(0.5,0.0)
f = pforce(curve3D,p,0.5;R=Radius,N=16)/π # should be 1

# @show tᵢ
# @show nᵢ
# plot()
# for uv in collect(0:0.1:1)[1:end-1]
#     @show rot(2π*uv,nᵢ,tᵢ)
#     xi = rot(2π*uv,nᵢ,tᵢ)
#     plot!([0,xi[2]],[0,xi[3]],color="black")
#     plot!([xi[2]],[xi[3]],color="black",marker=:circle)
# end
# plot!()





# function pflow_circle(x,center,R)
#     x = x .- center .+ 1.5 # whis is that here?
#     r,θ = √sum(abs2,x),atan(x[2],x[1])
#     r>R-10 ? WaterLily.μ₀(r-R,1)*(2*R^2/r^2*cos(2θ) - R^4/r^4) : 0.0
# end
# apply!(x->pflow_circle(x,center,radius),p)
# flood(p,clims=(-3,1))
