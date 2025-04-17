using StaticArrays
using ForwardDiff
using LinearAlgebra: ×, tr, norm # can this be an issue?
using WaterLily: kern, ∂, inside_u, AbstractBody
using WaterLily

# viscous stress tensor, 
∇²u(I::CartesianIndex{2},u) = @SMatrix [∂(i,j,I,u)+∂(j,i,I,u) for i ∈ 1:2, j ∈ 1:2]
∇²u(I::CartesianIndex{3},u) = @SMatrix [∂(i,j,I,u)+∂(j,i,I,u) for i ∈ 1:3, j ∈ 1:3]
"""normal componenent integration kernel"""
@inline function nds(body::AbstractBody,x,t)
    d,n,_ = measure(body,x,t)
    n*WaterLily.kern(clamp(d,-1,1))
end
"""moment kernel"""
@inline function xnds(body::AbstractBody,x₀::SVector{N,T},x,t,ϵ) where {N,T}
    (x-x₀)×nds_ϵ(body,x,t,ϵ)
end
"""surface integral of pressure"""
function ∮nds_ϵ(p::AbstractArray{T,N},df::AbstractArray{T},body::AutoBody,t=0,ε=1.) where {T,N}
    @WaterLily.loop df[I,:] = p[I]*nds_ϵ(body,loc(0,I,T),t,ε) over I ∈ inside(p)
    [sum(@inbounds(df[inside(p),i])) for i ∈ 1:N] |> Array
end
"""curvature corrected kernel evaluated ε away from the body"""
@inline function nds_ϵ(body::AbstractBody,x,t,ε)
    d,n,_ = measure(body,x,t); κ = 0.5tr(ForwardDiff.hessian(y -> body.sdf(y,t), x))
    κ = isnan(κ) ? 0. : κ;
    n*WaterLily.kern(clamp(d-ε,-1,1))/prod(1.0.+2κ*d)
end
"""for lack of a better name, this is the surface integral of the velocity"""
function diagnostics(a::Simulation,x₀::SVector{N,T}) where {N,T}
    # get time
    t = WaterLily.time(a); Nu,n = WaterLily.size_u(a.flow.u); Inside = CartesianIndices(map(i->(2:i-1),Nu)) 
    # compute pressure  and viscous contributions
    @WaterLily.loop a.flow.f[I,:] .= -a.flow.p[I]*nds_ϵ(a.body,loc(0,I,T),t,a.ϵ) over I ∈ inside(a.flow.p)
    @WaterLily.loop a.flow.f[I,:] .+= a.flow.ν*∇²u(I,a.flow.u)*nds_ϵ(a.body,loc(0,I,T),t,a.ϵ) over I ∈ inside(Inside)
    # integrate the pressure force
    force=[sum(@inbounds(a.flow.f[inside(a.flow.p),i])) for i ∈ 1:N] |> Array
    # compute pressure moment contribution
    @WaterLily.loop a.flow.σ[I] = -a.flow.p[I]*xnds(a.body,x₀,loc(0,I,T),t,a.ϵ) over I ∈ inside(a.flow.p)
    # integrate moments
    moment=sum(@inbounds(a.flow.σ[inside(a.flow.p)]))
    return force,moment
end
"""
    ∮τnds(u::AbstractArray{T,N},df::AbstractArray{T},body::AbstractBody,t=0)

Compute the viscous force on a immersed body. 
"""
function ∮τnds(u::AbstractArray{T,N},df::AbstractArray{T,N},body::AbstractBody,t=0) where {T,N}
    Nu,_ = WaterLily.size_u(u); In = CartesianIndices(map(i->(2:i-1),Nu)) 
    @WaterLily.loop df[I,:] .= ∇²u(I,u)*nds(body,loc(0,I,T),t) over I ∈ inside(In)
    [sum(@inbounds(df[inside(In),i])) for i ∈ 1:N-1] |> Array
end
