using StaticArrays
"""
    AbstractBody

Immersed body Abstract Type. Any `AbstractBody` subtype must implement

    d = sdf(body::AbstractBody, x, t=0)

and

    d,n,V = measure(body::AbstractBody, x, t=0)

where `d` is the signed distance from `x` to the body at time `t`,
and `n` & `V` are the normal and velocity vectors implied at `x`.
"""
abstract type AbstractBody end
"""
    measure!(flow::Flow, body::AbstractBody; t=0, ϵ=1)

Queries the body geometry to fill the arrays:

- `flow.μ₀`, Zeroth kernel moment
- `flow.μ₁`, First kernel moment scaled by the body normal
- `flow.V`,  Body velocity
- `flow.σᵥ`, Body velocity divergence scaled by `μ₀-1`

at time `t` using an immersion kernel of size `ϵ`.

See Maertens & Weymouth, doi:[10.1016/j.cma.2014.09.007](https://doi.org/10.1016/j.cma.2014.09.007).
"""
function measure!(a::Flow{N,T},body::AbstractBody;t=zero(T),ϵ=1) where {N,T}
    a.V .= zero(T); a.μ₀ .= one(T); a.μ₁ .= zero(T); a.σᵥ .= zero(T); d²=(2+ϵ)^2
    @fastmath @inline function fill!(μ₀,μ₁,V,σᵥ,d,I)
        d[I] = sdf(body,loc(0,I,T),t,fastd²=d²)
        σᵥ[I] = WaterLily.μ₀(d[I],ϵ)-1 # cell-center array
        if d[I]^2<d²
            for i ∈ 1:N
                dᵢ,nᵢ,Vᵢ = measure(body,loc(i,I,T),t,fastd²=d²)
                V[I,i] = Vᵢ[i]
                μ₀[I,i] = WaterLily.μ₀(dᵢ,ϵ)
                for j ∈ 1:N
                    μ₁[I,i,j] = WaterLily.μ₁(dᵢ,ϵ)*nᵢ[j]
                end
            end
        elseif d[I]<zero(T)
            for i ∈ 1:N
                μ₀[I,i] = zero(T)
            end
        end
    end
    @loop fill!(a.μ₀,a.μ₁,a.V,a.σᵥ,a.σ,I) over I ∈ inside(a.p)
    BC!(a.μ₀,zeros(SVector{N,T}),a.exitBC,a.perdir) # BC on μ₀, don't fill normal component yet
    BC!(a.V ,zeros(SVector{N,T}),a.exitBC,a.perdir)
    @inside a.σᵥ[I] = a.σᵥ[I]*div(I,a.V) # scaled divergence
    correct_div!(a.σᵥ)
end

# Convolution kernel and its moments
@fastmath kern(d) = 0.5+0.5cos(π*d)
@fastmath kern₀(d) = 0.5+0.5d+0.5sin(π*d)/π
@fastmath kern₁(d) = 0.25*(1-d^2)-0.5*(d*sin(π*d)+(1+cos(π*d))/π)/π

μ₀(d,ϵ) = kern₀(clamp(d/ϵ,-1,1))
μ₁(d,ϵ) = ϵ*kern₁(clamp(d/ϵ,-1,1))

function correct_div!(σ)
    s = sum(σ)/length(inside(σ))
    abs(s) <= 2eps(eltype(s)) && return
    @inside σ[I] = σ[I]-s
end

"""
    measure_sdf!(a::AbstractArray, body::AbstractBody, t=0)

Uses `sdf(body,x,t)` to fill `a`.
"""
measure_sdf!(a::AbstractArray,body::AbstractBody,t=0) = @inside a[I] = sdf(body,loc(0,I),t)

"""
    NoBody

Use for a simulation without a body.
"""
struct NoBody <: AbstractBody end
function measure!(a::Flow,body::NoBody;t=0,ϵ=1) end
