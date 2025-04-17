using ParametricBodies: _pforce, _vforce
using WaterLily
using StaticArrays
using WriteVTK
using Splines
using LinearAlgebra
using ForwardDiff
import WaterLily: interp

function force(b::ParametricBody{T,L},flow::Flow) where {T,L<:NurbsLocator}
    reduce(hcat,[-1.0*_pforce(b.curve,flow.p,s,0,Val{true}())+
                 flow.ν*_vforce(b.curve,flow.u,s,b.dotS(s,0),0,Val{true}()) for s ∈ integration_points])
end ##
function mean(p::AbstractArray;dims=1)
    sum(p,dims=dims)/size(p,dims)
end
function ∫dξ(f) 
    x,w=Splines.gausslegendre(64)
    dot(w./2,f.((x.+1)./2))
end
bbox(l::NurbsLocator) = (l.C-l.R,l.C+l.R)
bump(ξ;μ=0.5,σ=.1,C=0.) = C+exp(-0.5(ξ-μ)^2/σ^2)


########
# function force_fea2(sim::AbstractSimulation,n,R,L,numElem;T=promote_type(Float64,eltype(sim.flow.p)))
#     forces = []
#     a = zeros(n*numElem); b = zeros(n*numElem)
#     for i in 1:n*numElem
#         a[i] = integration_points[i]-2/L
#         b[i] = integration_points[i]+2/L
#     end
#     segmts = [sim.body.curve.(ξᵢ,0) for ξᵢ in zip(a[1:end],b[1:end])]
#     for (p₁,p₂) in segmts
#         sim.flow.f .= zero(eltype(sim.flow.p))
#         WaterLily.@loop sim.flow.f[I,:] .= sim.flow.p[I].*nds1(loc(0,I,T),p₁,p₂,R,n,L) over I ∈ inside(sim.flow.p)
#         push!(forces,sum(T,sim.flow.f,dims=ntuple(i->i,ndims(sim.flow.p)))[:])
#     end
#     return hcat(forces...)
# end

# function cylinder(x::SVector{N,T},p₁,p₂,R) where {N,T}
#     RR = R-1
#     ba = p₂ - p₁
#     pa = x - p₁
#     baba = dot(ba,ba)
#     paba = dot(pa,ba)
#     x = norm(pa*baba-ba*paba) - RR*baba
#     y = abs(paba-baba*0.5f0)-baba*0.5f0
#     x2 = x*x
#     y2 = y*y*baba
#     d = (max(x,y)<0) ? -min(x2,y2) : (((x>0) ? x2 : 0)+((y>0) ? y2 : 0))
#     return convert(T,sign(d)*sqrt(abs(d))/baba-1)
# end

# using ForwardDiff
# function nds1(x,p1,p2,R,n,L)
#     d = cylinder(x,p1,p2,R) 
#     aa = x - p1
#     bb = p2 - p1
#     angn = dot(aa,bb)/(norm(aa)*norm(bb))
#     cc = x - p2
#     dd = p1 - p2
#     ange = dot(cc,dd)/(norm(cc)*norm(dd)) 

#     gg = (abs(1-angn^2))^0.5*√sum(abs2,aa)
#     hh = (abs(1-ange^2))^0.5*√sum(abs2,cc)

#     if angn>=0 && ange>0 && gg>=R-1 && hh>=R-1
#         n = ForwardDiff.gradient(x->cylinder(x,p1,p2,R), x)  
#         m = √sum(abs2,n); d /= m; n /= m*norm(p2-p1) 
#         any(isnan.(n)) && return zero(x)
#         return n*WaterLily.kern(clamp(d,-1,1))
#     else
#         return zero(x)
#     end
# end

function force_fea(sim::AbstractSimulation,N,R1,R2,numElem,L,tapper;T=promote_type(Float64,eltype(sim.flow.p)))
    n = N*numElem
    forces = zeros(n,3)

    for i in 1:n
        xp = zeros(3)
        s = integration_points[i]
        p = sim.body.curve.(s,0)
        tan = ForwardDiff.derivative(s->sim.body.curve(s),s)
        tann = tan/norm(tan)
        
        n0 = [1f0,0,0]
        k = cross(n0,tann)/norm(cross(n0,tann))
        any(isnan.(k)) && (k=[0,0,0])

        γ = acos(dot(n0,tann)/(norm(n0)*norm(tann)))

        if s > tapper
            RR = ((R2-R1)/(1-tapper))*s + (R2-(R2-R1)/(1-tapper))
            α = atan((R1-R2)/((1-tapper)*L))
            cosα = cos(α)
            sinα = sin(α) 
        else RR = R1; cosα = 1; sinα = 1
        end

        nn = 64
        dθ = 2π/nn
        dl = dθ*RR
        pf1,pf2,pf3 = 0,0,0
        for j in 1:nn
            θ = (j-1)*dθ 
            xp[1] = 0 # make the code cant measure the x-force on the tapper end
            xp[2] = (RR)*cos(θ)
            xp[3] = (RR)*sin(θ)

            xpp = cos(γ)*xp + (1-cos(γ))*(dot(k,xp))*k + sin(γ)*cross(k,xp) # Rodrigues' rotation formula
            Nds = -xpp/(norm(xpp)) .* [sinα,cosα,cosα]
            xppp = (xpp.+p.+1.5) .+0.00*Nds ### pickup the points outside the body
            pres = interp(SA[xppp...],sim.flow.p)
            c = pres.*Nds*dl

            pf1 += c[1] # on the taper end, the force in x-direction is not correct.
            pf2 += c[2]
            pf3 += c[3]
        end
        forces[i,1] = pf1
        forces[i,2] = pf2
        forces[i,3] = pf3
    end
    return (forces')
end

function force_fea_parachute(sim::AbstractSimulation,N,R1,R2,R3,numElem,L,tapper,rope,parachute;T=promote_type(Float64,eltype(sim.flow.p)))
    n = N*numElem
    forces = zeros(n,3)

    for i in 1:n
        xp = zeros(3)
        s = integration_points[i]
        p = sim.body.curve.(s,0)
        tan = ForwardDiff.derivative(s->sim.body.curve(s),s)
        tann = tan/norm(tan)
        
        n0 = [1f0,0,0]
        k = cross(n0,tann)/norm(cross(n0,tann))
        any(isnan.(k)) && (k=[0,0,0])

        γ = acos(dot(n0,tann)/(norm(n0)*norm(tann)))

        if s > tapper && s <= rope # taper end
            RR = ((R2-R1)/(1-tapper))*s + (R2-(R2-R1)/(1-tapper))
            α = atan((R1-R2)/((1-tapper)*L))
            cosα = cos(α)
            sinα = sin(α) 
        elseif s > rope && s <= parachute # rope part
            RR = R2; cosα = 1; sinα = 1
        elseif s > parachute # parachute part
            RR = R3; cosα = 1; sinα = 1
        else 
            RR = R1; cosα = 1; sinα = 1 # cylinder part
        end

        nn = 64
        dθ = 2π/nn
        dl = dθ*RR
        pf1,pf2,pf3 = 0,0,0
        for j in 1:nn
            θ = (j-1)*dθ 
            xp[1] = 0 # make the code cant measure the x-force on the tapper end
            xp[2] = (RR)*cos(θ)
            xp[3] = (RR)*sin(θ)

            xpp = cos(γ)*xp + (1-cos(γ))*(dot(k,xp))*k + sin(γ)*cross(k,xp) # Rodrigues' rotation formula
            Nds = -xpp/(norm(xpp)) .* [sinα,cosα,cosα]
            xppp = (xpp.+p.+1.5) .+0.00*Nds ### pickup the points outside the body
            pres = interp(SA[xppp...],sim.flow.p)
            c = pres.*Nds*dl

            pf1 += c[1] # on the taper end, the force in x-direction is not correct.
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
    return (forces')
end