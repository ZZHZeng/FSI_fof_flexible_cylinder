# Description: Coupling methods for fluid-structure interaction simulations

include("Preconditionner.jl")

abstract type AbstractCoupling end

# utility function
function concatenate!(vec, a, b, subs)
    vec[subs[1]] = a[1,:]; vec[subs[2]] = a[2,:]; vec[subs[3]] = a[3,:];
    vec[subs[4]] = b[1,:]; vec[subs[5]] = b[2,:]; vec[subs[6]] = b[3,:];
end
function revert!(vec, a, b, subs)
    a[1,:] = vec[subs[1]]; a[2,:] = vec[subs[2]]; a[3,:] = vec[subs[3]]; 
    b[1,:] = vec[subs[4]]; b[2,:] = vec[subs[5]]; b[3,:] = vec[subs[6]];
end

"""
    Relaxation

Standard Relaxation coupling scheme for implicit fluid-structure interaction simultations.

"""
struct Relaxation{T,Vf<:AbstractArray{T}} <: AbstractCoupling
    ω :: T    # relaxation parameter
    x :: Vf   # vector of coupling variable
    r :: Vf   # vector of rediuals
    subs :: Tuple # indices
    function Relaxation(z::AbstractVector{T};relax=0.5,subs=(),mem=Array) where T
        N = length(z)
        x⁰,r = zeros(T,N) |> mem, zeros(T,N) |> mem
        new{T,typeof(x⁰)}(relax,x⁰,r,subs)
    end
end
function Relaxation(primary::AbstractArray{T},secondary::AbstractArray{T};
                    relax::T=0.5,mem=Array) where T
    n₁,m₁=size(primary); n₂,m₂=size(secondary); N = m₁*n₁+m₂*n₂
    subs = (1:m₁, m₁+1:2*m₁, 2*m₁+1:n₁*m₁, n₁*m₁+1:n₁*m₁+m₂, n₁*m₁+m₂+1:n₁*m₁+2*m₂, n₁*m₁+2*m₂+1:N) ###
    x = zeros(N)
    concatenate!(x,primary,secondary,subs)
    Relaxation(x;relax,subs,mem)
end
function update!(cp::Relaxation{T}, xᵏ; atol=10eps(T), kwargs...) where T
    # check convergence
    r₂ = res(cp.x, xᵏ); converged = r₂<atol
    println(" r₂: $r₂ converged: : $converged")
    converged && return true
    # store variable and residual
    cp.r .= xᵏ .- cp.x
    # relaxation updates
    xᵏ .= cp.x .+ cp.ω.*cp.r; cp.x .= xᵏ;
end
finalize!(cp::Relaxation, xᵏ) = nothing

struct IQNCoupling{T,Vf<:AbstractArray{T},Mf<:AbstractArray{T}} <: AbstractCoupling
    ω :: T                 # intial relaxation
    x :: Vf                # primary variable
    x̃ :: Vf                # old solver iter (not relaxed)
    r :: Vf                # primary residual
    V :: Mf                # primary residual difference
    W :: Mf                # primary variable difference
    c :: AbstractArray{T}  # least-square coefficients
    P :: AbstractPreconditionner    # preconditionner
    QR :: QRFactorization  # QR factorization
    subs :: Tuple          # sub residual indices
    svec :: Tuple
    iter :: Dict{Symbol,Int64}      # iteration counter #### origin: Int16
    function IQNCoupling(z::AbstractVector{T};relax=0.5,maxCol::Integer=100,Preconditionner=ResidualSum,subs=(),svec=(),mem=Array) where T
        N = length(z); M = min(N÷3,maxCol) ###
        x,r =         zeros(T,N) |> mem,     zeros(T,N) |> mem
        V, W, c = zeros(T,(N,M)) |> mem, zeros(T,(N,M)) |> mem, zeros(T,M) |> mem
        new{T,typeof(z),typeof(V)}(relax,z,x,r,V,W,c,Preconditionner(N;T=T,mem=mem),
                                    QRFactorization(V,0,0;f=mem),subs,svec,Dict(:k=>0,:first=>1))
    end
end
function IQNCoupling(primary::AbstractArray{T},secondary::AbstractArray{T};
                     relax::T=0.5,maxCol::Integer=100,mem=Array) where T
    n₁,m₁=size(primary); n₂,m₂=size(secondary); N = m₁*n₁+m₂*n₂; M = min(N÷3,maxCol) ### n=directions; m1=number of control point; m2=number of gauss point
    x⁰ = zeros(N)
    # subs = (1:m₁, m₁+1:n₁*m₁, n₁*m₁+1:n₁*m₁+m₂, n₁*m₁+m₂+1:N)
    subs = (1:m₁, m₁+1:2*m₁, 2*m₁+1:n₁*m₁, n₁*m₁+1:n₁*m₁+m₂, n₁*m₁+m₂+1:n₁*m₁+2*m₂, n₁*m₁+2*m₂+1:N) ###
    concatenate!(x⁰,primary,secondary,subs)
    svec = (1:n₁*m₁,n₁*m₁+1:N)
    IQNCoupling(x⁰;relax,maxCol,subs,svec,mem)
end

function update!(cp::IQNCoupling{T}, xᵏ; atol=10eps(T), stol=1e-2, kwargs...) where T # k=iteration steps
    # compute the residuals
    rᵏ = xᵏ .- cp.x; r₂ = res(cp.x, xᵏ); converged = r₂<atol
    println(" r₂: $r₂ converged: : $converged")
    # check convergence, if converged add this to the QR
    if converged
       # update V and W matrix
        k = update_VW!(cp,xᵏ,rᵏ)
        # apply the residual sum preconditioner, without recalculating
        apply_P!(cp.V,cp.P.w)
        # QR decomposition and filter columns
        apply_QR!(cp.QR, cp.V, cp.W, k, singularityLimit=0.0)
        apply_P!(cp.V,cp.P.iw) # revert scaling
        # reset the preconditionner
        cp.P.residualSum .= 0; cp.iter[:first]=1
        return true
    end
    # first step is relaxation
    if cp.iter[:k]==0
        # compute residual and store variable
        cp.r .= rᵏ; cp.x̃.=xᵏ
        # relaxation update
        xᵏ .= cp.x .+ cp.ω*cp.r; cp.x .= xᵏ
    else
        k = min(cp.iter[:k],cp.QR.dims[1]) # default we do not insert a column
        if !Bool(cp.iter[:first]) # on a first iteration, we simply apply the relaxation
            @debug "updating V and W matrix"
            k = update_VW!(cp,xᵏ,rᵏ)
        end
        cp.r .= rᵏ; cp.x̃ .= xᵏ # save old solver iter
        # residual sum preconditioner
        update_P!(cp.P,rᵏ,cp.svec,Val(false))
        # apply precondiotnner
        apply_P!(cp.V,cp.P.w)
        # recompute QR decomposition and filter columns
        ε = Bool(cp.iter[:first]) ? T(0.0) : stol
        @debug "updating QR factorization with ε=$ε and k=$k"
        apply_QR!(cp.QR, cp.V, cp.W, k, singularityLimit=ε)
        apply_P!(cp.V,cp.P.iw) # revert scaling
        # solve least-square problem 
        R = @view cp.QR.R[1:cp.QR.dims[1],1:cp.QR.dims[1]]
        Q = @view cp.QR.Q[:,1:cp.QR.dims[1]]
        apply_P!(rᵏ,cp.P.w) # apply preconditioer to the residuals
        # compute coefficients
        cᵏ = backsub(R,-Q'*rᵏ); cp.c[1:length(cᵏ)] .= cᵏ
        @debug "least-square coefficients: $cᵏ"
        # update for next step
        xᵏ .= cp.x .+ (@view cp.W[:,1:length(cᵏ)])*cᵏ .+ cp.r; cp.x .= xᵏ
    end
    cp.iter[:k]+=1; cp.iter[:first]=0
    return false
end

function update_VW!(cp,x,r)
    roll!(cp.V); roll!(cp.W)
    for I ∈ CartesianIndices((1:size(cp.V,1)))
        cp.V[I,1] = r[I] .- cp.r[I]
        cp.W[I,1] = x[I] .- cp.x̃[I]
    end
    min(cp.iter[:k],cp.QR.dims[1]+1)
end
function apply_P!(A::AbstractArray,p::AbstractVector)
    for I ∈ CartesianIndices(A)
        A[I] *= p[CartesianIndex(Base.front(I.I))]
    end
end
function apply_P!(a::AbstractVector,p::AbstractVector)
    for I ∈ CartesianIndices(a)
        a[I] *= p[I]
    end
end
function popCol!(A::AbstractArray,k)
    m,n = size(A)
    for I ∈ CartesianIndices((1:m,k:n-1))
        A[I] = A[I+CartesianIndex(0,1)] # roll the array
    end
    A[:,end].=0
end
function roll!(A::AbstractArray)
    m,n = size(A)
    for I ∈ CartesianIndices((1:m,n:-1:2))
        A[I] = A[I-CartesianIndex(0,1)] # roll the array
    end
end
"""
    relative residual norm, bounded
"""
function res(xᵏ::AbstractArray{T},xᵏ⁺¹::AbstractArray{T}) where T
    s = zero(T); r = zero(T)
    for I ∈ CartesianIndices(xᵏ)
        s += (xᵏ⁺¹[I]-xᵏ[I])^2; r += xᵏ[I]^2
    end
    return sqrt(s)/sqrt(r+eps(T))
end
