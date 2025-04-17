# Preconditionner

abstract type AbstractPreconditionner end

"""
    No preconditionner
"""
struct NoPreconditionner{T} <: AbstractPreconditionner
    residualSum :: AbstractArray{T}
    w :: AbstractArray{T}
    iw :: AbstractArray{T}
    function NoPreconditionner(N;T=Float64,mem=Array)
        w = ones(N) |> mem
        new{T}(w,w,w)
    end
end
# do nothing, regardless of the step
update_P!(P::NoPreconditionner,r,svec,arg) = nothing

"""
    Residual sum preconditionner
"""
struct ResidualSum{T} <: AbstractPreconditionner
    residualSum :: AbstractArray{T}
    w :: AbstractArray{T}
    iw :: AbstractArray{T}
    function ResidualSum(N;T=Float64,mem=Array)
        r₀, w, iw = zeros(N) |> mem, ones(N) |> mem, ones(N) |> mem
        new{T}(r₀,w,iw)
    end
end
# reset the summation
function update_P!(pr::ResidualSum,r,svec,reset::Val{true})
    @debug "reset preconditioner scaling factor"
    pr.residualSum .= 0;
end
# update the summation
function update_P!(pr::ResidualSum,r,svec,reset::Val{false})
    for s in svec
        pr.residualSum[s] .+= norm(r[s])/norm(r)
    end
    for s in svec
        @debug "preconditioner scaling factor $(1.0/pr.residualSum[s][1])"
        if pr.residualSum[s][1] ≠ 0.0
            pr.w[s] .= 1.0./pr.residualSum[s]
            pr.iw[s] .= pr.residualSum[s]
        end
    end
end