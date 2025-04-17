using LinearAlgebra
using StaticArrays
using Plots
using IterativeSolvers
# using BenchmarkTools
include("../src/QRFactorization.jl")
include("../src/Methods.jl")

L₂(x) = sqrt(sum(abs2,x))/length(x)

# non-symmetric matrix wih know eigenvalues
N = 64
λ = collect(2 .+ (1:N));
A = triu(rand(N,N),1) + diagm(λ)
b = rand(N);

# IQNILS method requires a fixed point
H(x) = x + (b - A*x)

# GMRES
x0 = copy(b)
sol,history = IterativeSolvers.gmres(A,b;log=true,reltol=1e-16)
r3 = history.data[:resnorm]

# setup plot
p = plot(r3, marker=:s, xaxis=:log10, yaxis=:log10, label="IterativeSolvers.gmres",
         xlabel="Iteration", ylabel="Residual",
         xlim=(1,200), ylim=(1e-16,1e2), legend=:bottomleft)

# constant relaxation
x0 = copy(b)
relax = Relaxation(copy(x0);relax=0.05)

k=1; resid=[]; rᵏ=1.0
@time while L₂(rᵏ) > 1e-16 && k < 2N
    global x0, rᵏ, k, resid, sol
    # fsi uperator
    xᵏ = H(x0)
    # compute update
    update!(relax, xᵏ; atol=1e-10)
    x0 .= relax.x
    rᵏ = relax.r
    push!(resid,L₂(rᵏ))
    k+=1
end
plot!(p, resid, marker=:o, xaxis=:log10, yaxis=:log10, label="Relaxation",
      legend=:bottomleft)

# QN couling
x0 = copy(b)
IQNSolver = IQNCoupling(x0;relax=0.05)

k=1; resid=[]; rᵏ=1.0
@time while L₂(rᵏ) > 1e-16 && k < 2N
    global x0, rᵏ, k, resid, sol
    # fsi uperator
    xᵏ = H(x0)
    # compute update
    update!(IQNSolver, xᵏ; atol=1e-16, stol=1e-16)
    x0 .= IQNSolver.x
    rᵏ = IQNSolver.r
    push!(resid,L₂(rᵏ))
    k+=1
end

plot!(p, resid, marker=:o, xaxis=:log10, yaxis=:log10, label="IQN-ILS",
      legend=:bottomleft)
savefig(p, "GMRESvsIQNILS.png")
p
