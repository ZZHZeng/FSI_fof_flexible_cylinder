using LinearAlgebra: norm, dot
include("../src/QRFactorization.jl")
include("../src/Methods.jl")

function myqr(V::AbstractArray{T}) where T
    m, n = size(V)
    Q = zeros(T, m, n)
    R = zeros(T, n, n)
    # fill in
    for j in 1:n
        v = V[:, j]
        for i in 1:j-1
            R[i, j] = dot(Q[:, i], V[:, j])
            v -= R[i, j] * Q[:, i]
        end
        R[j, j] = norm(v)
        Q[:, j] = v / (R[j, j] + eps(T))
    end
    return Q, R
end

# Example, make three column identical, i.e. they should be filtered out
V = rand(10,10); V[:,5] = V[:,1] ; V[:,8] = V[:,1]
Q,R,delIndices = QRFactorization(V,1e-8);
# check with another QR factorization
Q_,R_ = myqr(V)

# pop the column that are filtered out
for k in sort(delIndices,rev=true)
    popCol!(V,k);
end

# last two column should be zero
@test all(Q[:,end-1:end] .≈ 0)
@test all(V .≈ Q*R)
@test all(Q .≈ Q_)
@test all(R .≈ R_)
