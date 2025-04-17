using Test

arrays = [Array]

@info "Test backends: $(join(arrays,", "))"
@testset "QR factorization" begin
    include("test_QN_filer.jl")
end

@testset "QN solver" begin
    include("test_IQN_solver.jl")
end