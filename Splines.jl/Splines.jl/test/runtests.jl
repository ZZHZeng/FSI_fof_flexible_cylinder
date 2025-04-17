using Test
using Splines
using LinearAlgebra
using Plots

@testset "bernstein.jl" begin
    @test all([Splines.bernsteinBasis([-1.], 2)...] ≈ [[1. 0. 0.],[-1. 1. 0.],[0.5 -1. 0.5]])
    @test all([Splines.bernsteinBasis([ 0.], 2)...] ≈ [[.25 .5 .25],[-.5 0. .5],[0.5 -1. 0.5]])
    @test all([Splines.bernsteinBasis([ 1.], 2)...] ≈ [[0. 0. 1.],[0. -1. 1.],[0.5 -1. 0.5]])
end

@testset "Boundary.jl" begin
    @test 1.0 ≈ 1.0
end

@testset "Mesh.jl" begin
    @test 1.0 ≈ 1.0
end

@testset "NURBS.jl" begin
    @test 1.0 ≈ 1.0
end

@testset "Quadrature.jl" begin
    @test all(Splines.genGaussLegendre(1).nodes ≈ [0])
    @test all(Splines.genGaussLegendre(1).weights ≈ [2])
    @test all(Splines.genGaussLegendre(2).nodes ≈ [-√(1/3),√(1/3)])
    @test all(Splines.genGaussLegendre(2).weights ≈ [1,1])
    @test all(Splines.genGaussLegendre(3).nodes ≈ [-√(3/5),0,√(3/5)])
    @test all(Splines.genGaussLegendre(3).weights ≈ [5/9,8/9,5/9])
end

@testset "Solver.jl" begin
    @test 1.0 ≈ 1.0
end

@testset "utils.jl" begin
    @test 1.0 ≈ 1.0
end

@testset "Operator.jl" begin
    include("test_functions.jl")
    @test test_fixed_fixed_UDL(3, 4) ≤ 100eps()
    @test test_fixed_fixed_gravity(3, 4) ≤ 100eps()
    @test test_pinned_pinned_UDL(3, 4) ≤ 100eps()
    @test test_fixed_free_UDL(3, 4) ≤ 100eps()
    @test test_fixed_free_PL(3, 3) ≤ 100eps()
end

