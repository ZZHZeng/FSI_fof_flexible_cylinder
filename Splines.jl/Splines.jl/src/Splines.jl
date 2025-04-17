module Splines

using UnPack
using Plots

export Plots
export @unpack

using KernelAbstractions: get_backend, @index, @kernel
export @kernel,@index,get_backend

# given an element ID, return the CI of the stiffness matrix
@inline element(i::Integer,d::Integer) = CartesianIndices((i:i+d,i:i+d))
@inline nodes(i::Integer,d::Integer) = i:i+d
@inline δ(i=0,j=0) = CartesianIndex(i,j)
# point to the part of the stiffness matrix that is symmetric, i.e. the upper right or lower left submatrix
@inline symmetric(i::Integer) = CartesianIndices((i+1:2i,1:i))
@inline symmetric(I::CartesianIndices) = CartesianIndices((last(I.indices),first(I.indices)))
@inline symmetric(I::CartesianIndex) = CartesianIndex(last(I.I),first(I.I))
# Neumman boundary condition part of the stiffness
@inline Neumann(A,n) = CartesianIndices((2n+1:size(A,1),1:size(A,2)))
"""
Stolen from WaterLily.jl
"""
macro loop(args...)
    ex,_,itr = args
    _,I,R = itr.args; sym = []
    grab!(sym,ex)     # get arguments and replace composites in `ex`
    setdiff!(sym,[I]) # don't want to pass I as an argument
    @gensym kern      # generate unique kernel function name
    return quote
        @kernel function $kern($(rep.(sym)...),@Const(I0)) # replace composite arguments
            $I = @index(Global,Cartesian)
            $I += I0
            @fastmath @inbounds $ex
        end
        # $kern(get_backend($(sym[1])),ntuple(j->j==argmax(size($R)) ? 64 : 1,length(size($R))))($(sym...),$R[1]-oneunit($R[1]),ndrange=size($R)) #problems...
        $kern(get_backend($(sym[1])),64)($(sym...),$R[1]-oneunit($R[1]),ndrange=size($R))
    end |> esc
end
function grab!(sym,ex::Expr)
    ex.head == :. && return union!(sym,[ex])      # grab composite name and return
    start = ex.head==:(call) ? 2 : 1              # don't grab function names
    foreach(a->grab!(sym,a),ex.args[start:end])   # recurse into args
    ex.args[start:end] = rep.(ex.args[start:end]) # replace composites in args
end
grab!(sym,ex::Symbol) = union!(sym,[ex])        # grab symbol name
grab!(sym,ex) = nothing
rep(ex) = ex
rep(ex::Expr) = ex.head == :. ? Symbol(ex.args[2].value) : ex
export element,nodes,δ,symmetric,@loop,Neumann

include("bernstein.jl")
export norm

include("NURBS.jl")
# export NURBS,nrbmak,findspan,nrbline

include("Mesh.jl")
export Mesh1D,getSol

include("Boundary.jl")
export Boundary1D

include("Quadrature.jl")

include("Operator.jl")
export AbstractFEOperator,StaticFEOperator,DynamicFEOperator,integrate!,applyBC!

include("Solver.jl")
export NLsolve,LineSearches,ImplicitAD,GeneralizedAlpha,Newmark
export lsolve!,nlsolve!,solve_step!,global_mass!,residual!,jacobian!

include("utils.jl")
export dⁿ,vⁿ,points,uv_integration

end