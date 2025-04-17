using Splines
using Plots


function test_pinned_pinned_UDL(numElem=2, degP=3)
    println("Testing on pinned-pinned beam with UDL:")
    println(" numElem: ", numElem)
    println(" degP: ", degP)
    # Material properties and mesh
    ptLeft = 0.0
    ptRight = 1.0
    EI = 1.0
    EA = 1.0
    # exact_sol(x) = 1.0/(24EI).*(x .- 2x.^3 .+ x.^4) # pinned - pinned

    mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

    # boundary conditions
    Dirichlet_BC = [
        Boundary1D("Dirichlet", ptRight, 0.0; comp=1),
        Boundary1D("Dirichlet", ptRight, 0.0; comp=2),
        Boundary1D("Dirichlet", ptLeft, 0.0; comp=1),
        Boundary1D("Dirichlet", ptLeft, 0.0; comp=2)
    ]
    Neumann_BC = []

    # make a problem
    operator = FEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC, Neumann_BC)

    # uniform external loading at integration points
    force = zeros(2, 4operator.mesh.numBasis); force[2,:] .= 1.0
    result = lsolve!(operator, force)

    # get the solution
    u0 = result[1:mesh.numBasis]
    w0 = result[mesh.numBasis+1:2mesh.numBasis]
    x = LinRange(ptLeft, ptRight, numElem+1)
    u = x .+ getSol(mesh, u0, 1)
    w = getSol(mesh, w0, 1)
    return x,u,w
end

function test_fixed_fixed_UDL(numElem=2, degP=3)
    println("Testing on fixed-fixed beam with UDL:")
    println(" numElem: ", numElem)
    println(" degP: ", degP)
    # Material properties and mesh
    ptLeft = 0.0
    ptRight = 1.0
    EI = 1.0
    EA = 1.0

    mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

    # boundary conditions
    Dirichlet_BC = [
        Boundary1D("Dirichlet", ptRight, 0.0; comp=1),
        Boundary1D("Dirichlet", ptRight, 0.0; comp=2),
        Boundary1D("Dirichlet", ptLeft, 0.0; comp=1),
        Boundary1D("Dirichlet", ptLeft, 0.0; comp=2)
    ]
    Neumann_BC = [
        Boundary1D("Neumann", ptLeft, 0.0; comp=2),
        Boundary1D("Neumann", ptRight, 0.0; comp=2)
    ]

    # make a problem
    operator = FEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC, Neumann_BC)

    # uniform external loading at integration points
    force = zeros(2, 4operator.mesh.numBasis); force[2,:] .= 1.0
    result = lsolve!(operator, force)

    u0 = result[1:mesh.numBasis]
    w0 = result[mesh.numBasis+1:2mesh.numBasis]
    x = LinRange(ptLeft, ptRight, numElem+1)
    u = x .+ getSol(mesh, u0, 1)
    w = getSol(mesh, w0, 1)
    return x,u,w
end
# exact_sol(x) = x.^2/24.0.*(1 .- x).^2    # fixed - fixed

# for degP = 2:3 
#     for Nelem in 2:4
#         x,_,w = test_fixed_fixed_UDL(Nelem, degP)
#         # get the solution
#         we = exact_sol(x)
#         println(" Error: ", norm(w .- we))
#     end
# end

# res2 = []
# for numElem ∈ 2:2:32
#     # numElem = 32
#     degP=2
#     println("Testing on fixed-fixed beam with UDL:")
#     println(" numElem: ", numElem)
#     println(" degP: ", degP)
#     # Material properties and mesh
#     ptLeft = 0.0
#     ptRight = 1.0
#     EI = 1.0
#     EA = 1.0
#     exact_sol(x) = x.^2/(24EI).*(1 .- x).^2    # fixed - fixed

#     mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

#     # boundary conditions
#     Dirichlet_BC = [
#         Boundary1D("Dirichlet", ptRight, 0.0; comp=1),
#         Boundary1D("Dirichlet", ptRight, 0.0; comp=2),
#         Boundary1D("Dirichlet", ptLeft, 0.0; comp=1),
#         Boundary1D("Dirichlet", ptLeft, 0.0; comp=2)
#     ]
#     Neumann_BC = [
#         Boundary1D("Neumann", ptLeft, 0.0; comp=2),
#         Boundary1D("Neumann", ptRight, 0.0; comp=2)
#     ]

#     # make a problem
#     operator = FEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC, Neumann_BC)

#     # uniform external loading at integration points
#     int_pnts = uv_integration(operator)
#     force = zeros(2, length(int_pnts)); force[2,:] .= 1.0
#     result = lsolve!(operator, force)

#     u0 = result[1:mesh.numBasis]
#     w0 = result[mesh.numBasis+1:2mesh.numBasis]
#     x = LinRange(ptLeft, ptRight, numElem+1)
#     u = x .+ getSol(mesh, u0, 1)
#     w = getSol(mesh, w0, 1)
#     we = exact_sol(x)
#     push!(res2, norm(w .- we))
#     println("Error: ", norm(w .- we))
# # Plots.plot(u, w, label="Sol")
# # Plots.plot!(x, we, label="Exact")
# end

# res3 = []
# for numElem ∈ 2:2:32
#     # numElem = 32
#     degP=3
#     println("Testing on fixed-fixed beam with UDL:")
#     println(" numElem: ", numElem)
#     println(" degP: ", degP)
#     # Material properties and mesh
#     ptLeft = 0.0
#     ptRight = 1.0
#     EI = 1.0
#     EA = 1.0
#     exact_sol(x) = x.^2/(24EI).*(1 .- x).^2    # fixed - fixed

#     mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

#     # boundary conditions
#     Dirichlet_BC = [
#         Boundary1D("Dirichlet", ptRight, 0.0; comp=1),
#         Boundary1D("Dirichlet", ptRight, 0.0; comp=2),
#         Boundary1D("Dirichlet", ptLeft, 0.0; comp=1),
#         Boundary1D("Dirichlet", ptLeft, 0.0; comp=2)
#     ]
#     Neumann_BC = [
#         Boundary1D("Neumann", ptLeft, 0.0; comp=2),
#         Boundary1D("Neumann", ptRight, 0.0; comp=2)
#     ]

#     # make a problem
#     operator = FEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC, Neumann_BC)

#     # uniform external loading at integration points
#     int_pnts = uv_integration(operator)
#     force = zeros(2, degP*length(int_pnts)); force[2,:] .= 1.0
#     result = lsolve!(operator, force)

#     u0 = result[1:mesh.numBasis]
#     w0 = result[mesh.numBasis+1:2mesh.numBasis]
#     x = LinRange(ptLeft, ptRight, numElem+1)
#     u = x .+ getSol(mesh, u0, 1)
#     w = getSol(mesh, w0, 1)
#     we = exact_sol(x)
#     push!(res3, norm(w .- we))
#     println("Error: ", norm(w .- we))
# end

# Plots.plot(1.0./collect(2:2:32), res2, yaxis=:log10, xaxis=:log10, xlims=(1e-4,1e0), ylims=(1e-6,1e-2), label="Degree 2")
# Plots.plot!(1.0./collect(2:2:32), res3, yaxis=:log10, xaxis=:log10, xlims=(1e-4,1e0), ylims=(1e-6,1e-2), label="Degree 3")

l = @layout [a ; b]
numElem=2
degP=2

println("Testing on fixed-fixed beam with UDL:")
println(" numElem: ", numElem)
println(" degP: ", degP)
# Material properties and mesh
ptLeft = 0.0
ptRight = 1.0
EI = 1.0
EA = 1.0
exact_sol(x) = x.^2/(24EI).*(1 .- x).^2    # fixed - fixed

mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

# boundary conditions
Dirichlet_BC = [
    Boundary1D("Dirichlet", ptRight, 0.0; comp=1),
    Boundary1D("Dirichlet", ptRight, 0.0; comp=2),
    Boundary1D("Dirichlet", ptLeft, 0.0; comp=1),
    Boundary1D("Dirichlet", ptLeft, 0.0; comp=2)
]
Neumann_BC = [
    Boundary1D("Neumann", ptLeft, 0.0; comp=2),
    Boundary1D("Neumann", ptRight, 0.0; comp=2)
]

# make a problem
operator = FEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC, Neumann_BC)

# uniform external loading at integration points
int_pnts = uv_integration(operator)
force = zeros(2, degP*length(int_pnts)); force[2,:] .= 1.0
result = lsolve!(operator, force)

u0 = result[1:mesh.numBasis]
w0 = result[mesh.numBasis+1:2mesh.numBasis]
x = LinRange(ptLeft, ptRight, numElem+1)
u = x .+ getSol(mesh, u0, 1)
w = getSol(mesh, w0, 1)
we = exact_sol(x)
println("Error: ", norm(w .- we))
p1 = Plots.plot(u, w, marker=:x, label="Sol deg=2")

# same with degp=3
numElem = 2
degP = 3

mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

# boundary conditions
Dirichlet_BC = [
    Boundary1D("Dirichlet", ptRight, 0.0; comp=1),
    Boundary1D("Dirichlet", ptRight, 0.0; comp=2),
    Boundary1D("Dirichlet", ptLeft, 0.0; comp=1),
    Boundary1D("Dirichlet", ptLeft, 0.0; comp=2)
]
Neumann_BC = [
    Boundary1D("Neumann", ptLeft, 0.0; comp=2),
    Boundary1D("Neumann", ptRight, 0.0; comp=2)
]

# make a problem
operator = FEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC, Neumann_BC)

# uniform external loading at integration points
int_pnts = uv_integration(operator)
force = zeros(2, degP*length(int_pnts)); force[2,:] .= 1.0
result = lsolve!(operator, force)

u0 = result[1:mesh.numBasis]
w0 = result[mesh.numBasis+1:2mesh.numBasis]
x = LinRange(ptLeft, ptRight, numElem+1)
u = x .+ getSol(mesh, u0, 1)
w = getSol(mesh, w0, 1)
we = exact_sol(x)
println("Error: ", norm(w .- we))
plot!(p1,u, w, marker=:o, label="Sol deg=3")
plot!(p1, 0:0.01:1, exact_sol(0:0.01:1), color=:black, linewidth=1.0, label="Exact",
xlabel="x/L", ylabel="Deflection (w/L)", title="Fixed-fixed Beam with UDL (2 elements)")


numElem=2
degP=2
println("Testing on pinned-pinned beam with UDL:")
println(" numElem: ", numElem)
println(" degP: ", degP)
# Material properties and mesh
ptLeft = 0.0
ptRight = 1.0
EI = 1.0
EA = 1.0
exact_sol(x) = 1.0/(24EI).*(x .- 2x.^3 .+ x.^4) # pinned - pinned

mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

# boundary conditions
Dirichlet_BC = [
    Boundary1D("Dirichlet", ptRight, 0.0; comp=1),
    Boundary1D("Dirichlet", ptRight, 0.0; comp=2),
    Boundary1D("Dirichlet", ptLeft, 0.0; comp=1),
    Boundary1D("Dirichlet", ptLeft, 0.0; comp=2)
]
Neumann_BC = []

# make a problem
operator = FEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC, Neumann_BC)

# uniform external loading at integration points
force = zeros(2, 2operator.mesh.numBasis); force[2,:] .= 1.0
result = lsolve!(operator, force)

# get the solution
u0 = result[1:mesh.numBasis]
w0 = result[mesh.numBasis+1:2mesh.numBasis]
x = LinRange(ptLeft, ptRight, numElem+1)
u = x .+ getSol(mesh, u0, 1)
w = getSol(mesh, w0, 1)
we = exact_sol(x)
println(" Error: ", norm(w .- we))
p2 = plot(u, w, marker=:x, label="Sol deg=2")

numElem=2
degP=3
# Material properties and mesh
ptLeft = 0.0
ptRight = 1.0
EI = 1.0
EA = 1.0
exact_sol(x) = 1.0/(24EI).*(x .- 2x.^3 .+ x.^4) # pinned - pinned

mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

# boundary conditions
Dirichlet_BC = [
    Boundary1D("Dirichlet", ptRight, 0.0; comp=1),
    Boundary1D("Dirichlet", ptRight, 0.0; comp=2),
    Boundary1D("Dirichlet", ptLeft, 0.0; comp=1),
    Boundary1D("Dirichlet", ptLeft, 0.0; comp=2)
]
Neumann_BC = []

# make a problem
operator = FEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC, Neumann_BC)

# uniform external loading at integration points
force = zeros(2, 2operator.mesh.numBasis); force[2,:] .= 1.0
result = lsolve!(operator, force)

# get the solution
u0 = result[1:mesh.numBasis]
w0 = result[mesh.numBasis+1:2mesh.numBasis]
x = LinRange(ptLeft, ptRight, numElem+1)
u = x .+ getSol(mesh, u0, 1)
w = getSol(mesh, w0, 1)
we = exact_sol(x)
println(" Error: ", norm(w .- we))
plot!(p2, u, w, marker=:o, label="Sol deg=3")

plot!(p2, 0:0.01:1, exact_sol(0:0.01:1), color=:black, linewidth=1.0, label="Exact",
        xlabel="x/L", ylabel="Deflection (w/L)", title="Pinned-Pinned Beam with UDL (2 elements)")

plot(p2,p1, layout=l, size=(600,600))