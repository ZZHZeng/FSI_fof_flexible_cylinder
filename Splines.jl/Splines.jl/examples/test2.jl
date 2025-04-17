using Splines
using Plots

numElem=12;degP=3

println("Testing on fixed-fixed beam with UDL:")
println(" numElem: ", numElem)
println(" degP: ", degP)
# Material properties and mesh
ptLeft = 0.0
ptRight = 1.0
EI = 1000
EA = 1.0
qy = 1.0
qz = 0.1
L = 1.0

exact_sol_z(x,qz) = qz/(24EI).*(6*x.^2 .- 4*x.^3 .+ x.^4) # clamped - free
exact_sol_y1(x,qy) = qy/(24EI).*(6*(3/4*L^2)x.^2 - 4*(L/2)*x.^3)
exact_sol_y2(x,qy) = qy/(24EI).*(6x.^2 .- 4x.^3 .+ x.^4) .-1/(128EI)*(qy*L^4) .- 1/(48EI)*qy*L^3*(x.-0.5*L)

mesh, gauss_rule = Mesh1D(ptLeft, ptRight, numElem, degP)

# boundary conditions
Dirichlet_BC = [
    # Boundary1D("Dirichlet", ptRight, 0.0; comp=1),
    Boundary1D("Dirichlet", ptLeft, 0.0; comp=1),
    # Boundary1D("Dirichlet", ptRight, 0.0; comp=2),
    Boundary1D("Dirichlet", ptLeft, 0.0; comp=2)
]

Neumann_BC = [
    Boundary1D("Neumann", ptLeft, 0.0; comp=1),
    Boundary1D("Neumann", ptLeft, 0.0; comp=2)
] 

# make a problem
operator = DynamicFEOperator(mesh, gauss_rule, EI, EA, Dirichlet_BC, Neumann_BC)

# uniform external loading at integration points
force = zeros(2,length(uv_integration(operator))); force[1,25:48] .= -2qy; force[2,:] .= qz

# update the jacobian, the residual and the external force
# linearized residuals
integrate!(operator, zero(operator.resid), force)

# compute the residuals
operator.resid .= - operator.ext

# apply BC
applyBC!(operator)

# solve the system and return
result = -operator.jacob\operator.resid

name = "Fea_vali"
v0 = result[1:mesh.numBasis]
w0 = result[mesh.numBasis+1:2mesh.numBasis]
x = LinRange(ptLeft, ptRight, numElem+1)
x1 = LinRange(ptLeft, 0.5, round(Int,numElem/2)+1)
x2 = LinRange(0.5, ptRight, round(Int,numElem/2)+1)
v = getSol(mesh, v0, 1)
w = getSol(mesh, w0, 1) 

ve1 = exact_sol_y1(x1,-2qy)
ve2 = exact_sol_y2(x2,-2qy)
# @assert ve1[round(Int,numElem/2)+1] == ve2[1] 
ve = vcat(ve1[1:round(Int,numElem/2)+1],ve2[2:round(Int,numElem/2)+1])
println("Error: ", norm(v .- ve))
plot(x, v, label="Sol")
plot!(x, ve, label="Exact")
savefig(name*"_y"*".png")

we = exact_sol_z(x,qz)
println("Error: ", norm(w .- we))
plot(x, w, label="Sol")
plot!(x, we, label="Exact")
savefig(name*"_z"*".png")