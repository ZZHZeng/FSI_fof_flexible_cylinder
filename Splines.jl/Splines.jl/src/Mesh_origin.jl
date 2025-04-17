# mesh module
mutable struct Mesh{T}
    elemVertex    :: AbstractArray{T,2}
    elemNode      :: AbstractVector{UnitRange{Int64}}
    degP          :: AbstractArray{Int16,1}
    C             :: AbstractArray{AbstractArray{T},1}
    B             :: AbstractArray{T}
    dB            :: AbstractArray{T}
    ddB           :: AbstractArray{T}
    N             :: AbstractArray{AbstractArray{T},1}
    dN            :: AbstractArray{AbstractArray{T},1}
    ddN           :: AbstractArray{AbstractArray{T},1}
    Jac           :: AbstractArray{T,1}
    numBasis      :: Int16
    numElem       :: Int16
    controlPoints :: AbstractArray{T,2}
    weights       :: AbstractArray{T,1}
    knots         :: AbstractArray{T,1}
    function Mesh(elemVertex::AbstractArray{T,2}, elemNode, degP, C, B, dB, ddB, N, dN, ddN, Jac, numBasis, numElem,
                  controlPoints::AbstractArray{T,2}, weights::AbstractArray{T,1}, knots::AbstractArray{T,1}) where T
        new{T}(elemVertex, elemNode, degP, C, B, dB, ddB, N, dN, ddN, Jac, 
               numBasis, numElem, controlPoints, weights, knots)
    end
end

function Mesh1D(ptLeft, ptRight, numElem, degP)
    gauss_rule = genGaussLegendre(degP+1)
    nrb = nrbline(ptLeft, ptRight)
    new_knots = collect(LinRange(0, 1, numElem+1)[2:end-1])
    nrb = nrbdegelev(nrb, [degP-1])
    nrb = nrbkntins(nrb, [new_knots])
    IGAmesh = genMesh(nrb, gauss_rule)
    return IGAmesh, gauss_rule
end

"""
create the IEN (node index to element) array for a given knot vector and p and
elementVertex array

INPUT: knotVector - the given knot vector
        numElem - number of non-zero knot-spans
        p - polynomial degree
OUTPUT: IEN - array with nb rows and p+1 columns, where each row indicates the
                global node indices for a knot-span
        elemVertex - array with nb rows and 2 columns, where each row indicates
        the left and right knot of a non-empty knot-span
"""
function makeIEN(knotVector, numElem, p)
    IEN  = Vector{UnitRange{Int16}}(undef,numElem)
    elemVertex = zeros(numElem, 2)
    elementCounter = 0
    for indexKnot = 1:length(knotVector)-1
        if knotVector[indexKnot+1]>knotVector[indexKnot]+eps(eltype(knotVector))
            elementCounter += 1
            IEN[elementCounter] = indexKnot-p:indexKnot
            elemVertex[elementCounter, :] = [knotVector[indexKnot], knotVector[indexKnot+1]]
        end
    end
    @assert numElem==elementCounter "Wrong number of elements passed"
    return IEN, elemVertex
end

"""
Initializes an IGA mesh from a NURBS object
"""
function genMesh(nurbs::NURBS{T}, gauss_rule) where T
    #1D mesh
    knotU = nurbs.knots[1]
    degP = nurbs.order.-1
    Cmat, numElem = bezierExtraction(knotU, degP[1])
    C = Array{AbstractArray{T},1}(undef,numElem)
    for i=1:numElem
        C[i] = SArray{Tuple{degP[1]+1,degP[1]+1}}(Cmat[:,:,i]')
    end
    numBasis = length(knotU) - degP[1] - 1
    IEN, elemVertex = makeIEN(knotU, numElem, degP[1])
    cpts = reshape(nurbs.coefs[1:3,:,:,:], 3, numBasis)
    wgts = reshape(nurbs.coefs[4,:,:,:], numBasis)
    for i=1:3
        cpts[i,:] ./= wgts
    end
    # elemNode = Array{Array{Int64,1},1}(undef, numElem)
    # for i=1:numElem
    #     elemNode[i]=IEN[i,:]
    # end
    B,dB,ddB = bernsteinBasis(gauss_rule.nodes, degP[1])
    N = Array{AbstractArray{T},1}(undef,numElem)
    dN = Array{AbstractArray{T},1}(undef,numElem)
    ddN = Array{AbstractArray{T},1}(undef,numElem)
    Jac = zeros(numElem)
    for iElem = 1:numElem
        Jac[iElem] = (elemVertex[iElem,2]-elemVertex[iElem,1])/T(2.0)
        N[iElem], dN[iElem], ddN[iElem] = B*C[iElem], dB*C[iElem]/Jac[iElem], ddB*C[iElem]/Jac[iElem]^2
    end
    elemVertex = SArray{Tuple{numElem,2}}(elemVertex)
    
    IGAmesh = Mesh(elemVertex, IEN, degP, C, B, dB, ddB, N, dN, ddN, Jac, numBasis, numElem, cpts, wgts, knotU)
    return IGAmesh
end
"""
Plots the basis functions of a mesh in parameter space
"""
function plotBasisParam(mesh::Mesh)
    #1D plot
    colorList = ["blue", "red", "green", "black", "magenta"]
    graph=Plots.plot(title="B-Splines of degree $(mesh.degP[1])")
    numPtsElem = 11
    evalPts = LinRange(-1, 1, numPtsElem)
    B, dB, ddB = bernsteinBasis(evalPts, mesh.degP[1])

    for iBasis = 1:mesh.numBasis
        colorIndex = ((iBasis-1) % length(colorList))+1
        for iElem = 1:mesh.numElem
            localIndex = findall(isequal(iBasis), mesh.elemNode[iElem])
            if length(localIndex)>0
                uMin = mesh.elemVertex[iElem, 1]
                uMax = mesh.elemVertex[iElem, 2]
                plotPts = LinRange(uMin, uMax, numPtsElem)
                plotVal = B*(mesh.C[iElem][localIndex,:])'
                graph = plot!(plotPts, plotVal, color=colorList[colorIndex], leg=false, line=2)
                graph = scatter!([uMin], [0], color="red", leg=false, markersize=5)
                graph = scatter!([uMax], [0], color="red", leg=false, markersize=5)
            end
        end
    end
    display(graph)
end

"""
Plots the computed solution
"""
function plotSol(mesh::Mesh, sol0, exactSol::Function)
    graph=plot(title="Approximate solution")
    numPtsElem = 11
    evalPts = LinRange(-1, 1, numPtsElem)
    B, dB = bernsteinBasis(evalPts, degP[1])

    for iElem in 1:mesh.numElem
        curNodes = mesh.elemNode[iElem]
        uMin = mesh.elemVertex[iElem, 1]
        uMax = mesh.elemVertex[iElem, 2]
        plotPts = LinRange(uMin, uMax, numPtsElem)
        physPts = zeros(numPtsElem)
        splineVal = B*(mesh.C[iElem])
        cpts = mesh.controlPoints[1, curNodes]
        wgts = mesh.weights[curNodes]
        basisVal = zero(splineVal)
        for iPlotPt = 1:numPtsElem
            RR = splineVal[iPlotPt,:].* wgts
            w_sum = sum(RR)
            RR /= w_sum
            basisVal[iPlotPt,:] = RR
            physPts[iPlotPt] = RR'*cpts
        end
        solVal = basisVal*sol0[curNodes]
        graph = plot!(plotPts, solVal, color="blue", leg=false, line=2)
        exSolVal = real(exactSol.(physPts))
        graph = plot!(plotPts, exSolVal, color="green", leg=false, line=2)
        graph = plot!(plotPts, zeros(length(plotPts)), color="black", leg=false, line=1)
        graph = scatter!([uMin], [0], color="red", leg=false, markersize=5)
        graph = scatter!([uMax], [0], color="red", leg=false, markersize=5)
    end
    display(graph)
end

function Jξ(elemVertex::AbstractArray{T}, iElem::Int) where T
    (elemVertex[iElem,2]-elemVertex[iElem,1])/T(2.0)
end

function BSplineBasis(mesh::Mesh{T}, iElem::Int) where T
    Jac_ref_par = Jξ(mesh.elemVertex,iElem)
    N_mat = mesh.B * mesh.C[iElem]
    dN_mat = mesh.dB * mesh.C[iElem]/Jac_ref_par
    ddN_mat = mesh.ddB * mesh.C[iElem]/Jac_ref_par^2
    return Jac_ref_par,N_mat, dN_mat, ddN_mat
end


"""
Plots the error in the approximate solution
"""
function plotSolError(mesh::Mesh, sol0, exactSol::Function)
    graph=plot(title="Error \$u-u_h\$")
    numPtsElem = 11
    evalPts = LinRange(-1, 1, numPtsElem)
    B, dB = bernsteinBasis(evalPts, degP[1])

    for iElem in 1:mesh.numElem
        curNodes = mesh.elemNode[iElem]
        uMin = mesh.elemVertex[iElem, 1]
        uMax = mesh.elemVertex[iElem, 2]
        plotPts = LinRange(uMin, uMax, numPtsElem)
        physPts = zeros(numPtsElem)
        splineVal = B*(mesh.C[iElem])
        cpts = mesh.controlPoints[1, curNodes]
        wgts = mesh.weights[curNodes]
        basisVal = zero(splineVal)
        for iPlotPt = 1:numPtsElem
            RR = splineVal[iPlotPt,:].* wgts
            w_sum = sum(RR)
            RR /= w_sum
            physPts[iPlotPt] = RR'*cpts
            basisVal[iPlotPt,:] = RR
        end
        exSolVal = real(exactSol.(physPts))
        solVal = basisVal*sol0[curNodes]
        graph = plot!(plotPts, exSolVal-solVal, color="blue", leg=false, line=2)
        graph = plot!(plotPts, zeros(length(plotPts)), color="black", leg=false, line=1)
        graph = scatter!([uMin], [0], color="red", leg=false, markersize=5)
        graph = scatter!([uMax], [0], color="red", leg=false, markersize=5)
    end
    display(graph)
end
function getDerivSol(mesh::Mesh, sol0,numPts=mesh.numElem)
    numPtsElem = max(floor(Int, numPts/mesh.numElem)+1,2)
    evalPts = LinRange(-1, 1, numPtsElem)
    B, dB, ddB = bernsteinBasis(evalPts, mesh.degP[1])
    sol = zeros((numPtsElem-1)*mesh.numElem+1)
    is=1; ie=numPtsElem
    for iElem in 1:mesh.numElem
        uMin = mesh.elemVertex[iElem, 1]
        uMax = mesh.elemVertex[iElem, 2]
        Jac_ref_par = (uMax-uMin)/2
        curNodes = mesh.elemNode[iElem]
        N_mat = B*(mesh.C[iElem])
        dN_mat = dB * mesh.C[iElem]/Jac_ref_par
        wgts = mesh.weights[curNodes]
        solVal = zeros(numPtsElem)
        for iPlotPt = 1:numPtsElem
            RR = N_mat[iPlotPt,:].* wgts
            dR = dN_mat[iPlotPt,:].* wgts
            w_sum = sum(RR)
            dw_xi = sum(dR)
            dR = dR/w_sum - RR*dw_xi/w_sum^2
            solVal[iPlotPt] += dR' * sol0[curNodes]
        end
        sol[is:ie] .= solVal
        is += numPtsElem-1 ; ie += numPtsElem-1
    end
    return sol
end

function getSol(mesh::Mesh, sol0, numPts=mesh.numElem)
    numPtsElem = max(floor(Int, numPts/mesh.numElem)+1,2)
    evalPts = LinRange(-1, 1, numPtsElem)
    B, dB, ddB = bernsteinBasis(evalPts, mesh.degP[1])
    sol = zeros((numPtsElem-1)*mesh.numElem+1)
    is=1; ie=numPtsElem
    for iElem in 1:mesh.numElem
        curNodes = mesh.elemNode[iElem]
        splineVal = B*(mesh.C[iElem])
        wgts = mesh.weights[curNodes]
        basisVal = zeros(size(splineVal))
        for iPlotPt = 1:numPtsElem
            RR = splineVal[iPlotPt,:].* wgts
            w_sum = sum(RR)
            RR /= w_sum
            basisVal[iPlotPt,:] = RR
        end
        solVal = basisVal*sol0[curNodes]
        sol[is:ie] .= solVal
        is += numPtsElem-1 ; ie += numPtsElem-1
    end
    return sol
end
function getBasis(mesh::Mesh)
    numPtsElem = 11
    evalPts = LinRange(-1, 1, numPtsElem)
    B, dB, ddB = bernsteinBasis(evalPts, mesh.degP[1])
    R = zeros((numPtsElem-1)*mesh.numElem+1, mesh.numBasis)
    dR = zeros((numPtsElem-1)*mesh.numElem+1, mesh.numBasis)
    for iBasis = 1:mesh.numBasis
        for iElem = 1:mesh.numElem
            localIndex = findall(isequal(iBasis), mesh.elemNode[iElem])
            if length(localIndex)>0
                plotVal = B*(mesh.C[iElem][localIndex,:])
                R[(iElem-1)*(numPtsElem-1)+1:iElem*(numPtsElem-1)+1, iBasis] .= plotVal
                plotVal = dB*(mesh.C[iElem][localIndex,:])
                dR[(iElem-1)*(numPtsElem-1)+1:iElem*(numPtsElem-1)+1, iBasis] .= plotVal
            end
        end
    end
    return R,dR
end

function getBasis(mesh::Mesh, x::Float64)
    iElem = argmin(abs.(sum(mesh.elemVertex.-x, dims=2))[:])
    points = mesh.elemVertex[iElem,:]
    evalPts = [(x-points[1])/(points[2]-points[1])*2-1]
    B, dB, ddB = bernsteinBasis(evalPts, mesh.degP[1])
    R = zeros(mesh.numBasis)
    dR = zeros(mesh.numBasis)
    for iBasis = 1:mesh.numBasis
        localIndex = findall(isequal(iBasis), mesh.elemNode[iElem])
        if length(localIndex)>0
            plotVal = B*mesh.C[iElem][localIndex,:]'
            R[iBasis] = plotVal[1]
            plotVal = dB*mesh.C[iElem][localIndex,:]'
            dR[iBasis] = plotVal[1]
        end
    end
    return R,dR
end
