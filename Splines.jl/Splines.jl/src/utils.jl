"""
    returns the location of the integration points in physical coordinates
"""
function uv_integration(op::AbstractFEOperator)
    B, dB, ddB = bernsteinBasis(op.gauss_rule.nodes, op.mesh.degP[1])
    uv = []
    for iElem = 1:op.mesh.numElem
        #compute the (B-)spline basis functions and derivatives with Bezier extraction
        N_mat = B * op.mesh.C[iElem]'

        #compute the rational spline basis
        curNodes = op.mesh.elemNode[iElem]
        cpts = op.mesh.controlPoints[1, curNodes]
        wgts = op.mesh.weights[curNodes]
        for iGauss = 1:length(op.gauss_rule.nodes)
            #compute the rational basis
            RR = N_mat[iGauss,:].* wgts
            w_sum = sum(RR)
            RR /= w_sum

            # external force at physical point
            phys_pt = RR'*cpts
            push!(uv, phys_pt)
        end
    end
    return uv
end
# reshape to displacements
points(a) = dⁿ(a)
function dⁿ(op::AbstractFEOperator)
    return reshape(op.u[1][1:2op.mesh.numBasis],(op.mesh.numBasis,2))'
end
function vⁿ(op::AbstractFEOperator)
    return reshape(op.u[2][1:2op.mesh.numBasis],(op.mesh.numBasis,2))'
end