using FastGaussQuadrature
struct GaussQuad{T}
    nodes   :: Array{T,1}
    weights :: Array{T,1}
    function GaussQuad(nodes::Array{T,1}, weights::Array{T,1}) where T
        new{T}(nodes, weights)
    end
end

"""
Generates the Gauss-Legendre points of order n
"""
function genGaussLegendre(n)
    nodes, weights = gausslegendre(n)
    gauss_rule = GaussQuad(nodes, weights)
    return gauss_rule
end