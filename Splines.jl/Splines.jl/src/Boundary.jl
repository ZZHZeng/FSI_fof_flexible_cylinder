# Boundary module
struct Boundary1D{T}
    type   :: String #can be "Dirichlet, Neumann or Robin"
    x_val  :: T #boundary point in the physical space
    u_val  :: T #boundary point in the parameter space
    op_val :: T #boundary operator value
    comp   :: Int16 #component of the variable where the BC is applied
    function Boundary1D(type::String, x_val::T, u_val::T, op_val::T; comp=1) where T
        new{T}(type, x_val, u_val,  op_val, comp)
    end
end
function Boundary1D(type::String, x_val::T, op_val::T; comp=1) where T
    return Boundary1D(type, x_val, x_val, op_val; comp=comp)
end