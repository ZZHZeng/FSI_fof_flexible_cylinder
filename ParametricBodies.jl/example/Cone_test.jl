using WaterLily
using StaticArrays

function make_sim()
    function sdCone(x::SVector{N,T},c,h)
        # c = SA[sin(α),cos(α)], h = height
        q = SA[h*c[1]/c[2],-h]
        w = SA[norm(SA[x[1],x[3]]), x[2]]
        a = w - q.*clamp( dot(w,q)/dot(q,q), 0.0, 1.0 )
        b = w - q*SA[clamp( w[1]/q[1], 0.0, 1.0 ), 1.0 ]
        k = sign(q[2])
        d = min(dot( a, a ),dot(b, b))
        s = max( k*(w[1]*q[2]-w[2]*q[1]),k*(w[2]-q[2])  )
        return sqrt(d)*sign(s)
    end
    
    body = AutoBody((x,t)->sdCone(x,s,h))

    return Simulation()
end