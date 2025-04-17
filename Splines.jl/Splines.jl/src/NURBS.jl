# Spline utils
struct NURBS{T}
    number :: Array{Int16,1}
    coefs  :: Array{T}
    knots  :: Array{Array{T,1}}
    order  :: Array{Int16,1}
    function NURBS(number, coefs::Array{T}, knots::Array{Array{T,1}}, order) where T
        new{T}(number, coefs, knots, order)
    end
end

function nrbmak(coefs, knots)
    np = [i for i in size(coefs)]
    dim = np[1]

    #constructing a curve
    if length(np)==1
        number = [np[1]]
    else
        number = [np[2]]
    end
    if dim<4
        temp_coefs = coefs
        coefs = repeat([0., 0., 0., 1.], 1, number[1])
        coefs[1:dim,:,:,:] = temp_coefs
    end
    order = [length(knots[1])-number[1]]
    uknots = sort(knots[1])
    knots = [uknots]
    nrb = NURBS(number, coefs, knots, order)
    return nrb
end

function findspan(n, u, knot::Array)
    if (minimum(u)<knot[1]) || (maximum(u)>knot[end])
        error("Some value is outside the knot span")
    end
    if isa(u, Number)
        s = [0]
    else
        # s = similar(u, Int64)
    end
    for j=1:length(u)
        if u[j]==knot[n+2]
            s[j]=n
            continue
        end
        s[j]=findlast(knot.<=u[j])-1
    end
    return s
end

function nrbline(points...)
    numPts = length(points)
    if numPts==1
        error("More than one point should be specfified")
    end
    coefs = [zeros(3,numPts); ones(1,numPts)]
    for iPt = 1:numPts
        coefs[1:length(points[iPt]),iPt] .= points[iPt]
    end
    knots = vcat(0, LinRange(0,1,numPts), 1)
    line = nrbmak(coefs, [knots])
    return line
end

function bspkntins(d, c, k, u)
    tolEq = 1e-10
    mc, nc = size(c)
    sort!(u)
    nu = length(u)
    nk = length(k)

    ic = zeros(mc, nc+nu)
    ik = zeros(nk+nu)

    n = nc - 1
    r = nu - 1

    m = n + d + 1
    a = findspan(n, u[1], k)[1]
    b = findspan(n, u[r+1], k)[1]
    b += 1

    ic[:, 1:a-d+1] = c[:, 1:a-d+1]
    ic[:, b+nu:nc+nu] = c[:, b:nc]

    ik[1:a+1] = k[1:a+1]
    ik[b+d+nu+1:m+nu+1] = k[b+d+1:m+1]

    ii = b + d - 1
    ss = ii + nu

    for jj = r:-1:0
        ind = (a+1):ii
        ind = ind[u[jj+1].<=k[ind.+1]]
        ic[:, ind.+ss.-ii.-d] = c[:,ind.-d]
        ik[ind.+ss.-ii.+1] = k[ind.+1]
        ii = ii - length(ind)
        ss = ss - length(ind)

        ic[:,ss-d] = ic[:, ss-d+1]
        for l=1:d
            ind = ss - d + l
            alfa = ik[ss+l+1] - u[jj+1]
            if abs(alfa) < tolEq
                ic[:,ind] = ic[:, ind+1]
            else
                alfa = alfa/(ik[ss+l+1]-k[ii-d+l+1])
                tmp = (1-alfa) * ic[:, ind+1]
                ic[:,ind] = alfa*ic[:,ind] + tmp
            end
        end
        ik[ss+1] = u[jj+1]
        ss = ss-1
    end
    return ic, ik
end

function nrbkntins(nurbs::NURBS, iknots::AbstractArray)
    degree = nurbs.order .- 1
    fmax(x,y) = any(y.>maximum(x))
    fmin(x,y) = any(y.<minimum(x))
    for i=1:length(nurbs.knots)
        if any(fmax(nurbs.knots[i], iknots[i])) || any(fmin(nurbs.knots[i], iknots[i]))
            error("Trying to insert a knot outside the interval of definition")
        end
    end
    knots = Array{Array}(undef, length(nurbs.knots))
    # NURBS represents a curve
    if isempty(iknots[1])
        coefs = nurbs.coefs
        knots = nurbs.knots
    else
        coefs, knots[1] = bspkntins(degree[1], nurbs.coefs, nurbs.knots[1], iknots[1])
    end

    #construct the new NURBS
    inurbs = nrbmak(coefs, knots)
    return inurbs
end

function bspdegelev(d, c, k, t)
    mc, nc = size(c)
    ic = zeros(mc, nc*(t+1))
    n = nc - 1
    bezalfs = zeros(d+1, d+t+1)
    bpts = zeros(mc, d+1)
    ebpts = zeros(mc, d+t+1)
    Nextbpts = zeros(mc, d+1)
    alfs = zeros(d)

    m = n + d + 1
    ph = d + t
    ph2 = floor(Int, ph/2)
    # Compute Bezier degree elevation coefficients
    bezalfs[1,1] = 1
    bezalfs[d+1, ph+1] = 1

    for i=1:ph2
        inv = 1/binomial(ph, i)
        mpi = min(d,i)

        for j=max(0, i-t):mpi
            bezalfs[j+1, i+1] = inv*binomial(d,j)*binomial(t, i-j)
        end
    end

    for i=ph2+1:ph-1
        mpi = min(d,i)
        for j=max(0,i-t):mpi
            bezalfs[j+1, i+1] = bezalfs[d-j+1, ph-i+1]
        end
    end

    mh = ph
    kind = ph+1
    r = -1
    a = d
    b = d + 1
    cind = 1
    ua = k[1]

    ic[1:mc,1] = c[1:mc,1]
    ik = ua.*ones(ph+1)

    # Initialize the first Bezier segment
    #bpts[1:mc, 1:d+1] = c[1:mc, 1:d+1]
    bpts = copy(c)

    # Big loop through knot vector
    while b<m
        i=b
        while b<m && k[b+1] == k[b+2]
            b += 1
        end
        mul = b - i + 1
        mh = mh + mul + t
        ub = k[b+1]
        oldr = r
        r = d - mul

        # Insert knot u[b] r times
        if oldr > 0
            lbz = floor(Int, (oldr+2)/2)
        else
            lbz = 1
        end

        if r>0
            rbz = ph - floor(Int, (r+1)/2)
        else
            rbz = ph
        end

        if r>0
            # Insert knot to get Bezier segment
            numer = ub - ua
            for q = d:-1:mul+1
                alfs[q-mul] = numer / (k[a+q+1]-ua)
            end

            for j = 1:r
                save = r-j
                s = mul + j

                for q = d:-1:s
                    for ii = 0:mc-1
                        tmp1 = alfs[q-s+1]*bpts[ii+1, q+1]
                        tmp2 = (1-alfs[q-s+1])*bpts[ii+1, q]
                        bpts[ii+1, q+1] = tmp1 + tmp2
                    end
                end

                Nextbpts[:, save+1] = bpts[:, d+1]
            end
        end
        # End of insert knot

        #Degree elevate Bezier
        for i=lbz:ph
            ebpts[:,i+1] = zeros(mc)
            mpi = min(d,i)
            for j=max(0,i-t):mpi
                for ii=0:mc-1
                    tmp1 = ebpts[ii+1, i+1]
                    tmp2 = bezalfs[j+1, i+1]*bpts[ii+1, j+1]
                    ebpts[ii+1, i+1] = tmp1 + tmp2
                end
            end
        end
        # End of degree elevating Bezier

        if oldr > 1
            # Must remove knot u=k[a] oldr times
            first = kind - 2
            last = kind
            den = ub - ua
            bet = floor(Int, (ub-ik[kind])/den)

            # Knot removal loop
            for tr = 1:oldr-1
                i = first
                j = last
                kj = j - kind + 1
                while j-i > tr
                    # Loop and compute the new control points for one removal step
                    if i < cind
                        alf = (ub - ik[i+1])/(ua - ik[i+1])
                        tmp1 = alf.*ic[:,i+1]
                        tmp2 = (1-alf).*ic[:, i]
                        ic[:,i+1] = tmp1 + tmp2
                    end
                    if j >= lbz
                        if j-tr <= kind - ph + oldr
                            gam = (ub-ik[j-tr+1])/den
                            tmp1 = gam.*ebpts[:, kj+1]
                            tmp2 = (1-gam).*ebpts[:, kj+2]
                            ebpts[:, kj+1] = tmp1 + tmp2
                        else
                            tmp1 = bet.*ebpts[:, kj+1]
                            tmp2 = (1-bet).*ebpts[:, kj+2]
                            ebpts[:, kj+1] = tmp1 + tmp2
                        end
                    end
                    i += 1
                    j -= 1
                    kj -= 1
                end
                first -= 1
                last += 1
            end
        end
        # End of removing knot n=k[a]

        # Load the knot ua
        if a!=d
            for i=0:ph-oldr-1
                push!(ik, ua)
                #ik[kind+1] = ua
                kind += 1
            end
        end

        for j = lbz:rbz
            for ii = 0:mc-1
                ic[ii+1, cind+1] = ebpts[ii+1, j+1]
            end
            cind += 1
        end

        if b < m
            # Setup for next pass through loop
            bpts[:,1:r] = Nextbpts[:, 1:r]
            bpts[:,r+1:d+1] = c[:, b-d+r+1:b+1]
            a = b
            b += 1
            ua = ub
        else
            for i=0:ph
                #ik[kind+i+1] = ub
                push!(ik, ub)
            end
        end
    end
    #End big while loop
    ic = ic[:, 1:cind]
    return ic, ik
end

function nrbdegelev(nurbs::NURBS, ntimes::Array)
    degree = nurbs.order .- 1
    knots = Array{Array}(undef, length(nurbs.knots))
    # NURBS represents a curve
    if isempty(ntimes)
        coefs = nurbs.coefs
        knots = nurbs.knots
    else
        coefs, knots[1] = bspdegelev(degree[1], nurbs.coefs, nurbs.knots[1], ntimes[1])
    end
    #construct new NURBS
    inurbs = nrbmak(coefs, knots)
    return inurbs
end
