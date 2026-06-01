#----- pkg

using Serialization
using JLD2
using NPZ
using LinearAlgebra
using SpecialFunctions
using ProgressMeter
using LoopVectorization

#####

function lip(degree::Int64, x::Float64, xs::Array{Float64}, ys::Array{Float64})::Float64
    # Lagrange quadratic interpolation

    lx = length(xs)
    ly = length(ys)

    @assert lx == ly "xs and ys must have same length"

    P = 0
    @inbounds @fastmath for j in 1:degree+1
        lj = 1.0
        for m in 1:degree+1
            if m == j
                continue
            end
            D = xs[j] - xs[m]
            lj = lj*(x-xs[m])/D
         end
         P = P + ys[j]*lj
    end

    return P
end
