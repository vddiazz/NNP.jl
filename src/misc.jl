#----- pkg

using LinearAlgebra
using JLD2
using NPZ
using ProgressMeter
using LoopVectorization

#### baryon number

function b_dens(y1::Array{Float64},y2::Array{Float64},y3::Array{Float64},model::String,grid_size::String,r_idx::Int64,Q_idx::Int64,out::String, output_format::String)

    #----- read data

    U = open("/home/velni/phd/w/nnp/data/prod/$(model)/$(grid_size)/U_sym_r=$(r_idx)_Q=$(Q_idx).jls", "r") do io; deserialize(io); end
    d1U = open("/home/velni/phd/w/nnp/data/deriv/$(model)/$(grid_size)/d1U_r=$(r_idx)_Q=$(Q_idx).jls", "r") do io; deserialize(io); end
    d2U = open("/home/velni/phd/w/nnp/data/deriv/$(model)/$(grid_size)/d2U_r=$(r_idx)_Q=$(Q_idx).jls", "r") do io; deserialize(io); end
    d3U = open("/home/velni/phd/w/nnp/data/deriv/$(model)/$(grid_size)/d3U_r=$(r_idx)_Q=$(Q_idx).jls", "r") do io; deserialize(io); end

    l1 = length(U[:,1,1,1]); l2 = length(U[1,:,1,1]); l3 = length(U[1,1,:,1])

    #----- main
    
    println()
    println("#--------------------------------------------------#")
    println()
    println("Baryon density")
    println()

    dens = zeros(Float64, l1,l2,l3)

    @showprogress 1 "Computing..." for k in 1:l3
        @tturbo for j in 1:l2, i in 1:l1
            dens[i,j,k] = -12*(U[i,j,k,1]^2 + U[i,j,k,2]^2 + U[i,j,k,3]^2 + U[i,j,k,4]^2)*(d1U[i,j,k,2]*d2U[i,j,k,4]*d3U[i,j,k,3]*U[i,j,k,1] - d1U[i,j,k,2]*d2U[i,j,k,3]*d3U[i,j,k,4]*U[i,j,k,1] - d1U[i,j,k,1]*d2U[i,j,k,4]*d3U[i,j,k,3]*U[i,j,k,2] + d1U[i,j,k,1]*d2U[i,j,k,3]*d3U[i,j,k,4]*U[i,j,k,2] - d1U[i,j,k,2]*d2U[i,j,k,4]*d3U[i,j,k,1]*U[i,j,k,3] + d1U[i,j,k,1]*d2U[i,j,k,4]*d3U[i,j,k,2]*U[i,j,k,3] + d1U[i,j,k,2]*d2U[i,j,k,1]*d3U[i,j,k,4]*U[i,j,k,3] - d1U[i,j,k,1]*d2U[i,j,k,2]*d3U[i,j,k,4]*U[i,j,k,3] + d1U[i,j,k,4]*(d3U[i,j,k,3]*(-(d2U[i,j,k,2]*U[i,j,k,1]) + d2U[i,j,k,1]*U[i,j,k,2]) + d2U[i,j,k,3]*(d3U[i,j,k,2]*U[i,j,k,1] - d3U[i,j,k,1]*U[i,j,k,2]) + (d2U[i,j,k,2]*d3U[i,j,k,1] - d2U[i,j,k,1]*d3U[i,j,k,2])*U[i,j,k,3]) + (d1U[i,j,k,2]*(d2U[i,j,k,3]*d3U[i,j,k,1] - d2U[i,j,k,1]*d3U[i,j,k,3]) + d1U[i,j,k,1]*(-(d2U[i,j,k,3]*d3U[i,j,k,2]) + d2U[i,j,k,2]*d3U[i,j,k,3]))*U[i,j,k,4] + d1U[i,j,k,3]*(d3U[i,j,k,4]*(d2U[i,j,k,2]*U[i,j,k,1] - d2U[i,j,k,1]*U[i,j,k,2]) + d2U[i,j,k,4]*(-(d3U[i,j,k,2]*U[i,j,k,1]) + d3U[i,j,k,1]*U[i,j,k,2]) + (-(d2U[i,j,k,2]*d3U[i,j,k,1]) + d2U[i,j,k,1]*d3U[i,j,k,2])*U[i,j,k,4]))
        end
    end

    #----- data saving
    
    if output_format == "jld2"
        path = out*"/bdens.jld2"
        @save path dens

    elseif output_format == "npy"
        npzwrite(out*"/bdens.npy", dens)

    elseif output_format == "jls"
        open(out*"/bdens_r=$(r_idx)_Q=$(Q_idx).jls", "w") do io; serialize(io, dens); end
    end
    
    println()
    println("data saved at "*out )
    println()
    println("#--------------------------------------------------#")

    return dens

end

function b_num(dens::Array{Float64})::Float64
    
    l1 = size(dens[:,1,1]); l2 = size(dens[1,:,1]); l3 = size(dens[1,1,:]); 

    dy = 0.1    

    #----- computations

    B = (-1/(24*pi^2))*sum(dens)*(dy^3)

    #----- output

    return B

end



