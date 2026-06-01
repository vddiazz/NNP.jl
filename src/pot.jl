#----- pkg
 
using Serialization
using JLD2
using NPZ
using ProgressMeter
 
###
 
function pot(m::Float64,c6::Float64,r_idx::Int64,Q_idx::Int64,grid_size::String,hD::Float64,model::String,pot_terms::Vector{<:Function},out::String,output_format::String)::Float64

    #----- prepare fields
    
    u = open("/home/velni/phd/w/nnp/data/prod/$(model)/$(grid_size)/U_sym_r=$(r_idx)_Q=$(Q_idx).jls", "r") do io; deserialize(io); end
    d1 = open("/home/velni/phd/w/nnp/data/deriv/$(model)/$(grid_size)/d1U_r=$(r_idx)_Q=$(Q_idx).jls", "r") do io; deserialize(io); end
    d2 = open("/home/velni/phd/w/nnp/data/deriv/$(model)/$(grid_size)/d2U_r=$(r_idx)_Q=$(Q_idx).jls", "r") do io; deserialize(io); end
    d3 = open("/home/velni/phd/w/nnp/data/deriv/$(model)/$(grid_size)/d3U_r=$(r_idx)_Q=$(Q_idx).jls", "r") do io; deserialize(io); end

    l1 = length(u[:,1,1,1]); l2 = length(u[1,:,1,1]); l3 = length(u[1,1,:,1])

    #----- main

    println()
    println("#--------------------------------------------------#")
    println()
    println("Potential --- r_idx=$(r_idx), Q_idx=$(Q_idx)")
    println()

    v = 0.

    if model == "std"
        v1 = pot_terms[1]
        v2 = pot_terms[2]
        v3 = pot_terms[3]

        @showprogress 1 "Computing..." for k in 1:l3
            @inbounds @fastmath for j in 1:l2, i in 1:l1
                V1 = v1(i,j,k,u,d1,d2,d3)
                V2 = v2(i,j,k,u,d1,d2,d3)
                V3 = v3(i,j,k,u,d1,d2,d3)

                v = v + (V1 + 0.5*(V2 - V3))
            end
        end
    
    elseif model == "mss"
        v1 = pot_terms[1]
        v2 = pot_terms[2]
        v3 = pot_terms[3]
        v6 = pot_terms[4]

        @showprogress 1 "Computing..." for k in 1:l3
            @inbounds @fastmath for j in 1:l2, i in 1:l1
                V1 = v1(i,j,k,u,d1,d2,d3)
                V2 = v2(i,j,k,u,d1,d2,d3)
                V3 = v3(i,j,k,u,d1,d2,d3)

                v = v + (V1 + 0.5*(V2 - V3) + 2*(m^2)*(u[i,j,k,1] - 1))
            end
        end
    end

    return v*hD^3

end
