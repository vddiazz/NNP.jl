#----- pkg

using Serialization
using JLD2
using NPZ
using LinearAlgebra
using ProgressMeter
using LoopVectorization

###

function g_AB(A::Int64,B::Int64,c6::Float64,r_idx::Int64,Q_idx::Int64,grid_size::String,hD::Float64,model::String,metric_terms::Vector{<:Function},out::String,output_format::String)::Float64

    #----- prepare fields

    d1 = open("/home/velni/phd/w/nnp/data/deriv/$(model)/$(grid_size)/d1U_r=$(r_idx)_Q=$(Q_idx).jls", "r") do io; deserialize(io); end
    d2 = open("/home/velni/phd/w/nnp/data/deriv/$(model)/$(grid_size)/d2U_r=$(r_idx)_Q=$(Q_idx).jls", "r") do io; deserialize(io); end
    d3 = open("/home/velni/phd/w/nnp/data/deriv/$(model)/$(grid_size)/d3U_r=$(r_idx)_Q=$(Q_idx).jls", "r") do io; deserialize(io); end

    l1 = length(d1[:,1,1,1]); l2 = length(d1[1,:,1,1]); l3 = length(d1[1,1,:,1])

    #----- main

    println()
    println("#--------------------------------------------------#")
    println()
    println("Metric (A=$(A), B=$(B)) --- r_idx=$(r_idx), Q_idx=$(Q_idx)")
    println()
 
    DA = open("/home/velni/phd/w/nnp/data/deriv/$(model)/$(grid_size)/D$(A)U_r=$(r_idx)_Q=$(Q_idx).jls", "r") do io; deserialize(io); end
    DB = open("/home/velni/phd/w/nnp/data/deriv/$(model)/$(grid_size)/D$(B)U_r=$(r_idx)_Q=$(Q_idx).jls", "r") do io; deserialize(io); end

    g = 0.

    if model == "std"
        g1 = metric_terms[1]
        g2 = metric_terms[2]
        g3 = metric_terms[3]

        @showprogress 1 "Computing..." for k in 1:l3
            @inbounds @fastmath for j in 1:l2, i in 1:l1
                G1 = g1(i,j,k,DA,DB,d1,d2,d3)
                G2 = g2(i,j,k,DA,DB,d1,d2,d3)
                G3 = g3(i,j,k,DA,DB,d1,d2,d3)

                g = g + (2*(G1 + G2 - G3))
            end
        end

    elseif model == "gen"
        g1 = metric_terms[1]
        g2 = metric_terms[2]
        g3 = metric_terms[3]
        g4 = metric_terms[4]
        g5 = metric_terms[5]
        g6 = metric_terms[6]
        g7 = metric_terms[7]

        @showprogress 1 "Computing" for k in 1:l3
            @inbounds @fastmath for j in 1:l2, i in 1:l1
                G1 = g1(i,j,k,DA,DB,d1,d2,d3)
                G2 = g2(i,j,k,DA,DB,d1,d2,d3)
                G3 = g3(i,j,k,DA,DB,d1,d2,d3)
                G4 = g4(i,j,k,DA,DB,d1,d2,d3)
                G5 = g5(i,j,k,DA,DB,d1,d2,d3)
                G6 = g6(i,j,k,DA,DB,d1,d2,d3)
                G7 = g7(i,j,k,DA,DB,d1,d2,d3)

                g = g + (2*(G1 + G2 - G3 + c6*G4 - c6*G5 + (c6/2.)*G6 - (c6/2.)*G7) )
            end
        end
    end

    return g*hD^3
end

function g_AB_extra(A::Int64,B::Int64,factor::Float64,c6::Float64,r_idx::Int64,Q_idx::Int64,grid_size::String,hD::Float64,model::String,metric_terms::Vector{<:Function},out::String,output_format::String)::Float64

    #----- prepare fields

    d1_old = open("/home/velni/phd/w/nnp/data/deriv/$(model)/$(grid_size)/d1U_r=$(r_idx)_Q=$(Q_idx).jls", "r") do io; deserialize(io); end
    d2_old = open("/home/velni/phd/w/nnp/data/deriv/$(model)/$(grid_size)/d2U_r=$(r_idx)_Q=$(Q_idx).jls", "r") do io; deserialize(io); end
    d3_old = open("/home/velni/phd/w/nnp/data/deriv/$(model)/$(grid_size)/d3U_r=$(r_idx)_Q=$(Q_idx).jls", "r") do io; deserialize(io); end

    p1 = length(d1_old[:,1,1,1]); p2 = length(d1_old[1,:,1,1]); p3 = length(d1_old[1,1,:,1])

	DA_old = open("/home/velni/phd/w/nnp/data/deriv/$(model)/$(grid_size)/D$(A)U_r=$(r_idx)_Q=$(Q_idx).jls", "r") do io; deserialize(io); end
    DB_old = open("/home/velni/phd/w/nnp/data/deriv/$(model)/$(grid_size)/D$(B)U_r=$(r_idx)_Q=$(Q_idx).jls", "r") do io; deserialize(io); end

	#----- extrapolation (just add 0's in all directions)

	#p1e = round(Int64,factor*p1); p2e = round(Int64,factor*p2); p3e = round(Int64,factor*p3)

	#d1 = zeros(Float64, p1e,p2e,p3e,4)
	#d2 = zeros(Float64, p1e,p2e,p3e,4)
	#d3 = zeros(Float64, p1e,p2e,p3e,4)
	#DA = zeros(Float64, p1e,p2e,p3e,4)
	#DB = zeros(Float64, p1e,p2e,p3e,4)

	#l1 = p1*hD - 1; l2 = p2*hD - 1; l3 = p3*hD - 1
	#l1e = p1e*hD - 1; l2e = p2e*hD - 1; l3e = p3e*hD - 1

	#x1 = collect(-l1/2:hD:l1/2); x2 = collect(-l2/2:hD:l2/2); x3 = collect(-l3/2:hD:l3/2)
	#x1e = collect(-l1e/2:hD:l1e/2); x2e = collect(-l2e/2:hD:l2e/2); x3e = collect(-l3e/2:hD:l3e/2)

	#i1_old = findfirst(==(0.0),x1); i2_old = findfirst(==(0.0),x2); i3_old = findfirst(==(0.0),x3) 
	#i1 = findfirst(==(0.0),x1e); i2 = findfirst(==(0.0),x2e); i3 = findfirst(==(0.0),x3e)

	#d1[(i1 - i1_old + 1):(i1 - i1_old + p1), (i2 - i2_old + 1):(i2 - i2_old + p2),(i3 - i3_old + 1):(i3 - i3_old + p3),:] .= d1_old
	#d2[(i1 - i1_old + 1):(i1 - i1_old + p1), (i2 - i2_old + 1):(i2 - i2_old + p2),(i3 - i3_old + 1):(i3 - i3_old + p3),:] .= d2_old
	#d3[(i1 - i1_old + 1):(i1 - i1_old + p1), (i2 - i2_old + 1):(i2 - i2_old + p2),(i3 - i3_old + 1):(i3 - i3_old + p3),:] .= d3_old
	#DA[(i1 - i1_old + 1):(i1 - i1_old + p1), (i2 - i2_old + 1):(i2 - i2_old + p2),(i3 - i3_old + 1):(i3 - i3_old + p3),:] .= DA_old
	#DB[(i1 - i1_old + 1):(i1 - i1_old + p1), (i2 - i2_old + 1):(i2 - i2_old + p2),(i3 - i3_old + 1):(i3 - i3_old + p3),:] .= DB_old

	#----- extrapolation (just add 0's)
	
	p1e = round(Int64,factor*p1); p2e = round(Int64,factor*p2); p3e = round(Int64,factor*p3)

	d1 = zeros(Float64, p1e,p2e,p3e,4)
	d2 = zeros(Float64, p1e,p2e,p3e,4)
	d3 = zeros(Float64, p1e,p2e,p3e,4)
	DA = zeros(Float64, p1e,p2e,p3e,4)
	DB = zeros(Float64, p1e,p2e,p3e,4)

	@tturbo for c in 1:4, i3 in 1:p3, i2 in 1:p2, i1 in 1:p1
		d1[i1,i2,i3,c] = d1_old[i1,i2,i3,c]
		d2[i1,i2,i3,c] = d2_old[i1,i2,i3,c]
		d3[i1,i2,i3,c] = d3_old[i1,i2,i3,c]
		DA[i1,i2,i3,c] = DA_old[i1,i2,i3,c]
		DB[i1,i2,i3,c] = DB_old[i1,i2,i3,c]
	end

    #----- main

    println()
    println("#--------------------------------------------------#")
    println()
    println("Metric (A=$(A), B=$(B)) --- r_idx=$(r_idx), Q_idx=$(Q_idx)")
    println()
 
    g = 0.

    if model == "std"
        g1 = metric_terms[1]
        g2 = metric_terms[2]
        g3 = metric_terms[3]

        @showprogress 1 "Computing..." for k in 1:p3e
            @inbounds @fastmath for j in 1:p2e, i in 1:p1e
                G1 = g1(i,j,k,DA,DB,d1,d2,d3)
                G2 = g2(i,j,k,DA,DB,d1,d2,d3)
                G3 = g3(i,j,k,DA,DB,d1,d2,d3)

                g = g + (2*(G1 + G2 - G3))
            end
        end

    elseif model == "gen"
        g1 = metric_terms[1]
        g2 = metric_terms[2]
        g3 = metric_terms[3]
        g4 = metric_terms[4]
        g5 = metric_terms[5]
        g6 = metric_terms[6]
        g7 = metric_terms[7]

        @showprogress 1 "Computing" for k in 1:p3e
            @inbounds @fastmath for j in 1:p2e, i in 1:p1e
                G1 = g1(i,j,k,DA,DB,d1,d2,d3)
                G2 = g2(i,j,k,DA,DB,d1,d2,d3)
                G3 = g3(i,j,k,DA,DB,d1,d2,d3)
                G4 = g4(i,j,k,DA,DB,d1,d2,d3)
                G5 = g5(i,j,k,DA,DB,d1,d2,d3)
                G6 = g6(i,j,k,DA,DB,d1,d2,d3)
                G7 = g7(i,j,k,DA,DB,d1,d2,d3)

                g = g + (2*(G1 + G2 - G3 + c6*G4 - c6*G5 + (c6/2.)*G6 - (c6/2.)*G7) )
            end
        end
    end

    return g*hD^3
end

function g_AB_proy(A::Int64,B::Int64,c6::Float64,r_idx::Int64,Q_idx::Int64,grid_size::String,model::String,metric_terms::Vector{<:Function},out::String,output_format::String)::Float64

    #----- prepare fields

    dy1 = open("/lustre/HQCD/victor.diaz/nnp/data/sample/$(grid_size)/dy1.jls", "r") do io; deserialize(io); end
    dy2 = open("/lustre/HQCD/victor.diaz/nnp/data/sample/$(grid_size)/dy2.jls", "r") do io; deserialize(io); end
    dy3_all = open("/lustre/HQCD/victor.diaz/nnp/data/sample/$(grid_size)/dy3.jls", "r") do io; deserialize(io); end
    dy3 = dy3_all[r_idx,:]

    d1 = open("/lustre/HQCD/victor.diaz/nnp/data/deriv/$(model)/$(grid_size)/d1U_r=$(r_idx)_Q=$(Q_idx).jls", "r") do io; deserialize(io); end
    d2 = open("/lustre/HQCD/victor.diaz/nnp/data/deriv/$(model)/$(grid_size)/d2U_r=$(r_idx)_Q=$(Q_idx).jls", "r") do io; deserialize(io); end
    d3 = open("/lustre/HQCD/victor.diaz/nnp/data/deriv/$(model)/$(grid_size)/d3U_r=$(r_idx)_Q=$(Q_idx).jls", "r") do io; deserialize(io); end

    l1 = length(d1[:,1,1,1]); l2 = length(d1[1,:,1,1]); l3 = length(d1[1,1,:,1])

    #----- main

    println()
    println("#--------------------------------------------------#")
    println()
    println("Metric (A=$(A), B=$(B)) --- r_idx=$(r_idx), Q_idx=$(Q_idx)")
    println()
 
    DA = open("/lustre/HQCD/victor.diaz/nnp/data/deriv/$(model)/$(grid_size)/D$(A)U_r=$(r_idx)_Q=$(Q_idx).jls", "r") do io; deserialize(io); end
    DB = open("/lustre/HQCD/victor.diaz/nnp/data/deriv/$(model)/$(grid_size)/D$(B)U_r=$(r_idx)_Q=$(Q_idx).jls", "r") do io; deserialize(io); end

    g = 0.

    if model == "std"
        g1 = metric_terms[1]
        g2 = metric_terms[2]
        g3 = metric_terms[3]

        @showprogress 1 "Computing..." for k in 1:l3
            @inbounds @fastmath for j in 1:l2, i in 1:l1
                G1 = g1(i,j,k,DA,DB,d1,d2,d3)
                G2 = g2(i,j,k,DA,DB,d1,d2,d3)
                G3 = g3(i,j,k,DA,DB,d1,d2,d3)

                g = g + (2*(G1 + G2 - G3))*dy1[i]*dy2[j]*dy3[k]
            end
        end

    elseif model == "gen"
        g1 = metric_terms[1]
        g2 = metric_terms[2]
        g3 = metric_terms[3]
        g4 = metric_terms[4]
        g5 = metric_terms[5]
        g6 = metric_terms[6]
        g7 = metric_terms[7]

        @showprogress 1 "Computing" for k in 1:l3
            @inbounds @fastmath for j in 1:l2, i in 1:l1
                G1 = g1(i,j,k,DA,DB,d1,d2,d3)
                G2 = g2(i,j,k,DA,DB,d1,d2,d3)
                G3 = g3(i,j,k,DA,DB,d1,d2,d3)
                G4 = g4(i,j,k,DA,DB,d1,d2,d3)
                G5 = g5(i,j,k,DA,DB,d1,d2,d3)
                G6 = g6(i,j,k,DA,DB,d1,d2,d3)
                G7 = g7(i,j,k,DA,DB,d1,d2,d3)

                g = g + (2*(G1 + G2 - G3 + c6*G4 - c6*G5 + (c6/2.)*G6 - (c6/2.)*G7)*dy1[i]*dy2[j]*dy3[k] )
            end
        end
    end

    return g
end
