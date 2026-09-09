#----- pkg

using Serialization
using JLD2
using NPZ
using LinearAlgebra
using ProgressMeter
using LoopVectorization
using Interpolations

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
	DA_old = open("/home/velni/phd/w/nnp/data/deriv/$(model)/$(grid_size)/D$(A)U_r=$(r_idx)_Q=$(Q_idx).jls", "r") do io; deserialize(io); end
    DB_old = open("/home/velni/phd/w/nnp/data/deriv/$(model)/$(grid_size)/D$(B)U_r=$(r_idx)_Q=$(Q_idx).jls", "r") do io; deserialize(io); end

	l1,l2,l3,l4 = size(d1_old)

	#----- extrapolation

	# new arrays
	l1e = round(Int64, factor*l1); l2e = round(Int64, factor*l2); l3e = round(Int64, factor*l3)

	d1 = zeros(eltype(d1_old), l1e,l2e,l3e,4)
	d2 = zeros(eltype(d2_old), l1e,l2e,l3e,4)
	d3 = zeros(eltype(d3_old), l1e,l2e,l3e,4)
	DA = zeros(eltype(DA_old), l1e,l2e,l3e,4)
	DB = zeros(eltype(DB_old), l1e,l2e,l3e,4)

	# coordinate vectors
	Y1 = hD .* ((0:(l1-1)) .- (l1+1)/2); 	Y2 = hD .* ((0:(l2-1)) .- (l2+1)/2); 	Y3 = hD .* ((0:(l3-1)) .- (l3+1)/2)
	Y1e = hD .* ((0:(l1e-1)) .- (l1e+1)/2);	Y2e = hD .* ((0:(l2e-1)) .- (l2e+1)/2); Y3e = hD .* ((0:(l3e-1)) .- (l3e+1)/2)

	# extrapolation
	for c in 1:4
		e_d1 = extrapolate(
				interpolate((Y1,Y2,Y3),d1_old[:,:,:,c], Gridded(Linear())),
				Line())
		e_d2 = extrapolate(
				interpolate((Y1,Y2,Y3),d2_old[:,:,:,c], Gridded(Linear())),
				Line())
		e_d3 = extrapolate(
				interpolate((Y1,Y2,Y3),d3_old[:,:,:,c], Gridded(Linear())),
				Line())
		e_DA = extrapolate(
				interpolate((Y1,Y2,Y3),DA_old[:,:,:,c], Gridded(Linear())),
				Line())
		e_DB = extrapolate(
				interpolate((Y1,Y2,Y3),DB_old[:,:,:,c], Gridded(Linear())),
				Line())

		d1[:,:,:,c] = [e_d1(y1,y2,y3) for y1 in Y1e, y2 in Y2e, y3 in Y3e]
		d2[:,:,:,c] = [e_d2(y1,y2,y3) for y1 in Y1e, y2 in Y2e, y3 in Y3e]
		d3[:,:,:,c] = [e_d3(y1,y2,y3) for y1 in Y1e, y2 in Y2e, y3 in Y3e]
		DA[:,:,:,c] = [e_DA(y1,y2,y3) for y1 in Y1e, y2 in Y2e, y3 in Y3e]
		DB[:,:,:,c] = [e_DB(y1,y2,y3) for y1 in Y1e, y2 in Y2e, y3 in Y3e]
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
	#==
    dy1 = open("/lustre/HQCD/victor.diaz/nnp/data/sample/$(grid_size)/dy1.jls", "r") do io; deserialize(io); end
    dy2 = open("/lustre/HQCD/victor.diaz/nnp/data/sample/$(grid_size)/dy2.jls", "r") do io; deserialize(io); end
    dy3_all = open("/lustre/HQCD/victor.diaz/nnp/data/sample/$(grid_size)/dy3.jls", "r") do io; deserialize(io); end
    dy3 = dy3_all[r_idx,:]

    d1 = open("/lustre/HQCD/victor.diaz/nnp/data/deriv/$(model)/$(grid_size)/d1U_r=$(r_idx)_Q=$(Q_idx).jls", "r") do io; deserialize(io); end
    d2 = open("/lustre/HQCD/victor.diaz/nnp/data/deriv/$(model)/$(grid_size)/d2U_r=$(r_idx)_Q=$(Q_idx).jls", "r") do io; deserialize(io); end
    d3 = open("/lustre/HQCD/victor.diaz/nnp/data/deriv/$(model)/$(grid_size)/d3U_r=$(r_idx)_Q=$(Q_idx).jls", "r") do io; deserialize(io); end
	==#

    dy1 = open("/home/vddiazz/Desktop/temp_nnp/$(grid_size)/dy1.jls", "r") do io; deserialize(io); end
    dy2 = open("/home/vddiazz/Desktop/temp_nnp/$(grid_size)/dy2.jls", "r") do io; deserialize(io); end
    dy3_all = open("/home/vddiazz/Desktop/temp_nnp/$(grid_size)/dy3.jls", "r") do io; deserialize(io); end
    dy3 = dy3_all[r_idx,:]

    d1 = open("/home/vddiazz/Desktop/temp_nnp/d1U_r=$(r_idx)_Q=$(Q_idx).jls", "r") do io; deserialize(io); end
    d2 = open("/home/vddiazz/Desktop/temp_nnp/d2U_r=$(r_idx)_Q=$(Q_idx).jls", "r") do io; deserialize(io); end
    d3 = open("/home/vddiazz/Desktop/temp_nnp/d3U_r=$(r_idx)_Q=$(Q_idx).jls", "r") do io; deserialize(io); end

    l1 = length(d1[:,1,1,1]); l2 = length(d1[1,:,1,1]); l3 = length(d1[1,1,:,1])

    #----- main

    println()
    println("#--------------------------------------------------#")
    println()
    println("Metric (A=$(A), B=$(B)) --- r_idx=$(r_idx), Q_idx=$(Q_idx)")
    println()
 
    DA = open("/home/vddiazz/Desktop/temp_nnp/D$(A)U_r=$(r_idx)_Q=$(Q_idx).jls", "r") do io; deserialize(io); end
    DB = open("/home/vddiazz/Desktop/temp_nnp/D$(B)U_r=$(r_idx)_Q=$(Q_idx).jls", "r") do io; deserialize(io); end

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
