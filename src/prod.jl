#----- pkg

using Serialization
using JLD2
using NPZ
using LinearAlgebra
using SpecialFunctions

using ProgressMeter
using LoopVectorization

#----- field

function field_value(y1,y2,y3,r_val,f_plus,f_minus, coord::Vector{Int64}, Q1::Matrix{ComplexF64}, Q2::Matrix{ComplexF64})

    #----- params

    sigma = [[0im 1.; 1. 0im],[0. -1im; 1im 0.],[1. 0im; 0im -1.]]

    y = [y1[coord[1]], y2[coord[2]], y3[coord[3]]]
    xm = [0., 0., r_val/2.]
    pos_p = y.+xm; pos_m = y.-xm

    ar_p = norm(pos_p, 2)
    ar_m = norm(pos_m, 2)

    #----- f(r) values

    f_p = f_plus[coord[1],coord[2],coord[3],1]
    f_m = f_minus[coord[1],coord[2],coord[3],1]

    #----- U1

    dirs1 = [(Q1*s)*inv(Q1) for s in sigma]
    Z1 = pos_m[1]*dirs1[1] + pos_m[2]*dirs1[2] + pos_m[3]*dirs1[3]
    dirs1_param = [0.5*tr(Z1), 0.5*tr(sigma[1]*Z1), 0.5*tr(sigma[2]*Z1), 0.5*tr(sigma[3]*Z1)]
    phi1 = [cos(f_m)+dirs1_param[1], sin(f_m)*(1/ar_m)*dirs1_param[2], sin(f_m)*(1/ar_m)*dirs1_param[3], sin(f_m)*(1/ar_m)*dirs1_param[4]]

    #----- U2

    dirs2 = [(Q2*s)*inv(Q2) for s in sigma]
    Z2 = pos_p[1]*dirs2[1] + pos_p[2]*dirs2[2] + pos_p[3]*dirs2[3]
    dirs2_param = [0.5*tr(Z2), 0.5*tr(sigma[1]*Z2), 0.5*tr(sigma[2]*Z2), 0.5*tr(sigma[3]*Z2)]
    phi2 = [cos(f_p)+dirs2_param[1], sin(f_p)*(1/ar_p)*dirs2_param[2], sin(f_p)*(1/ar_p)*dirs2_param[3], sin(f_p)*(1/ar_p)*dirs2_param[4]]

    #----- symmetrized product field

    U = [ phi1[1]*phi2[1] - phi1[2]*phi2[2] - phi1[3]*phi2[3] - phi1[4]*phi2[4],
          phi1[1]*phi2[2] + phi1[2]*phi2[1],
          phi1[1]*phi2[3] + phi1[3]*phi2[1],
          phi1[1]*phi2[4] + phi1[4]*phi2[1]]

    #----- normalization

    C0 = (U[1])^2; Ck = U[2]^2 + U[3]^2 + U[4]^2
    N = sqrt(C0 + Ck)

    #----- field
    
    return (1/N) .* U
end

###

# UNUSED?
function field_grid(rtc, f_plus,f_minus, r_val::Float64, Q1::Matrix{ComplexF64}, Q2::Matrix{ComplexF64}, model::String)

    #----- prepare rtc and params

    y1 = rtc[1]; y2 = rtc[2]; y3 = rtc[3]

    #----- evaluate field values
    
    U_vals = zeros(Float64, length(y1),length(y2),length(y3),4)

    step = 0
    for k in 1:length(y3)
        for j in 1:length(y2)
            for i in length(y1)
                U_vals[i,j,k,:] .= real( field_value(y1,y2,y3,r_val,f_plus,f_minus, [i,j,k], Q1, Q2) )
                step =+ 1
                print("\rdone: $(round(step/(length(y1)*length(y2)*length(y3))*100,digits=4)) %")
            end
        end
    end

    return U_vals
end

###

function make_field(grid_size, r_vals, Q_vals, model::String, out::String, output_format::String)

    if (output_format != "jld2") && (output_format != "npy")
        println("invalid output data type")
    end

    #----- params

    y1 = npzread("/home/velni/phd/w/nnp/data/sample/$(grid_size)/y1.npy")
    y2 = npzread("/home/velni/phd/w/nnp/data/sample/$(grid_size)/y2.npy")
    y3 = npzread("/home/velni/phd/w/nnp/data/sample/$(grid_size)/y3.npy")

    rtc = [y1,y2,y3]
    
    f_plus = npzread("/home/velni/phd/w/nnp/data/interp/$(model)/$(grid_size)/f_$(model)_plus.npy")
    f_minus = npzread("/home/velni/phd/w/nnp/data/interp/$(model)/$(grid_size)/f_$(model)_minus.npy")

    IdM = [1. 0im;
           0im 1.]

    println()
    println("#--------------------------------------------------#")
    println()
    println("2-skyrmion field")
    println()

    #----- main loop

    for Q_idx in 1:length(Q_vals)
        for r_idx in 1:length(r_vals)
            
            field_grid(r_vals[r_idx], rtc, r_idx, IdM, Q_vals[Q_idx,:,:], model)
            
            println()
            println("\rdone: r=$(r_idx), Q=$(Q_idx)")
            println()

    #----- data saving

        if output_format == "jld2"
            path = out*"/U_sym_r=$(r_idx)_Q=$(Q_idx).jld2"
            @save path U_vals

        elseif output_format == "npy"
            npzwrite(out*"/U_sym_r=$(r_idx)_Q=$(Q_idx).npy", U_vals)
        end

        end
    end

    println()
    println("data saved at "*out )
    println()
    println("#--------------------------------------------------#")

end

###

function field_grid_single(grid_size::String,y1::Array{Float64},y2::Array{Float64},y3::Array{Float64},f_minus::Array{Float64},f_plus::Array{Float64}, r_val::Float64,Q1::Matrix{ComplexF64}, Q2::Matrix{ComplexF64}, model::String)

    #----- prepare rtc and params
 
    @assert eltype(y1) == Float64 && eltype(y2) == Float64 && eltype(y3) == Float64
    @assert eltype(f_plus) == Float64 && eltype(f_minus) == Float64 

    #----- evaluate field values
    
    U_vals = zeros(Float64, length(y1),length(y2),length(y3),4)

    @showprogress 1 "Computing..." for k in 1:length(y3)
        @inbounds @fastmath for j in 1:length(y2)
            for i in 1:length(y1)
                U_vals[i,j,k,:] .= real( field_value(y1,y2,y3,r_val,f_plus,f_minus, [i,j,k], Q1, Q2) )
            end
        end
    end

    return U_vals
end

###

function FAST_field_grid(grid_size::String,y1::Array{Float64},y2::Array{Float64},y3::Array{Float64},f_minus::Array{Float64},f_plus::Array{Float64}, r_val::Float64,Q1::Matrix{ComplexF64}, Q2::Matrix{ComplexF64}, model::String)

    #----- prepare params

    @assert eltype(y1) == Float64 && eltype(y2) == Float64 && eltype(y3) == Float64
    @assert eltype(f_plus) == Float64 && eltype(f_minus) == Float64 

    sigma1 = [0im 1.; 1. 0im]
    sigma2 = [0. -1im; 1im 0.]
    sigma3 = [1. 0im; 0im -1.]

    #----- prelocate arrays

    l1 = length(y1); l2 = length(y2); l3 = length(y3)

    y = zeros(Float64, 3)
    xm = zeros(Float64, 3)

    dirs1_1 = (Q1*sigma1)*inv(Q1); dirs1_2 = (Q1*sigma2)*inv(Q1); dirs1_3 = (Q1*sigma3)*inv(Q1)
    dirs2_1 = (Q2*sigma1)*inv(Q2); dirs2_2 = (Q2*sigma2)*inv(Q2); dirs2_3 = (Q2*sigma3)*inv(Q2)

    #----- evaluate field values
    
    U_vals = zeros(Float64, l1,l2,l3,4)

    #println()
    #println("2-skyrmion field -- Symmetrized product approximation")
    #println()

    @showprogress 1 "Computing..." for k in 1:l3
        @inbounds @fastmath for j in 1:l2, i in 1:l1
               
            #----- params        

            y[1] = y1[i]; y[2] = y2[j]; y[3] = y3[k]
            xm[3] = r_val/2.
            pos_p_1 = y[1]+xm[1]; pos_p_2 = y[2]+xm[2]; pos_p_3 = y[3]+xm[3]
            pos_m_1 = y[1]-xm[1]; pos_m_2 = y[2]-xm[2]; pos_m_3 = y[3]-xm[3]

            ar_p = sqrt(pos_p_1^2 + pos_p_2^2 + pos_p_3^2)
            ar_m = sqrt(pos_m_1^2 + pos_m_2^2 + pos_m_3^2)

            #----- f(r) values

            f_p = f_plus[i,j,k]
            f_m = f_minus[i,j,k]

            cos_f_p = 1 - 0.5*f_p^2 + (1/24.)*f_p^4 - (1/720.)*f_p^6 + (1/40320.)*f_p^8 - (1/3638800.)*f_p^10
            cos_f_m = 1 - 0.5*f_m^2 + (1/24.)*f_m^4 - (1/720.)*f_m^6 + (1/40320.)*f_m^8 - (1/3638800.)*f_m^10
            sin_f_p = f_p - (1/6.)*f_p^3 + (1/120.)*f_p^5 - (1/5040.)*f_p^7 + (1/362880.)*f_p^9 - (1/39916800.)*f_p^11
            sin_f_m = f_m - (1/6.)*f_m^3 + (1/120.)*f_m^5 - (1/5040.)*f_m^7 + (1/362880.)*f_m^9 - (1/39916800.)*f_m^11

            #----- U1

            Z1_11 = pos_m_1*(dirs1_1[1,1]) + pos_m_2*(dirs1_2[1,1]) + pos_m_3*(dirs1_3[1,1])
            Z1_12 = pos_m_1*(dirs1_1[1,2]) + pos_m_2*(dirs1_2[1,2]) + pos_m_3*(dirs1_3[1,2])
            Z1_21 = pos_m_1*(dirs1_1[2,1]) + pos_m_2*(dirs1_2[2,1]) + pos_m_3*(dirs1_3[2,1])
            Z1_22 = pos_m_1*(dirs1_1[2,2]) + pos_m_2*(dirs1_2[2,2]) + pos_m_3*(dirs1_3[2,2])

            # tr
            tr_11 = sigma1[1,1]*Z1_11+sigma1[1,2]*Z1_21 + sigma1[2,1]*Z1_12+sigma1[2,2]*Z1_22
            tr_21 = sigma2[1,1]*Z1_11+sigma2[1,2]*Z1_21 + sigma2[2,1]*Z1_12+sigma2[2,2]*Z1_22
            tr_31 = sigma3[1,1]*Z1_11+sigma3[1,2]*Z1_21 + sigma3[2,1]*Z1_12+sigma3[2,2]*Z1_22

            dirs1_param_1 = 0.5*(Z1_11+Z1_22); 
            dirs1_param_2 = 0.5*tr_11; 
            dirs1_param_3 = 0.5*tr_21; 
            dirs1_param_4 = 0.5*tr_31

            phi1_1 = cos_f_m + dirs1_param_1; 
            phi1_2 = sin_f_m*(1/ar_m)*dirs1_param_2; 
            phi1_3 = sin_f_m*(1/ar_m)*dirs1_param_3; 
            phi1_4 = sin_f_m*(1/ar_m)*dirs1_param_4

            #----- U2

            Z2_11 = pos_p_1*(dirs2_1[1,1]) + pos_p_2*(dirs2_2[1,1]) + pos_p_3*(dirs2_3[1,1])
            Z2_12 = pos_p_1*(dirs2_1[1,2]) + pos_p_2*(dirs2_2[1,2]) + pos_p_3*(dirs2_3[1,2])
            Z2_21 = pos_p_1*(dirs2_1[2,1]) + pos_p_2*(dirs2_2[2,1]) + pos_p_3*(dirs2_3[2,1])
            Z2_22 = pos_p_1*(dirs2_1[2,2]) + pos_p_2*(dirs2_2[2,2]) + pos_p_3*(dirs2_3[2,2])
           
            # tr
            tr_12 = sigma1[1,1]*Z2_11+sigma1[1,2]*Z2_21 + sigma1[2,1]*Z2_12+sigma1[2,2]*Z2_22
            tr_22 = sigma2[1,1]*Z2_11+sigma2[1,2]*Z2_21 + sigma2[2,1]*Z2_12+sigma2[2,2]*Z2_22
            tr_32 = sigma3[1,1]*Z2_11+sigma3[1,2]*Z2_21 + sigma3[2,1]*Z2_12+sigma3[2,2]*Z2_22

            dirs2_param_1 = 0.5*(Z2_11+Z2_22); 
            dirs2_param_2 = 0.5*tr_12; 
            dirs2_param_3 = 0.5*tr_22; 
            dirs2_param_4 = 0.5*tr_32
            
            phi2_1 = cos_f_p + dirs2_param_1; 
            phi2_2 = sin_f_p*(1/ar_p)*dirs2_param_2; 
            phi2_3 = sin_f_p*(1/ar_p)*dirs2_param_3; 
            phi2_4 = sin_f_p*(1/ar_p)*dirs2_param_4

            #----- symmetrized product field

            U_1 = real(phi1_1*phi2_1 - phi1_2*phi2_2 - phi1_3*phi2_3 - phi1_4*phi2_4)
            U_2 = real(phi1_1*phi2_2 + phi1_2*phi2_1)
            U_3 = real(phi1_1*phi2_3 + phi1_3*phi2_1)
            U_4 = real(phi1_1*phi2_4 + phi1_4*phi2_1)

            #----- normalization

            C0 = U_1^2
            Ck = U_2^2 + U_3^2 + U_4^2
            N = sqrt(C0 + Ck)

            #----- return

            U_vals[i,j,k,1] = (1/N)*U_1; U_vals[i,j,k,2] = (1/N)*U_2; U_vals[i,j,k,3] = (1/N)*U_3; U_vals[i,j,k,4] = (1/N)*U_4
        
        end
    end

    return U_vals
end

###

function FAST_make_field(grid_size::String,y1::Array{Float64},y2::Array{Float64},y3::Array{Float64},f_minus_all::Array{Float64},f_plus_all::Array{Float64}, r_vals::Array{Float64}, Q_vals::Array{ComplexF64}, model::String,out::String,output_format::String)

    if (output_format != "jld2") && (output_format != "npy") && (output_format != "jls")
        println("invalid output data type")
    end

    #----- params

    IdM = [1. 0im;
           0im 1.]

    println()
    println("#--------------------------------------------------#")
    println()
    println("2-skyrmion field")
    println()

    #----- main loop

    @showprogress 1 "Computing..." for Q_idx in 1:length(Q_vals)
        @inbounds @fastmath for r_idx in 1:length(r_vals)

            println()
            prinjtln("r_idx = $(r_idx), Q_idx = $(Q_idx)")
            println()

            f_minus = f_minus_all[r_idx]
            f_plus = f_plus_all[r_idx]
            
            FAST_field_grid(grid_size::String,y1::Array{Float64},y2::Array{Float64},y3::Array{Float64},f_minus::Array{Float64},f_plus::Array{Float64}, r_val::Float64,Q1::Matrix{ComplexF64}, Q2::Matrix{ComplexF64}, model::String)
            
        #----- data saving

            if output_format == "jld2"
                path = out*"/U_sym_r=$(r_idx)_Q=$(Q_idx).jld2"
                @save path U_vals

            elseif output_format == "npy"
                npzwrite(out*"/U_sym_r=$(r_idx)_Q=$(Q_idx).npy", U_vals)
        
            elseif output_format == "jls"
                open(out*"/U_sym_r=$(r_idx)_Q=$(Q_idx).jls", "w") do io
                    serialize(io, U_vals)
                end
        
            end
        end
    
    end

    println()
    println("data saved at "*out )
    println()
    println("#--------------------------------------------------#")

end
