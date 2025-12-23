#----- pkg

using Serialization
using JLD2
using NPZ
using LinearAlgebra
using SpecialFunctions
using ProgressMeter
using LoopVectorization

#####

function deriv_y(dir::String,grid_size::String,r_idx::Int,Q_idx::Int, model::String, input_format::String,out::String,output_format::String)

    if (output_format != "jld2") && (output_format != "npy") && (output_format != "jls")
       println("invalid output data type")
    end
    
    #----- params

    if input_format == "npy"
        field = npzread("/home/velni/phd/w/nnp/data/prod/$(model)/$(grid_size)/U_sym_r=$(r_idx)_Q=$(Q_idx).npy") # KICK OUTSIDE OF FUNCTION
    elseif input_format == "jls"
        y1 = open("/home/velni/phd/w/nnp/data/sample/$(grid_size)/y1.jls", "r") do io; deserialize(io); end
        y2 = open("/home/velni/phd/w/nnp/data/sample/$(grid_size)/y2.jls", "r") do io; deserialize(io); end
        y3 = (open("/home/velni/phd/w/nnp/data/sample/$(grid_size)/y3.jls", "r") do io; deserialize(io); end) # SLICE IF GRID TYPE IS PROY
        field = open("/home/velni/phd/w/nnp/data/prod/$(model)/$(grid_size)/U_sym_r=$(r_idx)_Q=$(Q_idx).jls", "r") do io; deserialize(io); end
    end
    @assert eltype(field) == Float64
    field :: Array{Float64}

    l1 = length(field[:,1,1,1]); l2 = length(field[1,:,1,1]); l3 = length(field[1,1,:,1])

    d_vals = zeros(Float64, l1,l2,l3,4)
    @assert eltype(field) == Float64
    d_vals :: Array{Float64}

    hD = y1[2]-y1[1] # INTRODUCE IN "reg" LOOP
    step = 12*hD

    #----- main loops

    if dir == "1" && startswith(grid_size, "reg")

        @tturbo for k in 3:l3-2, j in 3:l2-2, i in 3:l1-2, c in 1:4
            p1 = field[i+1,j,k,c]
            p2 = field[i+2,j,k,c]
            m1 = field[i-1,j,k,c]
            m2 = field[i-2,j,k,c]
                        
            d_vals[i,j,k,c] = (-p2 + 8*p1 - 8*m1 + m2)/step
        end

    elseif dir == "2" && startswith(grid_size, "reg")

        @tturbo for k in 3:l3-2, j in 3:l2-2, i in 3:l1-2, c in 1:4
            p1 = field[i,j+1,k,c]
            p2 = field[i,j+2,k,c]
            m1 = field[i,j-1,k,c]
            m2 = field[i,j-2,k,c]
                        
            d_vals[i,j,k,c] = (-p2 + 8*p1 - 8*m1 + m2)/step
        end
    
    elseif dir == "3" && startswith(grid_size, "reg")
        
        @tturbo for k in 3:l3-2, j in 3:l2-2, i in 3:l1-2, c in 1:4
            p1 = field[i,j,k+1,c]
            p2 = field[i,j,k+2,c]
            m1 = field[i,j,k-1,c]
            m2 = field[i,j,k-2,c]
                        
            d_vals[i,j,k,c] = (-p2 + 8*p1 - 8*m1 + m2)/step
        end
    end

    #----- data saving

    if output_format == "jld2"
        path = out*"/d$(dir)U_r=$(r_idx)_Q=$(Q_idx).jld2"
        @save path d_vals

    elseif output_format == "npy"
        npzwrite(out*"/d$(dir)U_r=$(r_idx)_Q=$(Q_idx).npy", d_vals)
    elseif output_format == "jls"
        open(out*"/d$(dir)U_r=$(r_idx)_Q=$(Q_idx).jls", "w") do io
            serialize(io, d_vals)
        end
    end

end

###

function deriv_x(dir::String,grid_size::String,hD::Float64,y1::Array{Float64},y2::Array{Float64},y3::Array{Float64},f_minus_m::Array{Float64},f_minus_p::Array{Float64},f_plus_m::Array{Float64},f_plus_p::Array{Float64},r_val::Float64,Q1::Matrix{ComplexF64}, Q2::Matrix{ComplexF64}, model::String)

    #----- prepare params

    @assert eltype(y1) == Float64 && eltype(y2) == Float64 && eltype(y3) == Float64
    @assert eltype(f_minus_m) == Float64 && eltype(f_minus_p) == Float64 && eltype(f_plus_m) == Float64 && eltype(f_plus_p) == Float64

    sigma1 = [0im 1.; 1. 0im]
    sigma2 = [0. -1im; 1im 0.]
    sigma3 = [1. 0im; 0im -1.]

    #----- prelocate arrays

    y = zeros(Float64, 3)
    xm = zeros(Float64, 3)

    dirs1_1 = (Q1*sigma1)*inv(Q1); dirs1_2 = (Q1*sigma2)*inv(Q1); dirs1_3 = (Q1*sigma3)*inv(Q1)
    dirs2_1 = (Q2*sigma1)*inv(Q2); dirs2_2 = (Q2*sigma2)*inv(Q2); dirs2_3 = (Q2*sigma3)*inv(Q2)

    #----- evaluate field values
    
    U_vals_m = zeros(Float64, length(y1),length(y2),length(y3),4)
    U_vals_p = zeros(Float64, length(y1),length(y2),length(y3),4)

    println()
    println("#--------------------------------------------------#")
    println()
    println("2-skyrmion field -- x$(dir) derivative")
    println()

    @showprogress 1 "Computing field (m)..." for k in 1:length(y3)
        @inbounds @fastmath for j in 1:length(y2), i in 1:length(y1)
               
            #----- params        

            y[1] = y1[i]; y[2] = y2[j]; y[3] = y3[k]
            if dir == "1"
                xm[1] = -hD; xm[3] = r_val
            elseif dir == "2"
                xm[2] = -hD; xm[3] = r_val
            elseif dir == "3"
                xm[3] = r_val - hD
            end
            pos_p_1 = y[1]+xm[1]/2.; pos_p_2 = y[2]+xm[2]/2.; pos_p_3 = y[3]+xm[3]/2.
            pos_m_1 = y[1]-xm[1]/2.; pos_m_2 = y[2]-xm[2]/2.; pos_m_3 = y[3]-xm[3]/2.

            ar_p = sqrt(pos_p_1^2 + pos_p_2^2 + pos_p_3^2)
            ar_m = sqrt(pos_m_1^2 + pos_m_2^2 + pos_m_3^2)

            #----- f(r) values

            f_p = f_plus_m[i,j,k]
            f_m = f_minus_m[i,j,k]

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

            U_vals_m[i,j,k,1] = (1/N)*U_1; U_vals_m[i,j,k,2] = (1/N)*U_2; U_vals_m[i,j,k,3] = (1/N)*U_3; U_vals_m[i,j,k,4] = (1/N)*U_4
        
        end
    end

    @showprogress 1 "Computing field (p)..." for k in 1:length(y3)
        @inbounds @fastmath for j in 1:length(y2), i in 1:length(y1)
               
            #----- params        

            y[1] = y1[i]; y[2] = y2[j]; y[3] = y3[k]
            if dir == "1"
                xm[1] = hD; xm[3] = r_val
            elseif dir == "2"
                xm[2] = hD; xm[3] = r_val
            elseif dir == "3"
                xm[3] = r_val + hD
            end
            pos_p_1 = y[1]+xm[1]/2.; pos_p_2 = y[2]+xm[2]/2.; pos_p_3 = y[3]+xm[3]/2.
            pos_m_1 = y[1]-xm[1]/2.; pos_m_2 = y[2]-xm[2]/2.; pos_m_3 = y[3]-xm[3]/2.

            ar_p = sqrt(pos_p_1^2 + pos_p_2^2 + pos_p_3^2)
            ar_m = sqrt(pos_m_1^2 + pos_m_2^2 + pos_m_3^2)

            #----- f(r) values

            f_p = f_plus_p[i,j,k]
            f_m = f_minus_p[i,j,k]

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

            U_vals_p[i,j,k,1] = (1/N)*U_1; U_vals_p[i,j,k,2] = (1/N)*U_2; U_vals_p[i,j,k,3] = (1/N)*U_3; U_vals_p[i,j,k,4] = (1/N)*U_4
        
        end
    end

    dU_vals = zeros(Float64, length(y1),length(y2),length(y3),4)
    dU_vals .= (1/(2*hD)).*(U_vals_p .- U_vals_m)

    return dU_vals
end

###

function deriv_Q1(dir::Int64,grid_size::String,hD::Float64,y1::Array{Float64},y2::Array{Float64},y3::Array{Float64},f_minus::Array{Float64},f_plus::Array{Float64},r_val::Float64,Q1::Matrix{ComplexF64}, Q2::Matrix{ComplexF64}, model::String)

    #----- prepare params

    @assert eltype(y1) == Float64 && eltype(y2) == Float64 && eltype(y3) == Float64
    @assert eltype(f_minus) == Float64 && eltype(f_plus) == Float64

    sigma1 = [0im 1.; 1. 0im]
    sigma2 = [0. -1im; 1im 0.]
    sigma3 = [1. 0im; 0im -1.]
    I = [1. 0im; 0im 1.]

    #----- prelocate arrays

    y = zeros(Float64, 3)
    xm = zeros(Float64, 3)

    arg_p = -0.5*hD
    arg_m = -0.5*(-hD)

    if dir == 1
        temp_m = cos(arg_m)*I + 1im*sin(arg_m)*sigma1
        temp_p = cos(arg_p)*I + 1im*sin(arg_p)*sigma1
    elseif dir == 2
        temp_m = cos(arg_m)*I + 1im*sin(arg_m)*sigma2
        temp_p = cos(arg_p)*I + 1im*sin(arg_p)*sigma2
    elseif dir == 3
        temp_m = cos(arg_m)*I + 1im*sin(arg_m)*sigma3
        temp_p = cos(arg_p)*I + 1im*sin(arg_p)*sigma3
    end

    Q1_p = Q1*temp_p
    Q1_m = Q1*temp_m

    dirs1_1_p = (Q1_p*sigma1)*inv(Q1_p); dirs1_2_p = (Q1_p*sigma2)*inv(Q1_p); dirs1_3_p = (Q1_p*sigma3)*inv(Q1_p)
    dirs1_1_m = (Q1_m*sigma1)*inv(Q1_m); dirs1_2_m = (Q1_m*sigma2)*inv(Q1_m); dirs1_3_m = (Q1_m*sigma3)*inv(Q1_m)
    dirs2_1 = (Q2*sigma1)*inv(Q2); dirs2_2 = (Q2*sigma2)*inv(Q2); dirs2_3 = (Q2*sigma3)*inv(Q2)

    #----- evaluate field values
    
    U_vals_m = zeros(Float64, length(y1),length(y2),length(y3),4)
    U_vals_p = zeros(Float64, length(y1),length(y2),length(y3),4)
    
    println()
    println("#--------------------------------------------------#")
    println()
    println("2-skyrmion field -- Q1_$(dir) derivative")
    println()

    @showprogress 1 "Computing field (m)..." for k in 1:length(y3)
        @inbounds @fastmath for j in 1:length(y2), i in 1:length(y1)
               
            #----- params        

            y[1] = y1[i]; y[2] = y2[j]; y[3] = y3[k]
            xm[3] = r_val
            
            pos_p_1 = y[1]+xm[1]/2.; pos_p_2 = y[2]+xm[2]/2.; pos_p_3 = y[3]+xm[3]/2.
            pos_m_1 = y[1]-xm[1]/2.; pos_m_2 = y[2]-xm[2]/2.; pos_m_3 = y[3]-xm[3]/2.

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

            Z1_11 = pos_m_1*(dirs1_1_m[1,1]) + pos_m_2*(dirs1_2_m[1,1]) + pos_m_3*(dirs1_3_m[1,1])
            Z1_12 = pos_m_1*(dirs1_1_m[1,2]) + pos_m_2*(dirs1_2_m[1,2]) + pos_m_3*(dirs1_3_m[1,2])
            Z1_21 = pos_m_1*(dirs1_1_m[2,1]) + pos_m_2*(dirs1_2_m[2,1]) + pos_m_3*(dirs1_3_m[2,1])
            Z1_22 = pos_m_1*(dirs1_1_m[2,2]) + pos_m_2*(dirs1_2_m[2,2]) + pos_m_3*(dirs1_3_m[2,2])

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

            U_vals_m[i,j,k,1] = (1/N)*U_1; U_vals_m[i,j,k,2] = (1/N)*U_2; U_vals_m[i,j,k,3] = (1/N)*U_3; U_vals_m[i,j,k,4] = (1/N)*U_4
        
        end
    end

    @showprogress 1 "Computing field (p)..." for k in 1:length(y3)
        @inbounds @fastmath for j in 1:length(y2), i in 1:length(y1)
               
            #----- params        

            y[1] = y1[i]; y[2] = y2[j]; y[3] = y3[k]
            xm[3] = r_val

            pos_p_1 = y[1]+xm[1]/2.; pos_p_2 = y[2]+xm[2]/2.; pos_p_3 = y[3]+xm[3]/2.
            pos_m_1 = y[1]-xm[1]/2.; pos_m_2 = y[2]-xm[2]/2.; pos_m_3 = y[3]-xm[3]/2.

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

            Z1_11 = pos_m_1*(dirs1_1_p[1,1]) + pos_m_2*(dirs1_2_p[1,1]) + pos_m_3*(dirs1_3_p[1,1])
            Z1_12 = pos_m_1*(dirs1_1_p[1,2]) + pos_m_2*(dirs1_2_p[1,2]) + pos_m_3*(dirs1_3_p[1,2])
            Z1_21 = pos_m_1*(dirs1_1_p[2,1]) + pos_m_2*(dirs1_2_p[2,1]) + pos_m_3*(dirs1_3_p[2,1])
            Z1_22 = pos_m_1*(dirs1_1_p[2,2]) + pos_m_2*(dirs1_2_p[2,2]) + pos_m_3*(dirs1_3_p[2,2])

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

            U_vals_p[i,j,k,1] = (1/N)*U_1; U_vals_p[i,j,k,2] = (1/N)*U_2; U_vals_p[i,j,k,3] = (1/N)*U_3; U_vals_p[i,j,k,4] = (1/N)*U_4
        
        end
    end

    dU_vals = zeros(Float64, length(y1),length(y2),length(y3),4)
    dU_vals .= (1/(2*hD)).*(U_vals_p .- U_vals_m)

    return dU_vals
end

###

function deriv_Q2(dir::Int64,grid_size::String,hD::Float64,y1::Array{Float64},y2::Array{Float64},y3::Array{Float64},f_minus::Array{Float64},f_plus::Array{Float64},r_val::Float64,Q1::Matrix{ComplexF64}, Q2::Matrix{ComplexF64}, model::String)

    #----- prepare params

    @assert eltype(y1) == Float64 && eltype(y2) == Float64 && eltype(y3) == Float64
    @assert eltype(f_minus) == Float64 && eltype(f_plus) == Float64

    sigma1 = [0im 1.; 1. 0im]
    sigma2 = [0. -1im; 1im 0.]
    sigma3 = [1. 0im; 0im -1.]
    I = [1. 0im; 0im 1.]

    #----- prelocate arrays

    y = zeros(Float64, 3)
    xm = zeros(Float64, 3)

    arg_p = -0.5*hD
    arg_m = -0.5*(-hD)

    if dir == 1
        temp_m = cos(arg_m)*I + 1im*sin(arg_m)*sigma1
        temp_p = cos(arg_p)*I + 1im*sin(arg_p)*sigma1
    elseif dir == 2
        temp_m = cos(arg_m)*I + 1im*sin(arg_m)*sigma2
        temp_p = cos(arg_p)*I + 1im*sin(arg_p)*sigma2
    elseif dir == 3
        temp_m = cos(arg_m)*I + 1im*sin(arg_m)*sigma3
        temp_p = cos(arg_p)*I + 1im*sin(arg_p)*sigma3
    end

    Q2_p = Q2*temp_p
    Q2_m = Q2*temp_m

    dirs2_1_p = (Q2_p*sigma1)*inv(Q2_p); dirs2_2_p = (Q2_p*sigma2)*inv(Q2_p); dirs2_3_p = (Q2_p*sigma3)*inv(Q2_p)
    dirs2_1_m = (Q2_m*sigma1)*inv(Q2_m); dirs2_2_m = (Q2_m*sigma2)*inv(Q2_m); dirs2_3_m = (Q2_m*sigma3)*inv(Q2_m)
    dirs1_1 = (Q1*sigma1)*inv(Q1); dirs1_2 = (Q1*sigma2)*inv(Q1); dirs1_3 = (Q1*sigma3)*inv(Q1)

    #----- evaluate field values
    
    U_vals_m = zeros(Float64, length(y1),length(y2),length(y3),4)
    U_vals_p = zeros(Float64, length(y1),length(y2),length(y3),4)
    
    println()
    println("#--------------------------------------------------#")
    println()
    println("2-skyrmion field -- Q2_$(dir) derivative")
    println()

    @showprogress 1 "Computing field (m)..." for k in 1:length(y3)
        @inbounds @fastmath for j in 1:length(y2), i in 1:length(y1)
               
            #----- params        

            y[1] = y1[i]; y[2] = y2[j]; y[3] = y3[k]
            xm[3] = r_val
            
            pos_p_1 = y[1]+xm[1]/2.; pos_p_2 = y[2]+xm[2]/2.; pos_p_3 = y[3]+xm[3]/2.
            pos_m_1 = y[1]-xm[1]/2.; pos_m_2 = y[2]-xm[2]/2.; pos_m_3 = y[3]-xm[3]/2.

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

            Z2_11 = pos_p_1*(dirs2_1_m[1,1]) + pos_p_2*(dirs2_2_m[1,1]) + pos_p_3*(dirs2_3_m[1,1])
            Z2_12 = pos_p_1*(dirs2_1_m[1,2]) + pos_p_2*(dirs2_2_m[1,2]) + pos_p_3*(dirs2_3_m[1,2])
            Z2_21 = pos_p_1*(dirs2_1_m[2,1]) + pos_p_2*(dirs2_2_m[2,1]) + pos_p_3*(dirs2_3_m[2,1])
            Z2_22 = pos_p_1*(dirs2_1_m[2,2]) + pos_p_2*(dirs2_2_m[2,2]) + pos_p_3*(dirs2_3_m[2,2])
           
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

            U_vals_m[i,j,k,1] = (1/N)*U_1; U_vals_m[i,j,k,2] = (1/N)*U_2; U_vals_m[i,j,k,3] = (1/N)*U_3; U_vals_m[i,j,k,4] = (1/N)*U_4
        
        end
    end

    @showprogress 1 "Computing field (p)..." for k in 1:length(y3)
        @inbounds @fastmath for j in 1:length(y2), i in 1:length(y1)
               
            #----- params        

            y[1] = y1[i]; y[2] = y2[j]; y[3] = y3[k]
            xm[3] = r_val

            pos_p_1 = y[1]+xm[1]/2.; pos_p_2 = y[2]+xm[2]/2.; pos_p_3 = y[3]+xm[3]/2.
            pos_m_1 = y[1]-xm[1]/2.; pos_m_2 = y[2]-xm[2]/2.; pos_m_3 = y[3]-xm[3]/2.

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

            Z2_11 = pos_p_1*(dirs2_1_p[1,1]) + pos_p_2*(dirs2_2_p[1,1]) + pos_p_3*(dirs2_3_p[1,1])
            Z2_12 = pos_p_1*(dirs2_1_p[1,2]) + pos_p_2*(dirs2_2_p[1,2]) + pos_p_3*(dirs2_3_p[1,2])
            Z2_21 = pos_p_1*(dirs2_1_p[2,1]) + pos_p_2*(dirs2_2_p[2,1]) + pos_p_3*(dirs2_3_p[2,1])
            Z2_22 = pos_p_1*(dirs2_1_p[2,2]) + pos_p_2*(dirs2_2_p[2,2]) + pos_p_3*(dirs2_3_p[2,2])
           
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

            U_vals_p[i,j,k,1] = (1/N)*U_1; U_vals_p[i,j,k,2] = (1/N)*U_2; U_vals_p[i,j,k,3] = (1/N)*U_3; U_vals_p[i,j,k,4] = (1/N)*U_4
        
        end
    end

    dU_vals = zeros(Float64, length(y1),length(y2),length(y3),4)
    dU_vals .= (1/(2*hD)).*(U_vals_p .- U_vals_m)

    return dU_vals
end

###

function deriv_y_subtract(dir::String,grid_size::String,hD::Float64,y1::Array{Float64},y2::Array{Float64},y3::Array{Float64},f_minus_m::Array{Float64},f_minus_p::Array{Float64},f_plus_m::Array{Float64},f_plus_p::Array{Float64},r_val::Float64,Q1::Matrix{ComplexF64}, Q2::Matrix{ComplexF64}, model::String)

    #----- prepare params

    l1 = length(y1); l2 = length(y2); l3 = length(y3)

    @assert eltype(y1) == Float64 && eltype(y2) == Float64 && eltype(y3) == Float64
    @assert eltype(f_minus_m) == Float64 && eltype(f_minus_p) == Float64 && eltype(f_plus_m) == Float64 && eltype(f_plus_p) == Float64

    sigma1 = [0im 1.; 1. 0im]
    sigma2 = [0. -1im; 1im 0.]
    sigma3 = [1. 0im; 0im -1.]

    #----- prelocate arrays

    y = zeros(Float64, 3)
    xm = zeros(Float64, 3)

    dirs1_1 = (Q1*sigma1)*inv(Q1); dirs1_2 = (Q1*sigma2)*inv(Q1); dirs1_3 = (Q1*sigma3)*inv(Q1)
    dirs2_1 = (Q2*sigma1)*inv(Q2); dirs2_2 = (Q2*sigma2)*inv(Q2); dirs2_3 = (Q2*sigma3)*inv(Q2)

    #----- evaluate field values
    
    U_vals_m = zeros(Float64, l1,l2,l3,4)
    U_vals_p = zeros(Float64, l1,l2,l3,4)

    println()
    println("#--------------------------------------------------#")
    println()
    println("2-skyrmion field -- y$(dir) derivative")
    println()

    @showprogress 1 "Computing field (m)..." for k in 1:l3
        @inbounds @fastmath for j in 1:l2, i in 1:l1
               
            #----- params        

            if dir == "1"
                y[1] = y1[i] - hD; y[2] = y2[j]; y[3] = y3[k]
                xm[1] = 0; xm[2] = 0; xm[3] = r_val
            elseif dir == "2"
                y[1] = y1[i]; y[2] = y2[j] - hD; y[3] = y3[k]
                xm[1] = 0; xm[2] = 0; xm[3] = r_val
            elseif dir == "3"
                y[1] = y1[i]; y[2] = y2[j]; y[3] = y3[k] - hD
                xm[1] = 0; xm[2] = 0; xm[3] = r_val
            end

            y[1] = y1[i]; y[2] = y2[j]; y[3] = y3[k]
            xm[1] = 0; xm[2] = 0; xm[3] = r_val
            
            pos_p_1 = y[1]+xm[1]/2.; pos_p_2 = y[2]+xm[2]/2.; pos_p_3 = y[3]+xm[3]/2.
            pos_m_1 = y[1]-xm[1]/2.; pos_m_2 = y[2]-xm[2]/2.; pos_m_3 = y[3]-xm[3]/2.

            ar_p = sqrt(pos_p_1^2 + pos_p_2^2 + pos_p_3^2)
            ar_m = sqrt(pos_m_1^2 + pos_m_2^2 + pos_m_3^2)

            #----- f(r) values

            f_p = f_plus_m[i,j,k]
            f_m = f_minus_m[i,j,k]

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

            U_vals_m[i,j,k,1] = (1/N)*U_1; U_vals_m[i,j,k,2] = (1/N)*U_2; U_vals_m[i,j,k,3] = (1/N)*U_3; U_vals_m[i,j,k,4] = (1/N)*U_4
        
        end
    end

    @showprogress 1 "Computing field (p)..." for k in 1:l3
        @inbounds @fastmath for j in 1:l2, i in 1:l1
               
            #----- params        

            if dir == "1"
                y[1] = y1[i] + hD; y[2] = y2[j]; y[3] = y3[k]
                xm[1] = 0; xm[2] = 0; xm[3] = r_val
            elseif dir == "2"
                y[1] = y1[i]; y[2] = y2[j] + hD; y[3] = y3[k]
                xm[1] = 0; xm[2] = 0; xm[3] = r_val
            elseif dir == "3"
                y[1] = y1[i]; y[2] = y2[j]; y[3] = y3[k] + hD
                xm[1] = 0; xm[2] = 0; xm[3] = r_val
            end
            
            pos_p_1 = y[1]+xm[1]/2.; pos_p_2 = y[2]+xm[2]/2.; pos_p_3 = y[3]+xm[3]/2.
            pos_m_1 = y[1]-xm[1]/2.; pos_m_2 = y[2]-xm[2]/2.; pos_m_3 = y[3]-xm[3]/2.

            ar_p = sqrt(pos_p_1^2 + pos_p_2^2 + pos_p_3^2)
            ar_m = sqrt(pos_m_1^2 + pos_m_2^2 + pos_m_3^2)

            #----- f(r) values

            f_p = f_plus_p[i,j,k]
            f_m = f_minus_p[i,j,k]

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

            U_vals_p[i,j,k,1] = (1/N)*U_1; U_vals_p[i,j,k,2] = (1/N)*U_2; U_vals_p[i,j,k,3] = (1/N)*U_3; U_vals_p[i,j,k,4] = (1/N)*U_4
        
        end
    end

    dU_vals = zeros(Float64, l1,l2,l3,4)
    dU_vals .= (1/(2*hD)).*(U_vals_p .- U_vals_m)

    #----- interp missing NaN

    dU_vals[40,40,40,:] .= dU_vals[40,40,41,:]
    dU_vals[40,40,120,:] .= dU_vals[40,40,121,:]

    return dU_vals
end


