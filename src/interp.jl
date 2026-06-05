#----- pkg

using Serialization
using JLD2
using NPZ
using LinearAlgebra
using SpecialFunctions
using ProgressMeter

#----- interp

using DelimitedFiles
using Interpolations

function interp_2sky_no(rtc,r_vals::Array{Float64}, model::String,data, out::String,output_format::String)

    r0 = data[:,1]; f0 = data[:,2]

    y1 = rtc[1]; y2 = rtc[2]; y3 = rtc[3]
    l1 = length(y1); l2 = length(y2); l3 = size(y3,ndims(y3))

    #----- main loop
    
    r0_itp = first(r0):0.01:last(r0)

    itp_inner = interpolate(f0, BSpline(Linear()))
    itp_scaled = scale(itp_inner, r0_itp)
    function itp(x)
        if x < first(r0)
            return 3.14159
        elseif x > last(r0)
            return 0
        else
            return itp_scaled(x)
        end
    end

    println()
    println("#--------------------------------------------------#")
    println()
    println("Radial function interpolation")
    println()

    @showprogress 1 "Computing..." for r_idx in 1:length(r_vals)
        
        matrix_f_plus = zeros(Float64, l1,l2,l3); matrix_f_minus = zeros(Float64, l1,l2,l3)

        r = r_vals[r_idx]
        
        @inbounds @fastmath for k in 1:l3, j in 1:l2, i in 1:l1
            temp_f_plus = itp(norm([y1[i],y2[j],y3[k]] .+ [0.,0.,r/2]) ) 
            temp_f_minus = itp(norm([y1[i],y2[j],y3[k]] .- [0.,0.,r/2]) )
            
            matrix_f_plus[i,j,k] = temp_f_plus
            matrix_f_minus[i,j,k] = temp_f_minus
        end 

        #----- data saving

        if output_format == "jld2"
            path1 = out*"/f_$(model)_plus_r=$(r_idx).jld2"; path2 = out*"/f_$(model)_minus_r=$(r_idx).jld2"
            @save path1 matrix_f_plus; @save path2 matrix_f_minus
        elseif output_format == "npy"
            npzwrite(out*"/f_$(model)_plus_r=$(r_idx).npy", matrix_f_plus); npzwrite(out*"/f_$(model)_minus_r=$(r_idx).npy", matrix_f_minus)
        elseif output_format == "jls"
            open(out*"/f_$(model)_plus_r=$(r_idx).jls", "w") do io
                serialize(io, matrix_f_plus)
            end
            open(out*"/f_$(model)_minus_r=$(r_idx).jls", "w") do io
                serialize(io, matrix_f_minus)
            end
        end

    end
        
    println()
    println("data saved at "*out )
    println()
    println("#--------------------------------------------------#")
end

#####

function interp_2sky_dx(rtc,r_vals, model::String,deriv::String,hD::Float64, out::String,output_format::String)

    data = readdlm("/home/vddiazz/phd/w/nnp/data/profile_f/profile_f_$(model).txt")
    r0 = data[:,1]; f0 = data[:,2]

    y1 = rtc[1]; y2 = rtc[2]; y3 = rtc[3]
    l1 = length(y1); l2 = length(y2); l3 = size(y3,ndims(y3))

    #----- main loop

    r0_itp = first(r0):0.01:last(r0)

    itp_inner = interpolate(f0, BSpline(Linear()))
    itp_scaled = scale(itp_inner, r0_itp)
    function itp(x)
        if x < first(r0)
            return 3.14159
        elseif x > last(r0)
            return 0
        else
            return itp_scaled(x)
        end
    end    

    println()
    println("#--------------------------------------------------#")
    println()
    println("Radial function interpolation --- $(deriv) direction")
    
    
    for r_idx in 1:length(r_vals)
        
        matrix_f_minus_m = zeros(Float64, l1,l2,l3); matrix_f_minus_p = zeros(Float64, l1,l2,l3)
        matrix_f_plus_m = zeros(Float64, l1,l2,l3); matrix_f_plus_p = zeros(Float64, l1,l2,l3)

        println()

        r = r_vals[r_idx]
   
        if deriv == "x1"
            x_p = [hD,0.,r]./2; x_m = [-hD,0.,r]./2
        elseif deriv == "x2"
            x_p = [0.,hD,r]./2; x_m = [0.,-hD,r]./2
        elseif deriv == "x3"
            x_p = [0.,0.,hD+r]./2; x_m = [0.,0.,-hD+r]./2
        end

        @showprogress 1 "Computing r=$(r_idx):" for k in 1:l3
            @inbounds @fastmath for j in 1:l2
                for i in 1:l1
                    temp_f_plus_p = itp(norm([y1[i],y2[j],y3[k]] .+ x_p) )
                    temp_f_plus_m = itp(norm([y1[i],y2[j],y3[k]] .+ x_m) )
                    temp_f_minus_p = itp(norm([y1[i],y2[j],y3[k]] .- x_p) )
                    temp_f_minus_m = itp(norm([y1[i],y2[j],y3[k]] .- x_m) )

                    matrix_f_plus_p[i,j,k] = temp_f_plus_p
                    matrix_f_plus_m[i,j,k] = temp_f_plus_m
                    matrix_f_minus_p[i,j,k] = temp_f_minus_p
                    matrix_f_minus_m[i,j,k] = temp_f_minus_m
                end
            end
        end
        
        #----- data saving

        println()
        println("Saving data...")

        if output_format == "jld2"
            path1 = out*"/f_$(model)_$(deriv)_plus_p_r=$(r_idx).jld2"; path2 = out*"/f_$(model)_$(deriv)_plus_m_r=$(r_idx).jld2"
            path3 = out*"/f_$(model)_$(deriv)_minus_p_r=$(r_idx).jld2"; path4 = out*"/f_$(model)_$(deriv)_minus_m_r=$(r_idx).jld2"
            @save path1 matrix_f_plus_p; @save path2 matrix_f_plus_m; @save path3 matrix_f_minus_p; @save path4 matrix_f_minus_m; 

        elseif output_format == "npy"
            npzwrite(out*"/f_$(model)_$(deriv)_plus_p_r=$(r_idx).npy", matrix_f_plus_p); npzwrite(out*"/f_$(model)_$(deriv)_plus_m_r=$(r_idx).npy", matrix_f_plus_m)
            npzwrite(out*"/f_$(model)_$(deriv)_minus_p_r=$(r_idx).npy", matrix_f_minus_p); npzwrite(out*"/f_$(model)_$(deriv)_minus_m_r=$(r_idx).npy", matrix_f_minus_m)
         elseif output_format == "jls"
            open(out*"/f_$(model)_$(deriv)_plus_p_r=$(r_idx).jls", "w") do io
                serialize(io, matrix_f_plus_p)
            end
            open(out*"/f_$(model)_$(deriv)_plus_m_r=$(r_idx).jls", "w") do io
                serialize(io, matrix_f_plus_m)
            end
            open(out*"/f_$(model)_$(deriv)_minus_p_r=$(r_idx).jls", "w") do io
                serialize(io, matrix_f_minus_p)
            end
            open(out*"/f_$(model)_$(deriv)_minus_m_r=$(r_idx).jls", "w") do io
                serialize(io, matrix_f_minus_m)
            end

        end
    end
    
    println()
    println("Data saved at "*out )
    println()
    println("#--------------------------------------------------#")
end

#####

function interp_2sky_dy(rtc,r_vals, model::String,deriv::String,hD::Float64, out::String,output_format::String)

    data = readdlm("/home/vddiazz/phd/w/nnp/data/profile_f/profile_f_$(model).txt")
    r0 = data[:,1]; f0 = data[:,2]

    y1 = rtc[1]; y2 = rtc[2]; y3 = rtc[3]
    l1 = length(y1); l2 = length(y2); l3 = size(y3,ndims(y3))

    #----- main loop

    r0_itp = first(r0):0.01:last(r0)

    itp_inner = interpolate(f0, BSpline(Linear()))
    itp_scaled = scale(itp_inner, r0_itp)
    function itp(x)
        if x < first(r0)
            return 3.14159
        elseif x > last(r0)
            return 0
        else
            return itp_scaled(x)
        end
    end    

    println()
    println("#--------------------------------------------------#")
    println()
    println("Radial function interpolation --- $(deriv) direction")
    
    
    for r_idx in 1:length(r_vals)
        
        matrix_f_minus_m = zeros(Float64, l1,l2,l3); matrix_f_minus_p = zeros(Float64, l1,l2,l3)
        matrix_f_plus_m = zeros(Float64, l1,l2,l3); matrix_f_plus_p = zeros(Float64, l1,l2,l3)

        println()

        r = r_vals[r_idx]
   
        if deriv == "y1"
            y1_p = y1 .+ hD; y1_m = y1 .- hD
            y2_p = y2; y2_m = y2
            y3_p = y3; y3_m = y3
        elseif deriv == "y2"
            y1_p = y1; y1_m = y1
            y2_p = y2 .+ hD; y2_m = y2 .- hD
            y3_p = y3; y3_m = y3 
        elseif deriv == "y3"
            y1_p = y1; y1_m = y1
            y2_p = y2; y2_m = y2
            y3_p = y3 .+ hD; y3_m = y3 .- hD
        end

        x = [0,0,r]/2.

        @showprogress 1 "Computing r=$(r_idx):" for k in 1:l3
            @inbounds @fastmath for j in 1:l2
                for i in 1:l1
                    temp_f_plus_p = itp(norm([y1_p[i],y2_p[j],y3_p[k]] .+ x) )
                    temp_f_plus_m = itp(norm([y1_m[i],y2_m[j],y3_m[k]] .+ x) )
                    temp_f_minus_p = itp(norm([y1_p[i],y2_p[j],y3_p[k]] .- x) )
                    temp_f_minus_m = itp(norm([y1_m[i],y2_m[j],y3_m[k]] .- x) )

                    matrix_f_plus_p[i,j,k] = temp_f_plus_p
                    matrix_f_plus_m[i,j,k] = temp_f_plus_m
                    matrix_f_minus_p[i,j,k] = temp_f_minus_p
                    matrix_f_minus_m[i,j,k] = temp_f_minus_m
                end
            end
        end
        
        #----- data saving

        println()
        println("Saving data...")

        if output_format == "jld2"
            path1 = out*"/f_$(model)_$(deriv)_plus_p_r=$(r_idx).jld2"; path2 = out*"/f_$(model)_$(deriv)_plus_m_r=$(r_idx).jld2"
            path3 = out*"/f_$(model)_$(deriv)_minus_p_r=$(r_idx).jld2"; path4 = out*"/f_$(model)_$(deriv)_minus_m_r=$(r_idx).jld2"
            @save path1 matrix_f_plus_p; @save path2 matrix_f_plus_m; @save path3 matrix_f_minus_p; @save path4 matrix_f_minus_m; 

        elseif output_format == "npy"
            npzwrite(out*"/f_$(model)_$(deriv)_plus_p_r=$(r_idx).npy", matrix_f_plus_p); npzwrite(out*"/f_$(model)_$(deriv)_plus_m_r=$(r_idx).npy", matrix_f_plus_m)
            npzwrite(out*"/f_$(model)_$(deriv)_minus_p_r=$(r_idx).npy", matrix_f_minus_p); npzwrite(out*"/f_$(model)_$(deriv)_minus_m_r=$(r_idx).npy", matrix_f_minus_m)
         elseif output_format == "jls"
            open(out*"/f_$(model)_$(deriv)_plus_p_r=$(r_idx).jls", "w") do io
                serialize(io, matrix_f_plus_p)
            end
            open(out*"/f_$(model)_$(deriv)_plus_m_r=$(r_idx).jls", "w") do io
                serialize(io, matrix_f_plus_m)
            end
            open(out*"/f_$(model)_$(deriv)_minus_p_r=$(r_idx).jls", "w") do io
                serialize(io, matrix_f_minus_p)
            end
            open(out*"/f_$(model)_$(deriv)_minus_m_r=$(r_idx).jls", "w") do io
                serialize(io, matrix_f_minus_m)
            end

        end
    end
    
    println()
    println("Data saved at "*out )
    println()
    println("#--------------------------------------------------#")
end

#---------- proy

function interp_2sky_no_proy(rtc,r_vals::Array{Float64}, model::String,data, out::String,output_format::String)

    r0 = data[:,1]; f0 = data[:,2]

    y1 = rtc[1]; y2 = rtc[2]; y3 = rtc[3]
    l1 = length(y1); l2 = length(y2); l3 = length(y3[1,:])

    #----- main loop
    
    r0_itp = first(r0):0.01:last(r0)

    itp_inner = interpolate(f0, BSpline(Linear()))
    itp_scaled = scale(itp_inner, r0_itp)
    function itp(x)
        if x < first(r0)
            return 3.14159
        elseif x > last(r0)
            return 0
        else
            return itp_scaled(x)
        end
    end

    println()
    println("#--------------------------------------------------#")
    println()
    println("Radial function interpolation")
    println()

    @showprogress 1 "Computing..." for r_idx in 1:length(r_vals)
        
        matrix_f_plus = zeros(Float64, l1,l2,l3); matrix_f_minus = zeros(Float64, l1,l2,l3)

        r = r_vals[r_idx]
        
        @inbounds @fastmath for k in 1:l3, j in 1:l2, i in 1:l1
            temp_f_plus = itp(norm([y1[i],y2[j],y3[r_idx,k]] .+ [0.,0.,r/2]) ) 
            temp_f_minus = itp(norm([y1[i],y2[j],y3[r_idx,k]] .- [0.,0.,r/2]) )
            
            matrix_f_plus[i,j,k] = temp_f_plus
            matrix_f_minus[i,j,k] = temp_f_minus
        end 

        #----- data saving

        if output_format == "jld2"
            path1 = out*"/f_$(model)_plus_r=$(r_idx).jld2"; path2 = out*"/f_$(model)_minus_r=$(r_idx).jld2"
            @save path1 matrix_f_plus; @save path2 matrix_f_minus
        elseif output_format == "npy"
            npzwrite(out*"/f_$(model)_plus_r=$(r_idx).npy", matrix_f_plus); npzwrite(out*"/f_$(model)_minus_r=$(r_idx).npy", matrix_f_minus)
        elseif output_format == "jls"
            open(out*"/f_$(model)_plus_r=$(r_idx).jls", "w") do io
                serialize(io, matrix_f_plus)
            end
            open(out*"/f_$(model)_minus_r=$(r_idx).jls", "w") do io
                serialize(io, matrix_f_minus)
            end
        end

    end
        
    println()
    println("data saved at "*out )
    println()
    println("#--------------------------------------------------#")
end

#####

function interp_2sky_dx_proy(rtc,r_vals, model::String,deriv::String,hD::Float64, out::String,output_format::String)

    data = readdlm("/lustre/HQCD/victor.diaz/nnp/data/profile_f/profile_f_$(model).txt")
    r0 = data[:,1]; f0 = data[:,2]

    y1 = rtc[1]; y2 = rtc[2]; y3 = rtc[3]
    l1 = length(y1); l2 = length(y2); l3 = length(y3[1,:])

    #----- main loop

    r0_itp = first(r0):0.01:last(r0)

    itp_inner = interpolate(f0, BSpline(Linear()))
    itp_scaled = scale(itp_inner, r0_itp)
    function itp(x)
        if x < first(r0)
            return 3.14159
        elseif x > last(r0)
            return 0
        else
            return itp_scaled(x)
        end
    end    

    println()
    println("#--------------------------------------------------#")
    println()
    println("Radial function interpolation --- $(deriv) direction")
    
    
    for r_idx in 1:length(r_vals)
        
        matrix_f_minus_m = zeros(Float64, l1,l2,l3); matrix_f_minus_p = zeros(Float64, l1,l2,l3)
        matrix_f_plus_m = zeros(Float64, l1,l2,l3); matrix_f_plus_p = zeros(Float64, l1,l2,l3)

        println()

        r = r_vals[r_idx]
   
        if deriv == "x1"
            x_p = [hD,0.,r]./2; x_m = [-hD,0.,r]./2
        elseif deriv == "x2"
            x_p = [0.,hD,r]./2; x_m = [0.,-hD,r]./2
        elseif deriv == "x3"
            x_p = [0.,0.,hD+r]./2; x_m = [0.,0.,-hD+r]./2
        end

        @showprogress 1 "Computing r=$(r_idx):" for k in 1:l3
            @inbounds @fastmath for j in 1:l2
                for i in 1:l1
                    temp_f_plus_p = itp(norm([y1[i],y2[j],y3[r_idx,k]] .+ x_p) )
                    temp_f_plus_m = itp(norm([y1[i],y2[j],y3[r_idx,k]] .+ x_m) )
                    temp_f_minus_p = itp(norm([y1[i],y2[j],y3[r_idx,k]] .- x_p) )
                    temp_f_minus_m = itp(norm([y1[i],y2[j],y3[r_idx,k]] .- x_m) )

                    matrix_f_plus_p[i,j,k] = temp_f_plus_p
                    matrix_f_plus_m[i,j,k] = temp_f_plus_m
                    matrix_f_minus_p[i,j,k] = temp_f_minus_p
                    matrix_f_minus_m[i,j,k] = temp_f_minus_m
                end
            end
        end
        
        #----- data saving

        println()
        println("Saving data...")

        if output_format == "jld2"
            path1 = out*"/f_$(model)_$(deriv)_plus_p_r=$(r_idx).jld2"; path2 = out*"/f_$(model)_$(deriv)_plus_m_r=$(r_idx).jld2"
            path3 = out*"/f_$(model)_$(deriv)_minus_p_r=$(r_idx).jld2"; path4 = out*"/f_$(model)_$(deriv)_minus_m_r=$(r_idx).jld2"
            @save path1 matrix_f_plus_p; @save path2 matrix_f_plus_m; @save path3 matrix_f_minus_p; @save path4 matrix_f_minus_m; 

        elseif output_format == "npy"
            npzwrite(out*"/f_$(model)_$(deriv)_plus_p_r=$(r_idx).npy", matrix_f_plus_p); npzwrite(out*"/f_$(model)_$(deriv)_plus_m_r=$(r_idx).npy", matrix_f_plus_m)
            npzwrite(out*"/f_$(model)_$(deriv)_minus_p_r=$(r_idx).npy", matrix_f_minus_p); npzwrite(out*"/f_$(model)_$(deriv)_minus_m_r=$(r_idx).npy", matrix_f_minus_m)
         elseif output_format == "jls"
            open(out*"/f_$(model)_$(deriv)_plus_p_r=$(r_idx).jls", "w") do io
                serialize(io, matrix_f_plus_p)
            end
            open(out*"/f_$(model)_$(deriv)_plus_m_r=$(r_idx).jls", "w") do io
                serialize(io, matrix_f_plus_m)
            end
            open(out*"/f_$(model)_$(deriv)_minus_p_r=$(r_idx).jls", "w") do io
                serialize(io, matrix_f_minus_p)
            end
            open(out*"/f_$(model)_$(deriv)_minus_m_r=$(r_idx).jls", "w") do io
                serialize(io, matrix_f_minus_m)
            end

        end
    end
    
    println()
    println("Data saved at "*out )
    println()
    println("#--------------------------------------------------#")
end

#####

function interp_2sky_dy_proy(rtc,r_vals, model::String,deriv::String,hD::Float64, out::String,output_format::String)

    data = readdlm("/lustre/HQCD/victor.diaz/nnp/data/profile_f/profile_f_$(model).txt")
    r0 = data[:,1]; f0 = data[:,2]

    y1 = rtc[1]; y2 = rtc[2]; y3 = rtc[3]
    l1 = length(y1); l2 = length(y2); l3 = length(y3[1,:])

    #----- main loop

    r0_itp = first(r0):0.01:last(r0)

    itp_inner = interpolate(f0, BSpline(Linear()))
    itp_scaled = scale(itp_inner, r0_itp)
    function itp(x)
        if x < first(r0)
            return 3.14159
        elseif x > last(r0)
            return 0
        else
            return itp_scaled(x)
        end
    end    

    println()
    println("#--------------------------------------------------#")
    println()
    println("Radial function interpolation --- $(deriv) direction")
    
    
    for r_idx in 1:length(r_vals)
        
        matrix_f_minus_m = zeros(Float64, l1,l2,l3); matrix_f_minus_p = zeros(Float64, l1,l2,l3)
        matrix_f_plus_m = zeros(Float64, l1,l2,l3); matrix_f_plus_p = zeros(Float64, l1,l2,l3)

        println()

        r = r_vals[r_idx]
   
        if deriv == "y1"
            y1_p = y1 .+ hD; y1_m = y1 .- hD
            y2_p = y2; y2_m = y2
            y3_p = y3[r_idx,:]; y3_m = y3[r_idx,:]
        elseif deriv == "y2"
            y1_p = y1; y1_m = y1
            y2_p = y2 .+ hD; y2_m = y2 .- hD
            y3_p = y3[r_idx,:]; y3_m = y3[r_idx,:] 
        elseif deriv == "y3"
            y1_p = y1; y1_m = y1
            y2_p = y2; y2_m = y2
            y3_p = y3[r_idx,:] .+ hD; y3_m = y3[r_idx,:] .- hD
        end

        x = [0,0,r]/2.

        @showprogress 1 "Computing r=$(r_idx):" for k in 1:l3
            @inbounds @fastmath for j in 1:l2
                for i in 1:l1
                    temp_f_plus_p = itp(norm([y1_p[i],y2_p[j],y3_p[k]] .+ x) )
                    temp_f_plus_m = itp(norm([y1_m[i],y2_m[j],y3_m[k]] .+ x) )
                    temp_f_minus_p = itp(norm([y1_p[i],y2_p[j],y3_p[k]] .- x) )
                    temp_f_minus_m = itp(norm([y1_m[i],y2_m[j],y3_m[k]] .- x) )

                    matrix_f_plus_p[i,j,k] = temp_f_plus_p
                    matrix_f_plus_m[i,j,k] = temp_f_plus_m
                    matrix_f_minus_p[i,j,k] = temp_f_minus_p
                    matrix_f_minus_m[i,j,k] = temp_f_minus_m
                end
            end
        end
        
        #----- data saving

        println()
        println("Saving data...")

        if output_format == "jld2"
            path1 = out*"/f_$(model)_$(deriv)_plus_p_r=$(r_idx).jld2"; path2 = out*"/f_$(model)_$(deriv)_plus_m_r=$(r_idx).jld2"
            path3 = out*"/f_$(model)_$(deriv)_minus_p_r=$(r_idx).jld2"; path4 = out*"/f_$(model)_$(deriv)_minus_m_r=$(r_idx).jld2"
            @save path1 matrix_f_plus_p; @save path2 matrix_f_plus_m; @save path3 matrix_f_minus_p; @save path4 matrix_f_minus_m; 

        elseif output_format == "npy"
            npzwrite(out*"/f_$(model)_$(deriv)_plus_p_r=$(r_idx).npy", matrix_f_plus_p); npzwrite(out*"/f_$(model)_$(deriv)_plus_m_r=$(r_idx).npy", matrix_f_plus_m)
            npzwrite(out*"/f_$(model)_$(deriv)_minus_p_r=$(r_idx).npy", matrix_f_minus_p); npzwrite(out*"/f_$(model)_$(deriv)_minus_m_r=$(r_idx).npy", matrix_f_minus_m)
         elseif output_format == "jls"
            open(out*"/f_$(model)_$(deriv)_plus_p_r=$(r_idx).jls", "w") do io
                serialize(io, matrix_f_plus_p)
            end
            open(out*"/f_$(model)_$(deriv)_plus_m_r=$(r_idx).jls", "w") do io
                serialize(io, matrix_f_plus_m)
            end
            open(out*"/f_$(model)_$(deriv)_minus_p_r=$(r_idx).jls", "w") do io
                serialize(io, matrix_f_minus_p)
            end
            open(out*"/f_$(model)_$(deriv)_minus_m_r=$(r_idx).jls", "w") do io
                serialize(io, matrix_f_minus_m)
            end

        end
    end
    
    println()
    println("Data saved at "*out )
    println()
    println("#--------------------------------------------------#")
end

##########

function cubic_spline(x, y)
    n = length(x)
    h = diff(x)

    q = zeros(n)
    for i in 2:n-1
        q[i] = (3 / h[i]) * (y[i+1] - y[i]) -
               (3 / h[i-1]) * (y[i] - y[i-1])
    end

    l = ones(n)
    m = zeros(n)
    z = zeros(n)

    for i in 2:n-1
        l[i] = 2 * (x[i+1] - x[i-1]) - h[i-1] * m[i-1]
        m[i] = h[i] / l[i]
        z[i] = (q[i] - h[i-1] * z[i-1]) / l[i]
    end

    c = zeros(n)
    b = zeros(n-1)
    d = zeros(n-1)
    a = copy(y)

    for j in n-1:-1:1
        c[j] = z[j] - m[j] * c[j+1]
        b[j] = (a[j+1] - a[j]) / h[j] -
               h[j] * (c[j+1] + 2*c[j]) / 3
        d[j] = (c[j+1] - c[j]) / (3 * h[j])
    end

    return a,b,c,d,x
end


function spline_eval(spline, r)
    a,b,c,d,x = spline
    n = length(x)

    # outside boundaries
    if r < x[1]
        return 3.14159
    elseif r > x[end]
		return 0.0
	end

    # find interval
    i = n-1
    for j in 1:n-1
        if x[j] <= r <= x[j+1]
            i = j
            break
        end
    end

    dx = r - x[i]
    return a[i] + b[i]*dx + c[i]*dx^2 + d[i]*dx^3
end

function interp_df_2sky_no_proy(rtc,r_vals::Array{Float64}, model::String,data, out::String,output_format::String)

    r0 = data[:,1]; f0 = data[:,2]

    y1 = rtc[1]; y2 = rtc[2]; y3 = rtc[3]
    l1 = length(y1); l2 = length(y2); l3 = size(y3,ndims(y3))

	h = 0.0001

	spline = cubic_spline(r0,f0)

    #----- main loop	
 
    println()
    println("#--------------------------------------------------#")
    println()
    println("Radial function interpolation")
    println()

    @showprogress 1 "Computing..." for r_idx in 1:length(r_vals)
        
        matrix_df_plus = zeros(Float64, l1,l2,l3); matrix_df_minus = zeros(Float64, l1,l2,l3)

        r = r_vals[r_idx]
 
        @inbounds for k in 1:l3, j in 1:l2, i in 1:l1
			Rp = sqrt(y1[i]^2 + y2[j]^2 + (y3[r_idx,k]+r/2)^2)
			Rm = sqrt(y1[i]^2 + y2[j]^2 + (y3[r_idx,k]-r/2)^2)
			
			temp_df_plus = (spline_eval(spline, Rp+h) - spline_eval(spline, Rp-h))/(2*h)
            temp_df_minus = (spline_eval(spline, Rm+h) - spline_eval(spline, Rm-h))/(2*h)
            
            matrix_df_plus[i,j,k] = temp_df_plus
            matrix_df_minus[i,j,k] = temp_df_minus
		end

        #----- data saving

        if output_format == "jld2"
            path1 = out*"/df_$(model)_plus_r=$(r_idx).jld2"; path2 = out*"/df_$(model)_minus_r=$(r_idx).jld2"
            @save path1 matrix_df_plus; @save path2 matrix_df_minus
        elseif output_format == "npy"
            npzwrite(out*"/df_$(model)_plus_r=$(r_idx).npy", matrix_df_plus); npzwrite(out*"/df_$(model)_minus_r=$(r_idx).npy", matrix_df_minus)
        elseif output_format == "jls"
            open(out*"/df_$(model)_plus_r=$(r_idx).jls", "w") do io
                serialize(io, matrix_df_plus)
            end
            open(out*"/df_$(model)_minus_r=$(r_idx).jls", "w") do io
                serialize(io, matrix_df_minus)
            end
        end

    end

end


