#----- pkg

using LinearAlgebra
using Serialization
using JLD2
using NPZ

#----- r values

function make_r(out::String,output_format::String)

    if (output_format != "jld2") && (output_format != "npy") && (output_format != "jls")
        println("invalid output data type")
        return
    end

    #----- r values

    r_vals = collect(1.731:0.1:7.731)

    #----- data saving

    if output_format == "jld2"
        path = out*"/r_vals.jld2"
        @save path r_vals

    elseif output_format == "npy"
        npzwrite(out*"/r_vals.npy", r_vals)

    elseif output_format == "jls"
        open(out*"/r_vals.jls", "w") do io
            serialize(io, r_vals)
        end
    end

    println()
    println("data saved at "*out )
    println()

    return r_vals
end

#----- Q values
#=
function make_Q(out::String,output_format::String)

    if (output_format != "jld2") && (output_format != "npy")
        println("invalid output data type")
        return
    end

    #----- Q values

    vals = [-1./2, 1./2]

    pr_1 = Float64[]
    pr_2 = Float64[]
    pr_3 = Float64[]
    pr_4 = Float64[]

    uvw_comb = Float64[]
    uvw_str = Float64[]

    for u in vals
        for v in vals
            for w in vals
                ell =  1./sqrt(1 + u^2 + v^2 + w^2)

                push!(pr_1, ell .* [1,u,v,w])
                push!(uvw_comb, [1,u,v,w])
                push!(uvw_str, ["1","u","v","w"])
=#
##### y sampling

function make_grid_reg(l1::Int64,l2::Int64,l3::Int64,step::Float64, out::String,output_format::String)

    if (output_format != "jld2") && (output_format != "npy") && (output_format != "jls")
        println("invalid output data type")
        return
    end
    
    #-----

    yt1 = collect(-l1/2:step:l1/2)
    yt2 = collect(-l2/2:step:l2/2)
    yt3 = collect(-l3/2:step:l3/2)

    dyt1 = yt1[2]-yt1[1]
    dyt2 = yt2[2]-yt2[1]
    dyt3 = yt3[2]-yt3[1]

    dr = 0.1

    #----- main
    
    println()
    println("#--------------------------------------------------#")
    println()
    println("Grid generation")
    println()

    y1 = yt1
    y2 = yt2
    y3 = yt3

    print("done: reg")

    #----- data saving

    if output_format == "jld2"
        path = out*"/reg_$(l1)x$(l2)x$(l3)/grid.jld2"
        @save path y1,y2,y3

    elseif output_format == "npy"
        npzwrite(out*"/reg_$(l1)x$(l2)x$(l3)/y1.npy", y1)
        npzwrite(out*"/reg_$(l1)x$(l2)x$(l3)/y2.npy", y2)
        npzwrite(out*"/reg_$(l1)x$(l2)x$(l3)/y3.npy", y3)
    elseif output_format == "jls"
        open(out*"/reg_$(l1)x$(l2)x$(l3)/y1.jls", "w") do io; serialize(io, y1); end
        open(out*"/reg_$(l1)x$(l2)x$(l3)/y2.jls", "w") do io; serialize(io, y2); end
        open(out*"/reg_$(l1)x$(l2)x$(l3)/y3.jls", "w") do io; serialize(io, y3); end
    end

    println()
    println("data saved at "*out*"/reg_$(l1)x$(l2)x$(l3)" )
    println()
    println("#--------------------------------------------------#")

    println()
    println("#--------------------------------------------------#")
    println()
    println("Differentials generation")
    println()

    dy1 = [step for it in length(yt1)]
    dy2 = [step for it in length(yt2)]
    dy3 = [step for it in length(yt3)]

    print("done: reg")

    #----- data saving
    
    if output_format == "jld2"
        path = out*"/reg_$(l1)x$(l2)x$(l3)/dy.jld2"
        @save path dy1,dy2,dy3

    elseif output_format == "npy"
        npzwrite(out*"/reg_$(l1)x$(l2)x$(l3)/dy1.npy", dy1)
        npzwrite(out*"/reg_$(l1)x$(l2)x$(l3)/dy2.npy", dy2)
        npzwrite(out*"/reg_$(l1)x$(l2)x$(l3)/dy3.npy", dy3)
    elseif output_format == "jls"
        open(out*"/reg_$(l1)x$(l2)x$(l3)/dy1.jls", "w") do io; serialize(io, dy1); end
        open(out*"/reg_$(l1)x$(l2)x$(l3)/dy2.jls", "w") do io; serialize(io, dy2); end
        open(out*"/reg_$(l1)x$(l2)x$(l3)/dy3.jls", "w") do io; serialize(io, dy3); end
    end

    println()
    println("data saved at "*out*"/reg_$(l1)x$(l2)x$(l3)" )
    println()
    println("#--------------------------------------------------#")

end

function make_grid_proy(p1::Int64,p2::Int64,p3::Int64,r_vals::Array{Float64}, out::String,output_format::String)

    if (output_format != "jld2") && (output_format != "npy") && (output_format != "jls")
        println("invalid output data type")
        return
    end
    
    #-----
   
    dyt1 = 2. /p1
    dyt2 = 2. /p2
    dyt3 = 2. /p3

    dr = 0.1

    yt1 = collect(-1+dyt1:dyt1:1-dyt1)
    yt2 = collect(-1+dyt2:dyt2:1-dyt2)
    yt3 = collect(-1+dyt3:dyt3:1-dyt3)

    #----- main
     
    println()
    println("#--------------------------------------------------#")
    println()
    println("Generating grid...")
    println()

    y1 = zeros(Float64, length(yt1))
    for i in 1:length(yt1)    
        y1[i] = yt1[i]/(1-yt1[i]^2)
    end

    y2 = zeros(Float64, length(yt2))
    for i in 1:length(yt2)
        y2[i] = yt2[i]/(1-yt2[i]^2)
    end

    y3 = zeros(Float64, length(r_vals),length(yt3))
    for (r_idx,r) in enumerate(r_vals)
        for i in 1:length(yt3)
            sec1 = r/2. + (2*abs(yt3[i])-1)/(8*(abs(yt3[i])-1)^2)
            sec2 = abs(yt3[i])*(2*r*(1-abs(yt3[i]))-(1-2*abs(yt3[i])))

            if yt3[i] >= 1/2.
                y3[r_idx,i] = sec1
            elseif (0.0 < yt3[i]) && (yt3[i] < 1/2.)
                y3[r_idx,i] = sec2
            elseif (-1/2. < yt3[i]) && (yt3[i] < 0.0)
                y3[r_idx,i] = -sec2
            elseif yt3[i] <= -1/2.
                y3[r_idx,i] = -sec1
            end
        end
    end
 
    #----- data saving

    if output_format == "jld2"
        path = out*"/proy_$(p1)x$(p2)x$(p3)/grid.jld2"
        @save path y1,y2,y3

    elseif output_format == "npy"
        npzwrite(out*"/proy_$(p1)x$(p2)x$(p3)/y1.npy", y1)
        npzwrite(out*"/proy_$(p1)x$(p2)x$(p3)/y2.npy", y2)
        npzwrite(out*"/proy_$(p1)x$(p2)x$(p3)/y3.npy", y3)
    elseif output_format == "jls"
        open(out*"/proy_$(p1)x$(p2)x$(p3)/y1.jls", "w") do io; serialize(io, y1); end
        open(out*"/proy_$(p1)x$(p2)x$(p3)/y2.jls", "w") do io; serialize(io, y2); end
        open(out*"/proy_$(p1)x$(p2)x$(p3)/y3.jls", "w") do io; serialize(io, y3); end
    end

    println()
    println("data saved at "*out*"/proy_$(p1)x$(p2)x$(p3)" )
    println()
    println("#--------------------------------------------------#")

    println()
    println("#--------------------------------------------------#")
    println()
    println("Generating differentials...")
    println()

    dy1 = zeros(Float64, length(yt1))
	for i in 1:length(yt1)
        dy1[i] = ((1+yt1[i]^2)/(1-yt1[i]^2)^2)*dyt1
    end
    
    dy2 = zeros(Float64, length(yt2))
	for i in 1:length(yt2)
        dy2[i] = ((1+yt2[i]^2)/(1-yt2[i]^2)^2)*dyt2
    end

    dy3_all = zeros(Float64, length(r_vals),length(yt3))
    for (r_idx,r) in enumerate(r_vals)
	    for i in 1:length(yt3)
	        if yt3[i] >= 1/2.
                dy3_all[r_idx,i] = dr/2. - (yt3[i]*dyt3)/(4*(yt3[i]-1)^3)
            elseif (yt3[i] < 1/2.) && (yt3[i] >= 0.)
                dy3_all[r_idx,i] = -2*(yt3[i]-1)*yt3[i]*dr + (4*yt3[i] + r*(2-4*yt3[i]) - 1)*dyt3
            elseif (yt3[i] > -1/2.) && (yt3[i] < 0)
                dy3_all[r_idx,i] = -2*(abs(yt3[i])-1)*abs(yt3[i])*dr + (4*abs(yt3[i]) + r*(2-4*abs(yt3[i])) - 1)*dyt3
		    elseif yt3[i] <= -1/2.
                dy3_all[r_idx,i] = dr/2. - (abs(yt3[i])*dyt3)/(4*(abs(yt3[i])-1)^3)
            end
        end
    end

    print("done: proy")

    #----- data saving

    if output_format == "jld2"
        path = out*"/proy_$(p1)x$(p2)x$(p3)/diffs.jld2"
        @save path dy1,dy2,dy3_all

    elseif output_format == "npy"
        npzwrite(out*"/proy_$(p1)x$(p2)x$(p3)/dy1.npy", dy1)
        npzwrite(out*"/proy_$(p1)x$(p2)x$(p3)/dy2.npy", dy2)
        npzwrite(out*"/proy_$(p1)x$(p2)x$(p3)/dy3.npy", dy3_all)
    elseif output_format == "jls"
        open(out*"/proy_$(p1)x$(p2)x$(p3)/dy1.jls", "w") do io; serialize(io, dy1); end
        open(out*"/proy_$(p1)x$(p2)x$(p3)/dy2.jls", "w") do io; serialize(io, dy2); end
        open(out*"/proy_$(p1)x$(p2)x$(p3)/dy3.jls", "w") do io; serialize(io, dy3_all); end
    end

    println()
    println("data saved at "*out*"/proy_$(p1)x$(p2)x$(p3)" )
    println()
    println("#--------------------------------------------------#")

end
