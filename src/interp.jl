#----- pkg

using Serialization
using JLD2
using NPZ
using LinearAlgebra
using SpecialFunctions
using ProgressMeter
using DelimitedFiles
using Interpolations
using OrdinaryDiffEq

#----- asymptotic f

using OrdinaryDiffEq, DelimitedFiles

# p = (m=..., c6=...); std is m=0, c6=0
function rhs!(du, u, p, r)
    f, fp = u
    s, c = sin(f), cos(f)
    A = r^2 + 2s^2 + p.c6*s^4/r^2
    B = 2r*fp + sin(2f)*(fp^2 - 1 - s^2/r^2) - p.m^2*r^2*s +
        p.c6*(2*s^3*c*fp^2/r^2 - 2*s^4*fp/r^3)
    du[1] = fp
    du[2] = -B/A
end

function model_setup(model; m=0.0, c6=0.0)
    if model == "std"
		m = 0.0; c6 = 0.0
    elseif model == "mss" 
		c6 = 0.0
    elseif model == "sxt"
		m = 0.0
    else 
		error("unknown model $model") 
	end
    
	p = (m=m, c6=c6)
    if m == 0
        r0   = 200.0
        u0   = C -> [C/r0^2, -2C/r0^3]
        tail = (C,x) -> C/x^2
    else
        r0   = 18/m
        u0   = C -> [C*exp(-m*r0)*(m/r0 + 1/r0^2),
                    -C*exp(-m*r0)*(m*(m/r0 + 1/r0^2) + m/r0^2 + 2/r0^3)]
        tail = (C,x) -> C*exp(-m*x)*(m/x + 1/x^2)
    end
    return (p=p, r0=r0, u0=u0, tail=tail)
end

function f_fit(model::String; m=0.0, c6=0.0, bracket=nothing, outdir="/home/vddiazz/Desktop")
    S = model_setup(model; m=m, c6=c6)
    r0 = S.r0
    flag = Ref(0)
    cb_over = ContinuousCallback((u,t,i)->u[1]-pi, i->(flag[]=+1; terminate!(i)))
    cb_turn = ContinuousCallback((u,t,i)->u[2],    i->(flag[]=-1; terminate!(i)))
    cbs = CallbackSet(cb_over, cb_turn)

    function shoot(C; dense=false)
        flag[] = 0
        prob = ODEProblem(rhs!, S.u0(C), (r0, 1e-6), S.p)
        sol = solve(prob, Vern9(); reltol=1e-13, abstol=1e-16, callback=cbs, dense=dense)
        return flag[], sol
    end

    # automatic bracket: scan C until the flag changes from -1 to +1
    function find_bracket()
        Cs = 10 .^ range(-1, 2, length=80)
        fl = [shoot(C)[1] for C in Cs]
        println("flags along the scan: ", fl)
        i = findfirst(k -> fl[k] == -1 && fl[k+1] == +1, 1:length(Cs)-1)
        i === nothing && error("no bracket found; widen the scan or check the flags")
        return Cs[i], Cs[i+1]
    end

    function bisect_C(lo, hi)
        flo,_ = shoot(lo); fhi,_ = shoot(hi)
        println("flag(lo)=$flo, flag(hi)=$fhi (expected -1, +1)")
        @assert flo == -1 && fhi == +1 "adjust the bracket"
        for _ in 1:200
            mid = (lo+hi)/2
            fm,_ = shoot(mid)
            fm == +1 ? (hi = mid) : (lo = mid)
            hi-lo < 1e-14 && break
        end
        return (lo+hi)/2
    end

    lo, hi = bracket === nothing ? find_bracket() : bracket
    C = bisect_C(lo, hi)
    _, sol = shoot(C; dense=true)
    rs = sol.t[end]
    println("C = $C,  r_stop = $rs")

    grid = collect(0.001:0.01:r0); grid = grid[grid .>= rs]
    ftab = [sol(r)[1] for r in grid]
    writedlm("$outdir/profile_$(model).dat", hcat(grid, ftab))
    open("$outdir/profile_$(model)_params.txt","w") do io
        println(io, "model = $model  m = $(S.p.m)  c6 = $(S.p.c6)")
        println(io, "C = $C  r0 = $r0  rs = $rs  f_s = $(sol(rs)[1])  fp_s = $(sol(rs)[2])")
    end
    return C, S, grid, ftab
end

#----- interp

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

function interp_2sky_no_proy(rtc,r_vals::Array{Float64}, model::String, out::String,output_format::String; m=0, c6=0)

	C, S, rgrid, ftab = f_fit(model; m=m, c6=c6, outdir=out)
	
	r0 = rgrid
	f0 = ftab
	mtail = S.p.m

    y1 = rtc[1]; y2 = rtc[2]; y3 = rtc[3]
    l1 = length(y1); l2 = length(y2); l3 = length(y3[1,:])

    #----- main loop
 
    r0_axis    = range(first(r0), last(r0), length=length(r0))
    itp_scaled = scale(interpolate(f0, BSpline(Cubic(Line(OnGrid())))), r0_axis)
    dfitp(x) = Interpolations.gradient(itp_scaled, x)[1]

    rho_s   = first(r0)
    f_s   = f0[1]
    fp_s  = (f0[2] - f0[1]) / (r0[2] - r0[1])     # slope at the inner end (≈ -π/…, negative)
    rho_max = last(r0)
    
    g(x)  = mtail == 0 ? 1/x^2 : exp(-mtail*x) * (mtail/x + 1/x^2)
    
    dg(x) = mtail == 0 ? -2/x^3 : -exp(-mtail*x) * (mtail*(mtail/x + 1/x^2) + mtail/x^2 + 2/x^3)

	Ctail = f0[end]/g(rho_max)

    function itp(x::Float64)
        if x < rho_s
            return min(Float64(pi), f_s + fp_s*(x - rho_s))   # linear continuation toward the origin
        elseif x > rho_max
            return Ctail*g(x)                            # massless tail, never 0
        else
            return itp_scaled(x)
        end
    end

    function ditp(x::Float64)
        if x < rho_s
            return f_s + fp_s*(x - rho_s) > Float64(pi) ? 0.0 : fp_s
        elseif x > rho_max
            return Ctail * dg(x)
        else
            return dfitp(x)
        end
    end

    println()
    println("#--------------------------------------------------#")
    println()
    println("Radial function interpolation")
    println()

    @showprogress 1 "Computing..." for r_idx in 1:length(r_vals)
        
        matrix_f_plus = zeros(Float64, l1,l2,l3)
        matrix_f_minus = zeros(Float64, l1,l2,l3)
        matrix_df_plus = zeros(Float64, l1,l2,l3)
        matrix_df_minus = zeros(Float64, l1,l2,l3)

        r = r_vals[r_idx]

        @inbounds for k in 1:l3
            zp = y3[r_idx,k] + r/2          # distance along z to the skyrmion at -r/2 (plus)
            zm = y3[r_idx,k] - r/2          # distance along z to the skyrmion at +r/2 (minus)
            for j in 1:l2, i in 1:l1
                s  = y1[i]^2 + y2[j]^2
                rp = sqrt(s + zp^2)
                rm = sqrt(s + zm^2)

                matrix_f_plus[i,j,k]   = itp(rp)
                matrix_f_minus[i,j,k]  = itp(rm)
                matrix_df_plus[i,j,k]  = ditp(rp)
                matrix_df_minus[i,j,k] = ditp(rm)
            end
        end
        
        #----- data saving

        if output_format == "jld2"
            #path1 = out*"/f_$(model)_plus_r=$(r_idx).jld2"; path2 = out*"/f_$(model)_minus_r=$(r_idx).jld2"
            #@save path1 matrix_f_plus; @save path2 matrix_f_minus
        elseif output_format == "npy"
            #npzwrite(out*"/f_$(model)_plus_r=$(r_idx).npy", matrix_f_plus); npzwrite(out*"/f_$(model)_minus_r=$(r_idx).npy", matrix_f_minus)
        elseif output_format == "jls"
            open(out*"/f_$(model)_plus_r=$(r_idx).jls", "w") do io
                serialize(io, matrix_f_plus)
            end
            open(out*"/f_$(model)_minus_r=$(r_idx).jls", "w") do io
                serialize(io, matrix_f_minus)
            end
            open(out*"/df_$(model)_plus_r=$(r_idx).jls", "w") do io
                serialize(io, matrix_df_plus)
            end
            open(out*"/df_$(model)_minus_r=$(r_idx).jls", "w") do io
                serialize(io, matrix_df_minus)
            end
        end

    end
        
    println()
    println("data saved at "*out )
    println()
    println("#--------------------------------------------------#")
end

#####

function interp_1sky_no_proy(rtc,r_vals::Array{Float64}, model::String,data, out::String,output_format::String)

    r0 = data[:,1]; f0 = data[:,2]

    y1 = rtc[1]; y2 = rtc[2]; y3 = rtc[3]
    l1 = length(y1); l2 = length(y2); l3 = length(y3[1,:])

    #----- main loop

    r0_axis    = range(first(r0), last(r0), length=length(r0))
    itp_scaled = scale(interpolate(f0, BSpline(Cubic(Line(OnGrid())))), r0_axis)

    rho_s   = first(r0)
    f_s   = f0[1]
    fp_s  = (f0[2] - f0[1]) / (r0[2] - r0[1])     # slope at the inner end (≈ -π/…, negative)
    rho_max = last(r0)
    Cinf  = f0[end] * rho_max^2                     # fixed by the table, so the join is continuous

    function itp(x::Float64)
        if x < rho_s
            return min(Float64(pi), f_s + fp_s*(x - rho_s))   # linear continuation toward the origin
        elseif x > rho_max
            return Cinf / x^2                              # massless tail, never 0
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
        
        matrix_f = zeros(Float64, l1,l2,l3)

        r = r_vals[r_idx]
        
        @inbounds @fastmath for k in 1:l3, j in 1:l2, i in 1:l1
            rho = norm([y1[i],y2[j],y3[r_idx,k]] .+ [0.,0.,r/2])
            matrix_f[i,j,k] = itp(rho)
        end 

        #----- data saving

        if output_format == "jld2"
        	#
		elseif output_format == "npy"
        	#
		elseif output_format == "jls"
            open(out*"/f_$(model)_hdg_r=$(r_idx).jls", "w") do io
                serialize(io, matrix_f)
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

function interp_2sky_dy_proy(rtc,hDps,hDms,r_vals, model::String,data,deriv::String, out::String,output_format::String)

    r0 = data[:,1]; f0 = data[:,2]

    y1 = rtc[1]; y2 = rtc[2]; y3 = rtc[3]
	hD_p1 = hDps[1]; hD_p2 = hDps[2]; hD_p3 = hDps[3]
	hD_m1 = hDms[1]; hD_m2 = hDms[2]; hD_m3 = hDms[3]    

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
            y1_p = hD_p1; y1_m = hD_m1
            y2_p = y2; y2_m = y2
            y3_p = y3[r_idx,:]; y3_m = y3[r_idx,:]
        elseif deriv == "y2"
            y1_p = y1; y1_m = y1
            y2_p = hD_p2; y2_m = hD_m2
            y3_p = y3[r_idx,:]; y3_m = y3[r_idx,:] 
        elseif deriv == "y3"
            y1_p = y1; y1_m = y1
            y2_p = y2; y2_m = y2
            y3_p = hD_p3[r_idx,:]; y3_m = hD_m3[r_idx,:]
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
#==
function interp_df_2sky_no_proy(rtc,r_vals::Array{Float64}, grid_size::String,model::String,data, out::String,output_format::String)

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

		#----- interp missing NaN

		if grid_size == "proy_80x80x160"
			matrix_df_minus[40,40,40] = matrix_df_minus[40,40,39]
			matrix_df_plus[40,40,120] = matrix_df_plus[40,40,119] 
		elseif grid_size == "proy_90x90x180"
			matrix_df_minus[45,45,45] = matrix_df_minus[45,45,44]
			matrix_df_plus[45,45,135] = matrix_df_plus[45,45,134]
		elseif grid_size == "proy_100x100x200"
			matrix_df_minus[50,50,50] = matrix_df_minus[50,50,49]
			matrix_df_plus[50,50,150] = matrix_df_plus[50,50,149]
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
==#
