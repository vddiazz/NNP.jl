#----- pkg

using LinearAlgebra
using JLD2
using NPZ
using ProgressMeter
using LoopVectorization

#### baryon number

function b_dens(U::Array{Float64},d1U::Array{Float64},d2U::Array{Float64},d3U::Array{Float64},out::String, output_format::String)

    l1 = length(U[:,1,1,1]); l2 = length(U[1,:,1,1]); l3 = length(U[1,1,:,1])

    #----- main
    
    dens = zeros(Float64, l1,l2,l3)

    @showprogress 1 "Computing..." for k in 1:l3
        @tturbo for j in 1:l2, i in 1:l1
            dens[i,j,k] = -12*(U[i,j,k,1]^2 + U[i,j,k,2]^2 + U[i,j,k,3]^2 + U[i,j,k,4]^2)*(d1U[i,j,k,2]*d2U[i,j,k,4]*d3U[i,j,k,3]*U[i,j,k,1] - d1U[i,j,k,2]*d2U[i,j,k,3]*d3U[i,j,k,4]*U[i,j,k,1] - d1U[i,j,k,1]*d2U[i,j,k,4]*d3U[i,j,k,3]*U[i,j,k,2] + d1U[i,j,k,1]*d2U[i,j,k,3]*d3U[i,j,k,4]*U[i,j,k,2] - d1U[i,j,k,2]*d2U[i,j,k,4]*d3U[i,j,k,1]*U[i,j,k,3] + d1U[i,j,k,1]*d2U[i,j,k,4]*d3U[i,j,k,2]*U[i,j,k,3] + d1U[i,j,k,2]*d2U[i,j,k,1]*d3U[i,j,k,4]*U[i,j,k,3] - d1U[i,j,k,1]*d2U[i,j,k,2]*d3U[i,j,k,4]*U[i,j,k,3] + d1U[i,j,k,4]*(d3U[i,j,k,3]*(-(d2U[i,j,k,2]*U[i,j,k,1]) + d2U[i,j,k,1]*U[i,j,k,2]) + d2U[i,j,k,3]*(d3U[i,j,k,2]*U[i,j,k,1] - d3U[i,j,k,1]*U[i,j,k,2]) + (d2U[i,j,k,2]*d3U[i,j,k,1] - d2U[i,j,k,1]*d3U[i,j,k,2])*U[i,j,k,3]) + (d1U[i,j,k,2]*(d2U[i,j,k,3]*d3U[i,j,k,1] - d2U[i,j,k,1]*d3U[i,j,k,3]) + d1U[i,j,k,1]*(-(d2U[i,j,k,3]*d3U[i,j,k,2]) + d2U[i,j,k,2]*d3U[i,j,k,3]))*U[i,j,k,4] + d1U[i,j,k,3]*(d3U[i,j,k,4]*(d2U[i,j,k,2]*U[i,j,k,1] - d2U[i,j,k,1]*U[i,j,k,2]) + d2U[i,j,k,4]*(-(d3U[i,j,k,2]*U[i,j,k,1]) + d3U[i,j,k,1]*U[i,j,k,2]) + (-(d2U[i,j,k,2]*d3U[i,j,k,1]) + d2U[i,j,k,1]*d3U[i,j,k,2])*U[i,j,k,4]))
        end
    end

    return dens

end

function b_num(dens::Array{Float64},dy1::Array{Float64},dy2::Array{Float64},dy3::Array{Float64})::Float64
    
    l1 = length(dy1); l2 = length(dy2); l3 = length(dy3) 

    #----- computations

	B = 0

	for k in 1:l3 
		for j in 1:l2
			for i in 1:l1
				
				B = B + (-1/(24*pi^2))*dens[i,j,k]*dy1[i]*dy2[j]*dy3[k]
			
			end
		end
	end

    #----- output

    return B

end



