function term_v1(i::Int64,j::Int64,k::Int64,u::Array{Float64},d1::Array{Float64},d2::Array{Float64},d3::Array{Float64})
	return d1[i,j,k,1]^2 + d1[i,j,k,2]^2 + d1[i,j,k,3]^2 + d1[i,j,k,4]^2 + d2[i,j,k,1]^2 + d2[i,j,k,2]^2 + d2[i,j,k,3]^2 + d2[i,j,k,4]^2 + d3[i,j,k,1]^2 + d3[i,j,k,2]^2 + d3[i,j,k,3]^2 + d3[i,j,k,4]^2
end
