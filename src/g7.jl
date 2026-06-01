function term_g7(i::Int64,j::Int64,k::Int64,DA::Array{Float64},DB::Array{Float64},d1::Array{Float64},d2::Array{Float64},d3::Array{Float64})
	return ((d1[i,j,k,1]^2 + d1[i,j,k,2]^2 + d1[i,j,k,3]^2 + d1[i,j,k,4]^2 + d2[i,j,k,1]^2 + d2[i,j,k,2]^2 + d2[i,j,k,3]^2 + d2[i,j,k,4]^2 + d3[i,j,k,1]^2 + d3[i,j,k,2]^2 + d3[i,j,k,3]^2 + d3[i,j,k,4]^2)^2)*(DA[i,j,k,1]*DB[i,j,k,1] + DA[i,j,k,2]*DB[i,j,k,2] + DA[i,j,k,3]*DB[i,j,k,3] + DA[i,j,k,4]*DB[i,j,k,4])
end
