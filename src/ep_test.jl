using Pkg
Pkg.activate("/home/velni/phd/w/code/jl/NNP.jl/NNP")

using NNP

using Interpolations
using NPZ

#-----

hD = 0.2
factor = 2.2

#-----

old = npzread("/home/velni/Escritorio/D4U_r=61_Q=32.npy")
l1,l2,l3,l4 = size(old)

l1e = round(Int64, factor*l1); l2e = round(Int64, factor*l2); l3e = round(Int64, factor*l3)
new = zeros(eltype(old), l1e,l2e,l3e,4)

Y1 = hD .* ((0:(l1-1)) .- (l1-1)/2); Y2 = hD .* ((0:(l2-1)) .- (l2-1)/2); Y3 = hD .* ((0:(l3-1)) .- (l3-1)/2)
Y1e = hD .* ((0:(l1e-1)) .- (l1e-1)/2); Y2e = hD .* ((0:(l2e-1)) .- (l2e-1)/2); Y3e = hD .* ((0:(l3e-1)) .- (l3e-1)/2)

# copy original into expanded

z1 = floor(Int64,(l1e-l1)/2) + 1; z2 = floor(Int64,(l2e-l2)/2) + 1; z3 = floor(Int64,(l3e-l3)/2) + 1

new[z3:z3+l3-1,z2:z2+l2-1,z1:z1+l1-1,:] = old

#----- add 0s

Y1s = cat([-hD*l1e/2],collect(Y1),[hD*l1e/2]; dims=1)
Y2s = cat([-hD*l2e/2],collect(Y2),[hD*l2e/2]; dims=1)
Y3s = cat([-hD*l3e/2],collect(Y3),[hD*l3e/2]; dims=1)

olds = zeros(eltype(old),l1+2,l2+2,l3+2,4)
olds[2:l1+1,2:l2+1,2:l3+1,:] .= old

#----- create interpolators

for c in 1:4
	itp = interpolate((Y1s,Y2s,Y3s),olds[:,:,:,c], Gridded(Linear()))

    for i3 in 2:(l3e-1), i2 in 2:(l2e-1), i1 in 2:(l1e-1)
        if i3 >= z3 && i3 < z3+l3 && i2 >= z2 && i2 < z2+l2 && i1 >= z1 && i1 < z1+l1
            continue
        end
        new[i1,i2,i3,c] = itp(Y1e[i1],Y2e[i2],Y3e[i3])
    end
end

#-----

npzwrite("/home/velni/Escritorio/ep2_D4.npy", new)
