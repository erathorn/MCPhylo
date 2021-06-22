mutable struct BrownianDistr <: DiscreteMatrixDistribution
    μ_Arr::Array{Float64,2}
    Σ_L_Arr::Array{Float64,3}
    latent::Array{Float64,2}
    dim::Tuple{Int64,Int64}

    #function BrownianDistr(μ_Arr::Array{Float64,2}, Σ_L_Arr::Array{Float64,3}, latent::Array{Float64,2})
    #    new(μ_Arr, Σ_L_Arr, latent, (size(Σ_L_Arr,1), size(Σ_L_Arr,3)), ones(3,3,3))
    #end

    function BrownianDistr(μ::Array{Float64,2}, σ::Array{Float64,1}, Σ::Array{Float64}, latent::Array{Float64,2})
        n_concs = size(μ,1)
        n_leaves = size(Σ,2)
        Σ_L_Arr = Array{Float64,3}(undef, n_concs, n_leaves, n_leaves)

        Base.Threads.@threads for i in 1:n_concs
            @inbounds Σ_L_Arr[i,:, :] .= cholesky(σ[i] * Σ).L
        end
        new(μ, Σ_L_Arr, latent, (n_concs, n_leaves))
    end
end

function BrownianDistr(μ::ArrayVariate, σ::ArrayVariate, Σ::ArrayVariate, latent::ArrayVariate)
    BrownianDistr(μ.value, σ.value, Σ.value, latent.value)
end

function BrownianDistr(μ_Arr::ArrayVariate, Σ_Arr::ArrayVariate, latent::ArrayVariate)
    BrownianDistr(μ_Arr.value, Σ_Arr.value, latent.value)
end

Base.minimum(d::BrownianDistr) = -Inf
Base.maximum(d::BrownianDistr) = Inf

Base.size(d::BrownianDistr) = d.dim

#sampler(d::BrownianDistr) = Sampleable{MatrixVariate,Discrete}

function logpdf(d::BrownianDistr, x::Array{Float64,2})::Float64
    _ , concs = size(x)
    res = Threads.Atomic{Float64}(0.0)
    Base.Threads.@threads for i in 1:concs
        Threads.atomic_add!(res, @inbounds sum(mbern.(invlogit.(d.μ_Arr[i,:]+d.Σ_L_Arr[i,:, :]*d.latent[i,:]), x[i,:])))
    end
    return res[]
end


function mbern(p::R, x::R)::R where R<:Real
    x == -1 ? 0 : log(x == 1 ? p : 1 - p)
end



mutable struct BrownianCatDistr <: DiscreteMatrixDistribution
    μ_Arr::Array{Float64,2}
    Σ_L_Arr::Array{Float64,2}
    sigmaarr::Array{Float64,1}
    latent::Array{Float64,1}
    nclasses_vec::Vector{Int}
    dim::Tuple{Int64,Int64}
    running_start::Vector{Int}
    running_end::Vector{Int}

    #function BrownianDistr(μ_Arr::Array{Float64,2}, Σ_L_Arr::Array{Float64,3}, latent::Array{Float64,2})
    #    new(μ_Arr, Σ_L_Arr, latent, (size(Σ_L_Arr,1), size(Σ_L_Arr,3)), ones(3,3,3))
    #end

    function BrownianCatDistr(μ::Array{Float64,2}, σ::Array{Float64,1}, Σ::Array{Float64,2}, latent::Array{Float64,1},
                                nclasses_vec::Vector{Int})
        n_concs = size(σ,1)
        n_leaves = size(Σ,1)
        runningind = 0
        runningend = 0
        running_start = Vector{Int}(undef, n_concs)
        running_end = Vector{Int}(undef, n_concs)
        
        @inbounds for i in 1:n_concs
            runningend = runningind + n_leaves*nclasses_vec[i]
            running_start[i] = runningind+1
            running_end[i] = runningend
            runningind = runningend
        end
        new(μ, Σ, σ, latent, nclasses_vec, (size(Σ,1), n_concs), running_start, running_end)
    end
end

function BrownianCatDistr(μ::ArrayVariate, σ::ArrayVariate, Σ::ArrayVariate, latent::ArrayVariate)
    BrownianCatDistr(μ.value, σ.value, Σ.value, latent.value)
end

function BrownianCatDistr(μ::ArrayVariate, σ::ArrayVariate, Σ::ArrayVariate, latent::ArrayVariate, nclasses::Vector)
    BrownianCatDistr(μ.value, σ.value, Σ.value, latent.value, nclasses)
end


function BrownianCatDistr(μ_Arr::ArrayVariate, Σ_Arr::ArrayVariate, latent::ArrayVariate)
    BrownianCatDistr(μ_Arr.value, Σ_Arr.value, latent.value)
end

function BrownianCatDistr(μ_Arr::ArrayVariate, Σ_Arr::ArrayVariate, latent::ArrayVariate, nclasses::Vector)
    BrownianCatDistr(μ_Arr.value, Σ_Arr.value, latent.value, nclasses)
end



Base.minimum(d::BrownianCatDistr) = -Inf
Base.maximum(d::BrownianCatDistr) = Inf

Base.size(d::BrownianCatDistr) = d.dim

#sampler(d::BrownianCatDistr) = Sampleable{MatrixVariate,Discrete}

function logpdf(d::BrownianCatDistr, x::Array{Float64,2})::Float64
    langs, concs = size(x)
    r1 = zero(Float64)    
    #Sigma = CuArray{Float32}(d.Σ_L_Arr)
    
    #x_c = CuArray{Int32}(Int.(x))
    x_c = Int.(x)
    #lat_c = CuArray{Float32}(d.latent)
    #A = CuArray{Float32}(Diagonal(ones(Base.maximum(d.nclasses_vec))))
    A = Diagonal(ones(Base.maximum(d.nclasses_vec)))
    for i = 1:concs
        s::Int = d.running_start[i]
        e::Int = d.running_end[i]
        if d.nclasses_vec[i] != 0
            S = d.sigmaarr[i] .* kron(A[1:d.nclasses_vec[i],1:d.nclasses_vec[i]],d.Σ_L_Arr)
            
            R1 = sparse(cholesky(S).L)#.L
            
            R = R1 * d.latent[s:e]
            #println(size(R))
            vals = reshape(R, (langs, d.nclasses_vec[i]))
            #println(size(vals))
            #println(d.nclasses_vec[i], x_c[:,i])
            lpraw = log.(exp.(vals) ./ sum(exp.(vals), dims=2))
            #r1 += sum(selector(x_c[:,i], lpraw))

            f = j -> x_c[j,i] != -1 ? lpraw[j,x_c[j,i]] : 0.0
            r1 += sum(map(f, 1:langs))
        end
    end
    return r1
end


function selector(x_c::CuArray{S}, lpraw::CuArray{T,2}) where {T,S}
    R = CuArray{T}(undef, size(x_c))
    CUDA.@sync @cuda threads = length(R) selector_inner(R, x_c, lpraw)
    return R
end


function selector_inner(R, x_c, lpraw)
    i = (blockIdx().x-1) * (blockDim().x) + threadIdx().x
    cdim = length(R)
    if i <= cdim
        R[i] = x_c[i] != -1 ? lpraw[i,x_c[i]] : 0.0
    end
    nothing
end


function Cukron(a::CuArray{T}, b::CuArray{T}) where {T}
    R = CuArray{T}(undef, size(a,1)*size(b,1), size(a,2)*size(b,2))
    
    nbl = size(R,1)#512
    nth = size(R,2)#1024
    CUDA.@sync @cuda blocks=nbl threads=nth kernel_kron(R, a,b)
    return R
end

function kernel_kron(out, A, B)

    i = (blockIdx().x-1) * (blockDim().x) + threadIdx().x
    Bvalsize = length(B)
    Avalsize = length(A)
    
    cdim = Bvalsize * Avalsize
    
    if i <= cdim
        bRowSize, bColSize = size(B)
        aRowSize, aColSize = size(A)
        width = size(A,1)*size(B,1)
        x = (i-1) % width
        y = div((i-1), width)
        
        xa = div(y, bColSize)
        ya = div(x , bRowSize)

        yb = x % bRowSize
        xb = y % bColSize
        
        aind = (ya+xa*aColSize)+1
        bind = (yb+xb*bColSize)+1
        out[i] = A[aind] * B[bind]
    end
    return
end


function mysoftmax(a::AbstractArray{<:Real})
    exp.(a) / sum(exp.(a))
end



# function _rand!(r::A, d::BrownianDistr, x::AbstractMatrix) where A <: AbstractRNG
#     n_leaves, n_concs = d.dim
#     @inbounds @simd for i in 1:n_concs
#         x[:,i] .= Int.(rand.(Bernoulli.(invlogit.(d.μ_Arr[i,:] + BLAS.gemv('N',1.0,d.Σ_L_Arr[:, :, i], d.latent[i, :])))))
#     end
#     return x
# end
#
# mutable struct BrownianDistrMulti <: DiscreteMatrixDistribution
#     μ::Array{Float64,1}
#     σ::Array{Float64,1}
#     Σ::Array{Float64,2}
#     dim::Tuple{Int64,Int64}
#
#     function BrownianDistrMulti(μ::Array{Float64,1}, σ::Array{Float64,1}, Σ::Array{Float64})
#         new(μ, σ, Σ, (size(Σ,1), size(μ,1)))
#     end
# end
#
# function BrownianDistrMulti(μ::ArrayVariate, σ::ArrayVariate, Σ::ArrayVariate)
#     BrownianDistrMulti(μ.value, σ.value, Σ.value)
# end
#
# minimum(d::BrownianDistrMulti) = -Inf
# maximum(d::BrownianDistrMulti) = Inf
#
# Base.size(d::BrownianDistrMulti) = d.dim
#
# sampler(d::BrownianDistrMulti) = Sampleable{MatrixVariate,Discrete}
#
# function logpdf(d::BrownianDistrMulti, x::Array{Float64,2})
#     return 0
# end
#
# function _rand!(r::A, d::BrownianDistrMulti, x::AbstractMatrix) where A <: AbstractRNG
#     n_leaves, n_concs = d.dim
#     μ_arr::Array{Float64,2} = reshape(repeat(d.μ, outer=n_leaves), n_concs, n_leaves)
#     @inbounds for i in 1:n_concs
#         x[:,i] .= Int.(invlogit.(rand(MvNormal(μ_arr[i,:], d.σ[i].*d.Σ))) .> 0.5)
#     end
#     return x
# end
