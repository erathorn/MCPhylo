mutable struct BrownianDistr <: DiscreteMatrixDistribution
    μ_Arr::Array{Float64,2}
    Σ_L_Arr::Array{Float64,3}
    latent::Array{Float64,2}
    dim::Tuple{Int64,Int64}

    # function BrownianDistr(μ_Arr::Array{Float64,2}, Σ_L_Arr::Array{Float64,3}, latent::Array{Float64,2})
    #    new(μ_Arr, Σ_L_Arr, latent, (size(Σ_L_Arr,1), size(Σ_L_Arr,3)), ones(3,3,3))
    # end

    function BrownianDistr(μ::Array{Float64,2}, σ::Array{Float64,1}, Σ::Array{Float64}, latent::Array{Float64,2})
        n_concs = size(μ, 2)
        n_leaves = size(Σ, 1)

        Σ_L_Arr = Array{Float64,3}(undef, n_leaves, n_leaves, n_concs)

        Threads.@threads for i in 1:n_concs
            @inbounds Σ_L_Arr[:, :, i] .= cholesky(σ[i] .* Σ).L
        end
        new(μ, Σ_L_Arr, latent, (size(Σ, 1), n_concs))
    end
end

function BrownianDistr(μ::ArrayVariate, σ::ArrayVariate, Σ::ArrayVariate, latent::ArrayVariate)
    BrownianDistr(μ.value, σ.value, Σ.value, latent.value)
end

function BrownianDistr(μ_Arr::ArrayVariate, Σ_Arr::ArrayVariate, latent::ArrayVariate)
    BrownianDistr(μ_Arr.value, Σ_Arr.value, latent.value)
end

minimum(d::BrownianDistr) = -Inf
maximum(d::BrownianDistr) = Inf

Base.size(d::BrownianDistr) = d.dim

# sampler(d::BrownianDistr) = Sampleable{MatrixVariate,Discrete}

function logpdf(d::BrownianDistr, x::Array{Float64,2})::Float64
    langs, concs = size(x)
    res = Threads.Atomic{Float64}(0.0)
    Base.Threads.@threads for i in 1:concs
        # res +=sum(logpdf.(Bernoulli.(invlogit.(d.μ_Arr[:,i]+d.Σ_L_Arr[:, :, i]*d.latent[:,i])), x[:,i]))
        Threads.atomic_add!(res, @inbounds sum(logpdf.(Bernoulli.(invlogit.(d.μ_Arr[:,i] + d.Σ_L_Arr[:, :, i] * d.latent[:,i])), x[:,i])))
    end
    return res[]
end



mutable struct BrownianCatDistr <: DiscreteMatrixDistribution
    μ_Arr::Array{Float64,2}
    Σ_L_Arr::Array{Float64,2}
    sigmaarr::Array{Float64,1}
    latent::Array{Float64,1}
    nclasses_vec::Vector{Int}
    dim::Tuple{Int64,Int64}

    # function BrownianDistr(μ_Arr::Array{Float64,2}, Σ_L_Arr::Array{Float64,3}, latent::Array{Float64,2})
    #    new(μ_Arr, Σ_L_Arr, latent, (size(Σ_L_Arr,1), size(Σ_L_Arr,3)), ones(3,3,3))
    # end

    function BrownianCatDistr(μ::Array{Float64,2}, σ::Array{Float64,1}, Σ::Array{Float64,2}, latent::Array{Float64,1},
                                nclasses_vec::Vector{Int})
        n_concs = size(μ, 2)
        n_leaves = size(Σ, 1)

        # Σ_L_Arr = Array{Float64,3}(undef, n_leaves, n_leaves, n_concs)
        Σ_L_Arr = Dict{Int,Array{Float64,2}}()
        # Threads.@threads
        # for i in 1:n_concs
            # @inbounds
            # Σ_L_Arr[:, :, i] .= cholesky(σ[i] .* Σ).L
        #    Σ_L_Arr[i] = cholesky(σ[i] .* kron(Diagonal(ones(nclasses_vec[i])),Σ)).L
        # end
        new(μ, Σ, σ, latent, nclasses_vec, (size(Σ, 1), n_concs))
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



minimum(d::BrownianCatDistr) = -Inf
maximum(d::BrownianCatDistr) = Inf

Base.size(d::BrownianCatDistr) = d.dim

function logpdf(d::BrownianCatDistr, x::Array{Float64,2})::Float64
    langs, concs = size(x)
    res = Threads.Atomic{Float64}(0.0)

    runningind = 0
    runningend = 0
    running_start = Vector{Int}(undef, concs)
    running_end = Vector{Int}(undef, concs)
    @inbounds for i in 1:concs
        runningend = runningind + langs * d.nclasses_vec[i]
        running_start[i] = runningind + 1
        running_end[i] = runningend
        runningind = runningend
    end


    Base.Threads.@threads for i in 1:concs
        s::Int = running_start[i]
        e::Int = running_end[i]
        if d.nclasses_vec[i] != 0
            @inbounds @views R = cholesky(d.sigmaarr[i] .* kron(Diagonal(ones(nclasses_vec[i])), d.Σ_L_Arr)).L * d.latent[s:e]
            @inbounds vals = reshape(R, (langs, d.nclasses_vec[i]))
            for (ind, entry) in enumerate(x[:,i])
                Threads.atomic_add!(res, @inbounds entry != -1 ? log(NNlib.softmax(vals[ind,:])[Int(entry)]) : 0.0)
            end
        end


    end
    return res[]
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
