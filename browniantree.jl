mutable struct BrownianTreeDistr <: DiscreteMatrixDistribution
    μ_Arr::A where A <: DenseArray{<:Real,2}
    tree::T where T <: GeneralNode
    σ::A where A <: DenseArray{<:Real,1}
    latent::A where A <: DenseArray{<:Real,2}
    dim::Tuple{Int64,Int64}

    #function BrownianDistr(μ_Arr::Array{Float64,2}, Σ_L_Arr::Array{Float64,3}, latent::Array{Float64,2})
    #    new(μ_Arr, Σ_L_Arr, latent, (size(Σ_L_Arr,1), size(Σ_L_Arr,3)), ones(3,3,3))
    #end

    function BrownianTreeDistr(μ::DenseArray{Float64,2}, tree::T, σ::DenseArray{Float64,1}, latent::DenseArray{Float64,2}) where T<:GeneralNode
        n_concs, n_leaves = size(μ)
        new(μ, tree, σ, latent, (n_concs, n_leaves))
    end
end

function BrownianTreeDistr(μ::AbstractVariate, σ::AbstractVariate, Σ::TreeVariate, latent::AbstractVariate)
    BrownianTreeDistr(μ.value, Σ.value, σ.value, latent.value)
end

#function BrownianTreeDistr(μ::DenseArray, σ::DenseArray, Σ::GeneralNode, latent::DenseArray)
#    BrownianTreeDistr(Array(μ), Σ, Array(σ), Array(latent))
#end


Base.minimum(d::BrownianTreeDistr) = -Inf
Base.maximum(d::BrownianTreeDistr) = Inf

Base.size(d::BrownianTreeDistr) = d.dim

#sampler(d::BrownianDistr) = Sampleable{MatrixVariate,Discrete}

function logpdf(d::BrownianTreeDistr, x::A)::Float64 where A<:(AbstractMatrix{T} where T<:Real)
    blv = CuArray(get_branchlength_vector(d.tree))
    om = CuArray(other_matrix(d.tree, length(blv)))
    
    #res = Base.Threads.Atomic{Float64}(0.0)
        
    r = lpdfcalculator_i(blv, om, d.μ_Arr, d.latent, d.σ, x)
    #r2 = lpdfcalculator_i(collect(blv), collect(om), collect(d.μ_Arr), collect(d.latent), collect(d.σ), collect(x))
    
    
    r
end

function gradlogpdf(d::BrownianTreeDistr, x::AbstractArray)
    blv = CuArray(get_branchlength_vector(d.tree))
    
    om = CuArray(other_matrix(d.tree, length(blv)))
    #f(y) = lpdfcalculator(y, om, d.μ_Arr, d.latent, d.σ, x)
    #r = Zygote.pullback(f, blv)
    #res = r[1],r[2](1.0)[1]

    f1(y) = lpdfcalculator_i(y, om, d.μ_Arr, d.latent, d.σ, x)
    #grd = zeros(size(blv))
    #mr = Base.Threads.Atomic{Float64}(0.0)
    ##grd = Array{Base.Threads.Atomic{Float64},1}(0, length(blv))
    #grd = [Base.Threads.Atomic{Float64}(0.0) for i in 1:length(blv)]
    
    #Base.Threads.atomic_xchg!.(grd, 0.0)
    r = Zygote.pullback(f1, blv)
    # @show r
    gr = r[2](1.0)[1]
    
    #@show isapprox(mr, res[1])
    #@show isapprox.(grd, res[2])

    return r[1], gr
end


function lpdfcalculator_i(blv::A, om::B, μ_Arr::B, latent::B, σ::A, x::B)::F where {A<:AbstractArray{F, 1}, B<:AbstractArray{F,2}} where  F<:Real
    cl = om * Diagonal(sqrt.(blv))
    bernp = μ_Arr .+  cl * (transpose(sqrt.(σ)) .* latent)
    res = sum(mbern.(logistic.(bernp), x))
    res
end




# function lpdfcalculator_i(blv::Array{F,1}, om::Array{F,2}, μ_Arr::Array{F,2}, latent::Array{F,2}, σ::Array{F,1}, x::Array{F,2})::F where F<:Real
#     cl = om * Diagonal(sqrt.(blv))
#     res = Atomic{Float64}(0.0)
#     @threads for i in 1:size(x,2)
#         @inbounds tmp = innerll(μ_Arr[:,i], sqrt(σ[i]), cl, latent[:,i], x[:,i])
#         atomic_add!(res, tmp)
#     end
#     res[]
# end

# function innerll(μ_Arr::Array{F,1}, s_σ::F, cl::Array{F,2}, latent::Array{F,1}, x::Array{F,1})::F where F <: Real
#     bern_p::Array{F,1} = μ_Arr + BLAS.gemv('N', s_σ, cl, latent)
#     @inbounds mapreduce(j -> mbern(logistic(bern_p[j]), x[j]), +, 1:size(bern_p,1), init=0.0)
# end


const global l_fmin_F64 = log(floatmin(Float64))
const global z_F64 = zero(Float64)

"""
    let bernoulli handle missing data
"""
@inline function mbern(p::R, x::R)::R where R<:Real
    x == -1 && return z_F64
    x == 0 ? succ_prob(p) : fail_prob(p)
end

function succ_prob(x::R)::R where R <: Real
    x == 0 ? l_fmin_F64 : log(x)
end

function fail_prob(x::R)::R where R <: Real
    x == 1 ? l_fmin_F64 : log(1-x)
end
