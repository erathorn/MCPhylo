


"""
    This structure implements a Distribution whos likelihood is calculated
    according to Felsensteins algorithm.
"""
struct PhyloDist{T<:GeneralNode,F<:Function} <: DiscreteMatrixDistribution
    tree::T
    base_freq::Vector{Float64}
    substitution_rates::Vector{Float64}
    rates::Vector{Float64}
    substitution_model::F
    nbase::Int64
    nnodes::Int64

    function PhyloDist(
        tree::T,
        base_freq::A,
        substitution_rates::A,
        rates::A,
        substitution_model::F,
    ) where {T<:GeneralNode,A<:AbstractArray{<:Real},F<:Function}
        new{T,F}(
            tree,
            base_freq,
            substitution_rates,
            rates,
            substitution_model,
            length(base_freq),
            length(post_order(tree)),
        )
    end

end

function PhyloDist(
    tree::T,
    base_freq::S,
    substitution_rates::A,
    rates::B,
    substitution_model::Function,
) where {
    T<:TreeVariate,
    S<:DenseArray{Float64},
    A<:DenseArray{Float64},
    B<:DenseArray{Float64},
}
    PhyloDist(
        tree.value,
        Array(base_freq),
        Array(substitution_rates),
        Array(rates),
        substitution_model,
    )
end

function PhyloDist(
    my_tree::T,
    base_freq::S,
    substitution_rates::R,
    rates::R,
    substitution_model::Function,
) where {T<:TreeVariate,S<:DenseArray{Float64},R<:Real}
    PhyloDist(
        my_tree.value,
        Array(base_freq),
        [substitution_rates],
        [rates],
        substitution_model,
    )
end

"""
    function PhyloDist(my_tree::T, base_freq::S, substitution_rates::R, rates::R, substitution_model::Function) where {T<:GeneralNode, S<:DenseArray{Float64}, R<:Real}

Convenience function which can work with MCPhylo types.
"""
function PhyloDist(
    my_tree::T,
    base_freq::S,
    substitution_rates::R,
    rates::R,
    substitution_model::Function,
) where {T<:GeneralNode,S<:DenseArray{Float64},R<:Real}
    PhyloDist(my_tree, Array(base_freq), [substitution_rates], [rates], substitution_model)
end

function PhyloDist(
    my_tree::A,
    substitution_rates::B,
    rates::R,
    substitution_model::K,
) where {A<:AbstractNode,B<:DenseArray{Float64},R<:DenseArray{Float64},K<:typeof(freeK)}
    U, D, Uinv, _ = freeK(Float64[], Array(substitution_rates))
    D[:] .= 0
    D[end] = 1
    eq = real.((U*diagm(D)*Uinv)[1, :])
    PhyloDist(my_tree, eq, substitution_rates, rates, substitution_model)
end

minimum(d::PhyloDist) = -Inf
maximum(d::PhyloDist) = Inf

Base.size(d::PhyloDist) = (d.nbase, 1, d.nnodes)

function logpdf(d::PhyloDist{T,F}, x::AbstractArray{<:Real,3})::Float64 where {T,F}
    mt = post_order(d.tree)

    U, D, Uinv, mu = d.substitution_model(
        d.base_freq,
        d.substitution_rates,
    )::Tuple{Matrix,Vector,Matrix,Float64}

    blv = get_branchlength_vector(last(mt))
    leaveinds = [l.num for l in get_leaves(last(mt))]

    data_ext = my_repeat(x, length(d.rates), leaveinds)
    Down = deepcopy(data_ext)
    trans_probs = parallel_transition_prob(U, D, Uinv, d.rates, mu, blv)
    FelsensteinFunction(mt, data_ext, Down, d.base_freq, trans_probs)
end

function gradlogpdf(d::PhyloDist{T,F}, x::AbstractArray) where {T,F}
    mt = post_order(d.tree)
    U, D, Uinv, mu = d.substitution_model(d.base_freq, d.substitution_rates)#::Tuple{Matrix{Float64},Vector{Float64},Matrix{Float64},Float64}
    blv = get_branchlength_vector(d.tree)

    data_ext = my_repeat(x, length(d.rates), [l.num for l in get_leaves(d.tree)])
    Down = deepcopy(data_ext)
    trans_probs = parallel_transition_prob(U, D, Uinv, d.rates, mu, blv)
    ptg = zeros(size(trans_probs))
    out2 = similar(ptg)
    R_gemmturbo_large_ptg!(out2, U, mu, blv, D, d.rates)
    L_gemmturbo_large!(ptg, out2, Uinv)
    ll, gr = FelsensteinFunction(mt, data_ext, Down, d.base_freq, trans_probs, ptg)
    ll, gr
end




mutable struct MultiplePhyloDist <: DiscreteMatrixDistribution
    DistCollector::Array{PhyloDist}
    size_array::Array{Int64}

    function MultiplePhyloDist(
        tree_array::Array{T},
        base_freq::Array{Float64,2},
        substitution_rates::Array{Float64,2},
        rates::Array{Float64,2},
        substitution_model::Function,
    ) where {T<:GeneralNode}
        size_array::Array{Int64,1} = Array{Int64,1}(undef, length(tree_array))
        pd_array = PhyloDist[]
        for (ind, tree) in enumerate(tree_array)
            pd = PhyloDist(
                tree,
                base_freq[:, ind],
                substitution_rates[:, ind],
                rates[:, ind],
                substitution_model,
            )
            push!(pd_array, pd)
            size_array[ind] = length(post_order(tree))
        end
        new(pd_array, size_array)
    end
end


function MultiplePhyloDist(
    tree_array::Array{T},
    base_freq::S,
    substitution_rates::R,
    rates::U,
    substitution_model::Function,
) where {
    T<:GeneralNode,
    S<:DenseArray{Float64},
    R<:DenseArray{Float64},
    U<:DenseArray{Float64},
}
    n_t = length(tree_array)
    base_freq_l = Array{Float64,2}(undef, size(base_freq, 1), n_t)
    substitution_rates_l = Array{Float64,2}(undef, size(substitution_rates, 1), n_t)
    rates_l = Array{Float64,2}(undef, size(rates, 1), n_t)
    if size(base_freq, 2) == n_t
        base_freq_l .= base_freq
    elseif size(base_freq, 2) == 1
        for i = 1:n_t
            base_freq_l[:, i] .= base_freq
        end
    else
        throw(DimensionMismatch("Size of base_freq and tree_array are incompatible"))
    end

    if size(substitution_rates, 2) == n_t
        substitution_rates_l .= substitution_rates
    elseif size(substitution_rates, 2) == 1
        for i = 1:n_t
            substitution_rates_l[:, i] .= substitution_rates
        end
    else
        throw(
            DimensionMismatch("Size of substitution_rates and tree_array are incompatible"),
        )
    end

    if size(rates, 2) == n_t
        rates_l .= rates
    elseif size(rates, 2) == 1
        for i = 1:n_t
            rates_l[:, i] .= rates
        end
    else
        throw(DimensionMismatch("Size of rates and tree_array are incompatible"))
    end
    MultiplePhyloDist(
        tree_array,
        base_freq_l,
        substitution_rates_l,
        rates_l,
        substitution_model,
    )
end


function MultiplePhyloDist(
    tree_array::Array{T},
    substitution_rates::R,
    rates::S,
    substitution_model::K,
) where {T<:GeneralNode,S<:DenseArray{Float64},R<:DenseArray{Float64},K<:typeof(freeK)}
    n = size(substitution_rates, 1)
    s = Int(round(sqrt(n+1/4)+1/2))

    n_t = length(tree_array)
    base_freq_l = Array{Float64,2}(undef, s, n_t)
    substitution_rates_l = Array{Float64,2}(undef, size(substitution_rates, 1), n_t)
    if size(substitution_rates, 2) == n_t
        for i = 1:n_t
            U, D, Uinv, _ = freeK(Float64[], Array(substitution_rates[:, i]))
            D[:] .= 0
            D[end] = 1
            eq = real.((U*diagm(D)*Uinv)[1, :])
            substitution_rates_l[:, i] .= substitution_rates[:, i]
            base_freq_l[:, i] .= eq
        end
    elseif size(substitution_rates, 2) == 1
        U, D, Uinv, _ = freeK(Float64[], Array(substitution_rates))
        D[:] .= 0
        D[end] = 1
        eq = real.((U*diagm(D)*Uinv)[1, :])
        for i = 1:n_t
            substitution_rates_l[:, i] .= substitution_rates
            base_freq_l[:, i] .= eq
        end
    else
        throw(
            DimensionMismatch("Size of substitution_rates and tree_array are incompatible"),
        )
    end

    MultiplePhyloDist(
        tree_array,
        base_freq_l,
        substitution_rates_l,
        rates,
        substitution_model,
    )
end


minimum(d::MultiplePhyloDist) = -Inf
maximum(d::MultiplePhyloDist) = Inf

Base.size(d::MultiplePhyloDist) =
    (size(d.DistCollector[1].base_freq, 1), 1, maximum(d.size_array), length(d.size_array))

function logpdf(d::MultiplePhyloDist, x::AbstractArray{<:Real,4})::Float64
    res = zero(Float64)
    @inbounds for (ind, s) in enumerate(d.size_array)
        xt = x[:, :, 1:s, ind]
        res += logpdf(d.DistCollector[ind], xt)
    end
    res
end

function __logpdf(d::MultiplePhyloDist, x::AbstractArray)
    res = Tuple[]
    @inbounds for (ind, s) in enumerate(d.size_array)
        xt = x[:, :, 1:s, ind]
        push!(res, gradlogpdf(d.DistCollector[ind], xt))
    end
    res
end


const PhylogeneticDistribution = Union{PhyloDist,MultiplePhyloDist}
