
@inline function mygemmturbo!(C::R, A::S, B::T)::Nothing where {T,R,S}
    @tturbo for m ∈ axes(A, 1), n ∈ axes(B, 2)
        Cmn = zero(eltype(C))
        for k ∈ axes(A, 2)
            Cmn += A[m, k] * B[k, n]
        end
        C[m, n] = Cmn
    end
    nothing
end

function my_repeat(
    data::Array{F,N},
    nrates::Int,
    linds::Vector{T},
)::Array{F,4} where {F,N,T}
    x, y, z = size(data)
    out = Array{F,4}(undef, x, y, nrates, z)

    @turbo check_empty = false for l in axes(out, 4),
        rind in axes(out, 3),
        xs in axes(out, 1),
        ys in axes(out, 2)

        out[xs, ys, rind, l] = ifelse(l in linds, data[xs, ys, l], 1.0)
    end
    out
end

function by_max!(data::Array{F,4}, ll::F, nodenum::Int)::F where {F<:Real}
    maxi = fill(-Inf, size(data, 2), size(data, 3))
    @inbounds for k in axes(data, 3), j in axes(data, 2), i in axes(data, 1)
        maxi[j, k] =
            ifelse(maxi[j, k] < data[i, j, k, nodenum], data[i, j, k, nodenum], maxi[j, k])
    end
    @tturbo check_empty = false for i in eachindex(maxi)
        ll += log(maxi[i])
        maxi[i] = 1 / maxi[i]
    end

    @tturbo check_empty = false for i in axes(data, 1),
        k in eachindex(CartesianIndices(maxi))

        data[i, k, nodenum] *= maxi[k]
    end
    ll
end


function bymax!(pre_order_partial::A, nodenum::Int)::Nothing where {A}
    maxi = fill(-Inf, size(pre_order_partial, 2), size(pre_order_partial, 3))
    @inbounds for r in axes(pre_order_partial, 3),
        i in axes(pre_order_partial, 2),
        j in axes(pre_order_partial, 1)

        maxi[i, r] = ifelse(
            maxi[i, r] < pre_order_partial[j, i, r, nodenum],
            pre_order_partial[j, i, r, nodenum],
            maxi[i, r],
        )
    end

    @tturbo check_empty = false for j in axes(pre_order_partial, 2),
        r in axes(pre_order_partial, 3)

        tmp = 1 / maxi[j, r]
        for i in axes(pre_order_partial, 1)
            pre_order_partial[i, j, r, nodenum] *= tmp
        end
    end
    nothing
end


@inline function root_sum(data::Array{F,4}, pi_::S, lind::Int, ll::F)::F where {F,S}

    @tturbo check_empty = false for k in CartesianIndices((axes(data, 2), axes(data, 3)))
        tmp = zero(F)
        for i in axes(data, 1)
            tmp += data[i, k, lind] * pi_[i]
        end
        ll += log(tmp)
    end

    ll
end

function R_gemmturbo_large_ptg!(
    C::T,
    A::S,
    mu::F,
    blv::V,
    D::V,
    rates::V,
)::Nothing where {T,S,F,V}
    @tturbo check_empty = false for m ∈ axes(A, 1),
        i in axes(C, 2),
        l in axes(C, 3),
        n in axes(C, 4)

        Cmn = zero(eltype(C))
        for k ∈ axes(A, 2)
            Cmn += ifelse(
                i == k,
                A[m, k] * D[i] * rates[l] * mu * exp(mu * blv[n] * D[i] * rates[l]),
                0.0,
            )
        end
        C[m, i, l, n] = Cmn
    end
    nothing
end


function R_gemmturbo_large!(
    C::T,
    A::S,
    mu::F,
    blv::V,
    D::V,
    rates::V,
)::Nothing where {T,S,V,F}
    @tturbo check_empty = false for m ∈ axes(A, 1),
        i in axes(C, 2),
        l in axes(C, 3),
        n in axes(C, 4)

        Cmn = zero(eltype(C))
        for k ∈ axes(A, 2)
            Cmn += ifelse(i == k, A[m, k] * exp(mu * blv[n] * D[i] * rates[l]), 0.0)
        end
        C[m, i, l, n] = Cmn
    end
    nothing
end


function L_gemmturbo_large!(C::T, A::T, B::S)::Nothing where {T,S}
    @tturbo check_empty = false for r ∈ axes(A, 3),
        l ∈ axes(A, 4),
        m ∈ axes(A, 1),
        n ∈ axes(B, 2)

        Cmn = zero(eltype(C))
        for k ∈ axes(A, 2)
            Cmn += A[m, k, r, l] * B[k, n]
        end
        C[m, n, r, l] = Cmn
    end
    nothing
end


function parallel_transition_prob(
    U::Matrix{R},
    D::Vector{R},
    Uinv::Matrix{R},
    rates::Vector{R},
    mu::Float64,
    blv::Vector{R},
)::Array{R,4} where {R<:Real}
    out = Array{R,4}(undef, (length(D), length(D), length(rates), length(blv)))
    out2 = Array{R,4}(undef, (length(D), length(D), length(rates), length(blv)))
    R_gemmturbo_large!(out2, U, mu, blv, D, rates)
    L_gemmturbo_large!(out, out2, Uinv)
    out
end


function parallel_transition_prob(
    U::Matrix,
    D::Vector,
    Uinv::Matrix,
    rates::Vector,
    mu::Float64,
    blv::Vector{R},
)::Array{R,4} where {R<:Real}
    
    res = broadcast(x -> diagm(exp.(D .* mu .* x)), rates * blv')
    res2 = broadcast( x-> U * x * Uinv, res)
    return reshape(hcat(res2...), (length(D), length(D), length(rates), length(blv)))
end


function turbo_mul!(data::A, tmp_data::A, pnum::D, nums::Vector{D})::Nothing where {A,D}
    @tturbo check_empty = false inline = true for num_ind in eachindex(nums),
        r in CartesianIndices((axes(tmp_data, 1), axes(tmp_data, 2), axes(tmp_data, 3)))

        data[r, pnum] *= tmp_data[r, nums[num_ind]]
    end
    nothing
end

function turbo_dot(x::Vector{T}, y::Vector{T})::T where {T}
    res = zero(T)
    @turbo check_empty = false for i in eachindex(x, y)
        res += x[i] * y[i]
    end
    res
end

function comb_sum_product_loop!(
    Down::A,
    data::A,
    transprobs::A,
    nums::Vector{N},
    pnum::N,
)::Nothing where {A,N}

    @tturbo check_empty = false for s in axes(data, 1),
        c in axes(data, 2),
        r in axes(data, 3)

        mulcomb = one(eltype(data))
        for num_ind in eachindex(nums)
            tmp = zero(eltype(data))
            num = nums[num_ind]
            for s1 in axes(data, 1)
                tmp += data[s1, c, r, num] * transprobs[s, s1, r, num]
            end
            Down[s, c, r, num] = tmp
            mulcomb *= tmp
        end
        data[s, c, r, pnum] = mulcomb
    end
    nothing

end
