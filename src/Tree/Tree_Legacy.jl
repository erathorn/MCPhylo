

"""
    get_branchlength_vector(post_order::Vector{T})::Vector{Float64}  where T<:GeneralNode

Return a vector of branch lengths.

* `post_order` : Vector of Nodes of a tree.
"""
function get_branchlength_vector(post_order::Vector{T})::Vector{Float64}  where T<:GeneralNode
    #println("You called a legacy function.")
    @warn "You called a legacy function"
    out = zeros(length(post_order)-1)
    @views @simd for i in eachindex(post_order)
        if !post_order[i].root
            out[post_order[i].num]= post_order[i].inc_length
        end
    end
    return out
end # function get_branchlength_vector
