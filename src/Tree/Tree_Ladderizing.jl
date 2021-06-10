"""
    ladderize_tree!(root::T, ascending::Bool=true)
        ::Nothing where T<:GeneralNode

This function ladderizes a tree inplace, i.e. sorts the nodes on all levels by the count
of their descendants.

* `root` : root Node of tree.

* `ascending` : Boolean, determines whether to sort in ascending (true) or
                descending (false) order.
"""
function ladderize_tree!(root::T, ascending::Bool=true) where T<:GeneralNode
    root.nchild == 0 && return nothing
    ndescendants = Array{Float64,1}(undef, length(root.children))
    for (index, child) in enumerate(root.children)
        ndescendants[index] = size(child)[1]
    end
    perm = sortperm(ndescendants, rev=!ascending)
    root.children = root.children[perm]

    for child in root.children
        ladderize_tree!(child, ascending)
    end
end


"""
    ladderize_tree(root::T, ascending::Bool=true)::T where T<:GeneralNode

This function returns a ladderized copy of a tree, i.e. a copy with all the
nodes on all levels sorted by the count of their descendants.

* `root` : root Node of tree.

* `ascending` : Boolean, determines whether to sort in ascending (true) or 
                descending (false) order.
"""
function ladderize_tree(root::T, ascending::Bool=true)::T where T<:GeneralNode
    copyroot::T = deepcopy(root)
    ladderize_tree!(copyroot, ascending)
    return copyroot
end
