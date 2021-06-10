"""
    prune_tree(root::T, node_names::Vector{String})::T where T<:GeneralNode

This function removes specific nodes, including their descendants, from a tree.

* `root` : root Node of tree to prune.

* `node_names` : vector of strings, used to specify nodes to remove.
"""
function prune_tree(root::T, node_names::Vector{String})::T where T<:GeneralNode
    # copy the tree and call the inplace version of the function on the copy
    if root.name in node_names
        throw(ArgumentError("trying to prune root, please set root to nothing instead"))
    end
    copyroot = deepcopy(root)
    prune_tree!(copyroot, node_names)
    return copyroot
end


"""
    prune_tree!(root::T, node_names::Vector{String})::Nothing where T<:GeneralNode

In-place version of prune_tree.

* `root` : root Node of tree to prune.

* `node_names` : vector of strings, used to specify nodes to remove.
"""
function prune_tree!(root::T, node_names::Vector{String})::Nothing where T<:GeneralNode
    nodes = post_order(root)
    names = [node.name for node in nodes]
    nodes_to_prune = Vector{T}()
    for name in node_names
        indeces = findall(x->x == name, names)
        if length(indeces) > 1
            throw(ArgumentError("Multiple nodes are named \"$name\""))
        elseif length(indeces) == 0
            print("Warning: No node named \"$name\" in tree")
        else
            push!(nodes_to_prune, nodes[indeces[1]])
        end # if/else
    end # for
    if length(nodes_to_prune) == 0
        throw(ArgumentError("None of the node names correspond to a node in the tree"))
    end # if
    prune_tree!(root, nodes_to_prune)
end


"""
    prune_tree!(root::T, node_names::Vector{T})::Nothing where T<:GeneralNode

In-place version of prune_tree.

* `root` : root node of tree to prune.

* `node_names`: vector of Node objects to be removed from tree.
"""
function prune_tree!(root::T, nodes::Vector{T})::Nothing where T<:GeneralNode
    if root in nodes
        throw(ArgumentError("trying to prune root, please set root to nothing instead"))
    else
        for node in nodes
            remove_child!(node.mother, node)
        end
    end
end
