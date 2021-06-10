
#################### Safe searches ####################

#"""
# proper Markdown comments are not possible
#
#This part creates functions which enable the search for different nodes in the
#tree. It is possible to look for a node via its name, its binary representation
#or to find the root.
#This functionality can be extended by adding more fields to the nodes and the
#meta programmming part here.

# These functions are safe, in the sense that they properly exit if the node is
# not found.
#"""
for (sym, my_type) in [(:binary, :String), (:name, :AbstractString), (:root ,:Bool), (:num, :Int64)]
    # extend the list to look for more fields in the node
    @eval function $(Symbol(string("find_by_$sym")))(tree::T, identifier::$my_type)::T  where T<:GeneralNode
        # create each function and make it so it only accepts the correct type
        local all_nodes = post_order(tree) # make sure all_nodes only belongs to this function
        for node in all_nodes
            if node.$sym == identifier
                # return the node if it is found
                return node
            end # if
        end # for
        # the node is not found. Therefore throw an error!
        throw("The node identified by $identifier is not in the tree.")
    end # function
end


#################### Unsafe searches ####################

"""
    find_num(root::T, num::Int64)  where T<:GeneralNode

Find a node by its number. The function assumes that the node is present in the
tree.

Do not use this function if you are unsure whether the node is in the tree at all.

Returns reference to Node.

* `root` : root Node of tree to be searched.

* `num` : number of desired Node.
"""
function find_num(root::T, num::I)::T  where {T<:GeneralNode, I<:Integer}
    po = post_order(root)
    store = T[]
    found = find_num(root, num, store)
    if length(store) == 0
        throw(ArgumentError("Node not found"))
    else
        return store[1]
    end
end


"""
    find_num(root::T, num::Int64, rn::Vector{T})::Bool  where T<:GeneralNode

Do a post order traversal to find the node corresponding to the `num`.

Returns true if node is found, false otherwise. Desired Node is pushed to rn.

* `root` : root Node of tree to be searched.

* `num` : number of desired Node.

* `rn` : Vector of Nodes; desired Node is pushed to this vector when found.
"""
function find_num(root::T, num::I, rn::Vector{T})::Bool  where {T<:GeneralNode, I<:Integer}
    # if the current node is the correct one store it in rn
    if root.num === num
        push!(rn, root)
        found = true
    else
        found = false
    end

    if !found
        # if the node is not yet found continue
        for child in root.children
            found = find_num(child, num,  rn)
        end
    end # if
    return found
end

"""
    find_binary(root::T, bin::String)::T where T<:GeneralNode

Find a node by its binary representation. The function assumes that the node is
present in the tree.

Do not use this function if you are unsure whether the node is in the tree at all.

Returns a reference to the desired Node.

* `root` : root Node of tree to search.

* `bin` : binary representation of desired Node as a String.
"""
function find_binary(root::T, bin::String)::T where T<:GeneralNode
    rv = root
    for i in split(bin[1:end-1],",")[2:end]
        rv = rv.children[parse(Int64, i)+1]
    end
    rv
end

"""
    find_root(node::T)::T where T <: GeneralNode

Finds the root of tree indicated by Node.

Returns reference to root Node of the tree.

* `node` : Node in Tree of interest.
"""
function find_root(node::T)::T where T <: GeneralNode
    while node.root == false
        node = node.mother
    end # while
    return node
end
