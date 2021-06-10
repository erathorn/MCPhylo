"""
    to_df(root::GeneralNode)::Tuple{Array{Float64}, Vector{String}}

This function returns a matrix representation of the tree structure and a vector with the column names.
The entry `mat[i,j]` is the length of the edge connecting node `i` with node `j`.
Returns Tuple containing the matrix and a vector of names.

* `root` : root of tree used to create matrix represenation.
"""
function to_df(root::GeneralNode)::Tuple{Array{Float64}, Vector{String}}

    post_order_iteration = post_order(root)

    name_list = [i.name for i in post_order_iteration]
    temp_ar = zeros(Float64, (length(post_order_iteration), length(post_order_iteration)))
    for i in post_order_iteration
        if i.nchild != 0
            ind = indexin(i.name, name_list)
            for j in i.child
                ind2 = indexin(j.name, name_list)
                temp_ar[ind[1], ind2[1]] = j.inc_length
            end # end for
        end # end if
    end # end for

    return df, name_list
end # end function to_df


"""
    from_df(df::Array{Float64,2}, name_list::Vector{String})::GeneralNode

This function takes an adjacency matrix and a vector of names
and turns it into a tree. No checks are performed.

Returns the root node of the tree.

* `df` : matrix with edge weights
* `name_list` : a list of names such that they match the column indices of the matrix
"""
function from_df(df::Array{Float64,2}, name_list::Vector{String})::FNode

    node_list::Vector{FNode} = [Node(String(i)) for i in name_list]

    for (col_index, col) in enumerate(eachcol(df))
        for (row_index, entry) in enumerate(col)
            if entry != 0
                # there is a branch connecting these nodes
                node_list[col_index].inc_length = entry
                add_child!(node_list[row_index], node_list[col_index])
            end # end if
        end # end for
    end # end for
    i::Int = 0
    # find the root
    for (ind, n) in enumerate(node_list)
        if n.root == true
            i = ind
            break
        end # end if
    end # for
    node::FNode = node_list[i]

    # do some bookeeping here and set the binary representation of the nodes
    set_binary!(node)
    number_nodes!(node)
    return node
end # function from_df

"""
    newick(root::T)::String  where T<:GeneralNode
Creates a newick representation of the tree.

Returns a properly formatted newick String.

* `node` : root node of tree used to create the newick string.
"""
function newick(root::T)::String  where T<:GeneralNode
    # get the newickstring
    newickstring = newick(root, "")

    # Some polishing.
    newickstring = chop(newickstring)
    newickstring = string(newickstring, ";")
    return newickstring
end

"""
    newick(root::T, newickstring::AbstractString) where T<:GeneralNode

Do the newick recursion. It is meant as the internal iterator function.
"""
function newick(root::T, newickstring::AbstractString) where T<:GeneralNode
    if root.nchild != 0
        # internal node
        newickstring = string(newickstring, "(")
        for child in root.children
            newickstring = string(newick(child,newickstring))
        end # for
        newickstring = chop(newickstring)
        return string(newickstring,")", root.name, ":", root.inc_length,",")

    else
        # leave
        return string(newickstring, root.name, ":", root.inc_length, ",")
    end # if
end


#################### Covariance wrapper ####################
function to_covariance(tree::TreeStochastic)::Array{Float64,2}
    blv::Vector = get_branchlength_vector(tree)
    to_covariance(tree.value, blv)
end # end to_covariance

function to_covariance(tree::N) where N<:GeneralNode
    blv = get_branchlength_vector(tree)
    to_covariance(tree, blv)
end # end to_covariance

function to_covariance(tree::TreeStochastic, blv::Vector{T})::Array{T,2} where T<: Real
    to_covariance(tree.value, blv)
end # end to_covariance

"""
    to_covariance_ultra(tree::N)::Array{R,2} where {N <:GeneralNode{R,I}} where {R,I}

Get the covariance matrix of the ultrametric version of `tree` with height 1.

Returns an Array of Real numbers.

* `tree` : root of tree used to perform calculation.

"""
function to_covariance_ultra(tree::N)::Array{R,2} where {N <:GeneralNode{R,I}} where {R,I}
    # scale the branchlength between 0 and 1
    blv = get_branchlength_vector(tree)
    blv ./= tree_height(tree)

    # copy the tree, to maintain the original one
    root = deepcopy(tree)
    set_branchlength_vector!(root, blv)
    # make the tree ultrametric
    force_ultrametric!(root)

    # calculate variance-covariance matrix
    to_covariance(root, blv)
end # end function to_covariance_ultra


#################### To Matrix convertes ####################
"""
    to_distance_matrix(tree::T)::Array{Float64,2} where T <:GeneralNode

Calculate the distance matrix over the set of leaves.

Returns an Array of Floats.

* `tree` : root node of tree used to perform caclulcation.
"""
function to_distance_matrix(tree::T)::Array{Float64,2} where T <:GeneralNode
    leaves::Vector{T} = get_leaves(tree)
    ll = length(leaves)
    distance_mat = zeros(Float64, ll, ll)
    for i in 1:ll
        for j in 1:ll
            if i>j
                d = node_distance(tree, leaves[i], leaves[j])
                distance_mat[i,j] = d
                distance_mat[j,i] = d
            end # if
        end # for
    end #for
    distance_mat
end # function to_distance_matrix

"""
    to_covariance(tree::N, blv::Array{T})::Array{T,2} where {N<:GeneralNode,T<: Real}

Calcualte the variance-covariance matrix from `tree`. An entry (i,j) of the matrix
is defined as the length of the path connecting the latest common ancestor
of i and j with the root of the tree.

Returns an Array of Real numbers.

* `tree` : Node in tree of interest.

* `blv` : branchlength vector of tree.

"""
function to_covariance(tree::N, blv::Vector{T})::Array{T,2} where {N<:GeneralNode,T<: Real}
    leaves::Vector{N} = get_leaves(tree)
    ll = length(leaves)
    covmat = zeros(T, ll, ll)
    #@inbounds for ((ind,itm),(jnd,jtm)) in Iterators.product(enumerate(leaves), enumerate(leaves))
    @inbounds for ind = 1:ll, jnd = 1:ind
        itm = leaves[ind]
        if ind == jnd
            covmat[ind,jnd] = covmat[ind,jnd] + reduce(+, @view blv[get_path(tree, itm)])
        else

            lca = find_lca(tree, itm, leaves[jnd])
            if !lca.root
                tmp = reduce(+, @view blv[get_path(tree, lca)])
                covmat[ind,jnd] = covmat[ind,jnd] + tmp
                covmat[jnd,ind] = covmat[jnd,ind] + tmp
            end # if
        end # if

    end # for
    covmat
end# function to_covariance


function to_covariance_func(tree::N)::Array{Function,2} where {N<: GeneralNode}
    leaves = get_leaves(tree)
    ll = length(leaves)
    covmat = Array{Function, 2}(undef, ll, ll)
    @inbounds for ind = 1:ll, jnd = 1:ind
        itm = leaves[ind]
        if ind == jnd
            sympath = get_path(tree, itm)
            covmat[ind,jnd] = y -> sum(y[sympath])
        else

            lca = find_lca(tree, itm, leaves[jnd])
            if !lca.root
                sympath = get_path(tree, lca)
                covmat[ind,jnd] = y -> sum(y[sympath])
                covmat[jnd,ind] = y -> covmat[ind,jnd](y)
            else
                covmat[ind,jnd] = y -> 0.0
                covmat[jnd,ind] = y -> 0.0
            end # if

        end # if

    end # for
    covmat
end# function to_covariance

function to_covariance_ultra(tree::GeneralNode)
    mv = tree_height(tree)
    blv = get_branchlength_vector(tree)
    blv ./= mv
    root = deepcopy(tree)
    set_branchlength_vector!(root, blv)
    force_ultrametric(root)

    leaves = get_leaves(root)
    ll = length(leaves)
    covmat = zeros(ll, ll)
    for i in 1:ll
        node1 = leaves[i]
        for j in 1:ll
            if i == j

                path = path_length(root, node1)
                covmat[i,j] = path
            elseif i>j
                lca = find_lca(root, node1, leaves[j])
                if lca.root != true
                    path = path_length(root, lca)
                    covmat[i,j] = path
                    covmat[j,i] = path


                end
            end
        end

    end
    covmat
end
