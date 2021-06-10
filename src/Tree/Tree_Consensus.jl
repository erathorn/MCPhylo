#=
    The consensus tree functions are implemented on basis of the following paper:

   Jesper Jansson, Chuanqi Shen, and Wing-Kin Sung. 2016. Improved algorithms
   for constructing consensustrees. J. ACM 63, 3, Article 28 (June 2016), 24 pages

   Link: https://dl.acm.org/doi/pdf/10.1145/2925985

   Most of the major functions include a note to find the section of the paper
   they were based on.
=#


"""
    majority_consensus_tree(trees::Vector{T}, percentage::Float64=0.5)
        ::T where T<:GeneralNode

Construct the majority rule consensus tree from a set of trees that share the
same leafset. By default the output tree includes clusters that occur in over
50% of the trees. This can be customized when calling the function. The function
returns the root node of the majority consensus tree, from which it can be
traversed. The algorithm is based on section 3 and 6.1 of:

Jesper Jansson, Chuanqi Shen, and Wing-Kin Sung. 2016. Improved algorithms
for constructing consensustrees. J. ACM 63, 3, Article 28 (June 2016), 24 pages
https://dl.acm.org/doi/pdf/10.1145/2925985
"""
function majority_consensus_tree(trees::Vector{T}, percentage::Float64=0.5)::T where T<:GeneralNode
    merged_tree = deepcopy(trees[1])
    check_leafsets(trees)
    nodes = post_order(merged_tree)
    # save leaf ranks to order the resulting tree in the end
    leaf_ranks = get_leaf_ranks(nodes)
    node_counts = convert(Vector{Int64}, ones(length(nodes)))
    count_dict = Dict(zip(nodes, node_counts))
    for tree in trees[2:end]
        nodes = level_order(merged_tree)
        is_common_cluster = find_common_clusters(tree, merged_tree)
        for node in nodes
            # increment count of clusters of the merged_tree that are in the other tree
            if is_common_cluster[node.num][1] == true
                count_dict[node] += 1
            # delete clusters which are not
            else
                count_dict[node] -= 1
                if count_dict[node] == 0
                    delete_node!(node)
                end # if
            end # else
        end # for
        set_binary!(merged_tree)
        number_nodes!(merged_tree)
        compatible_tree = one_way_compatible(tree, merged_tree)
        inserted_nodes = merge_trees!(compatible_tree, merged_tree)
        # intialize counts for the new nodes
        for node in inserted_nodes
            count_dict[node] = 1
        end # for
    end # for
    # find clusters of the final tree in the other tree, store inc_length and count
    set_node_stats!(merged_tree, trees, true, percentage)
    # order the resulting tree
    nodes = post_order(merged_tree)
    cluster_start_indeces = Dict{T, Int64}()
    for node in nodes
        if node.nchild != 0
            cluster_start_indeces[node] = cluster_start_indeces[node.children[1]]
        else
            cluster_start_indeces[node] = leaf_ranks[node.name]
        end # if/else
    end # for
    order_tree!(merged_tree, cluster_start_indeces)
    return merged_tree
end


"""
    loose_consensus_tree(trees::Vector{T})::T where T<:GeneralNode

Construct the loose consensus tree from a set of trees that share the same
leafset. I.e. a tree with all the clusters that appear in at least one tree
and are compatible with all trees. Returns the root node of the loose consensus
tree, from which it can be traversed. This algorithm is based on section 4 and
6.1 of:

Jesper Jansson, Chuanqi Shen, and Wing-Kin Sung. 2016. Improved algorithms
for constructing consensustrees. J. ACM 63, 3, Article 28 (June 2016), 24 pages
https://dl.acm.org/doi/pdf/10.1145/2925985
"""
function loose_consensus_tree(trees::Vector{T})::T where T<:GeneralNode
    r_tree = trees[1]
    check_leafsets(trees)
    nodes = post_order(r_tree)
    # save leaf ranks to order the resulting tree in the end
    leaf_ranks = get_leaf_ranks(nodes)
    for i in 2:length(trees)
        compatible_tree = one_way_compatible(r_tree, trees[i])
        r_tree = deepcopy(trees[i])
        merge_trees!(compatible_tree, r_tree)
    end
    for tree in trees
        r_tree = one_way_compatible(r_tree, tree)
    end
    set_node_stats!(r_tree, trees, false)
    # order the resulting tree
    nodes = post_order(r_tree)
    cluster_start_indeces = Dict{T, Int64}()
    for node in nodes
        if node.nchild != 0
            cluster_start_indeces[node] = cluster_start_indeces[node.children[1]]
        else
            cluster_start_indeces[node] = leaf_ranks[node.name]
        end # if/else
    end # for
    order_tree!(r_tree, cluster_start_indeces)
    return r_tree
end


"""
    greedy_consensus_tree(trees::Vector{T})::T where T<:GeneralNode


Construct the greedy consensus tree from a set of trees that share the same
leafset.  Returns the root node of the greedy consensus tree, from which it can
be traversed. This algorithm is based on section 5 and 6.1 of:

Jesper Jansson, Chuanqi Shen, and Wing-Kin Sung. 2016. Improved algorithms
for constructing consensustrees. J. ACM 63, 3, Article 28 (June 2016), 24 pages
https://dl.acm.org/doi/pdf/10.1145/2925985
"""
function greedy_consensus_tree(trees::Vector{T})::T where T <: GeneralNode
    leafset = Set([n.name for n in get_leaves(trees[1])])
    l = length(leafset)
    check_leafsets(trees)
    nodes = post_order(trees[1])
    # PSEUDOCODE Step 1
    leaf_ranks = get_leaf_ranks(nodes)
    bit_vectors = Vector{BitVector}()
    for tree in trees
        temp_bit_vectors = Dict{T, BitVector}()
        for node in post_order(tree)
            if node.root
                continue
            elseif node.nchild == 0
                bit_vector = falses(l)
                bit_vector[leaf_ranks[node.name]] = 1
                temp_bit_vectors[node] = bit_vector
            else
                bit_vector = temp_bit_vectors[node.children[1]]
                for child in node.children[2:end]
                    bit_vector = bit_vector .| temp_bit_vectors[child]
                end # for
                temp_bit_vectors[node] = bit_vector
            end # if/else
        end # for
        append!(bit_vectors, values(temp_bit_vectors))
    end # for
    # PSEUDOCODE Step 2
    sort!(bit_vectors, alg=QuickSort)
    # PSEUDOCODE Step 3
    cluster_queue = count_cluster_occurences(bit_vectors)
    # PSEUDOCODE Step 4
    greedy_consensus_tree = Node()
    for leaf in get_leaves(trees[1])
        add_child!(greedy_consensus_tree, deepcopy(leaf))
    end # for
    # PSEUDOCODE Step 5
    while !isempty(cluster_queue)
        cluster = dequeue!(cluster_queue)
        # ignore trivial leaf clusters
        sum(cluster) == 1 && continue
        cluster_length = sum(cluster)
        # num stores the number of shared leaves between a node and the current
        # cluster. It also stores the number of leaf nodes of the node.
        num = Dict{T, Vector{Int64}}()
        for node in post_order(greedy_consensus_tree)
            if node.nchild == 0
                cluster[leaf_ranks[node.name]] == 1 ? num[node] = [1, 1] :
                                                      num[node] = [0, 1]
            else
                num[node] = [sum([num[child][1] for child in node.children])]
                n_leaves = 0
                for child in node.children
                        n_leaves += num[child][2]
                end # for
                push!(num[node], n_leaves)
                children = Vector{T}()
                if num[node][1] == cluster_length
                    insert = true
                    for child in node.children
                        if num[child][1] == num[child][2]
                            push!(children, child)
                        elseif num[child][1] != 0
                            insert = false
                            break
                        end # if
                    end # for
                    insert && insert_node!(node, children)
                    number_nodes!(greedy_consensus_tree)
                    break
                end # if
            end # if/else
        end # for
    end # while
    set_node_stats!(greedy_consensus_tree, trees, false)
    return greedy_consensus_tree
end # greedy_consensus_tree


"""
    count_cluster_occurences(bit_vectors::BitVector)
        ::PriorityQueue{BitVector, Int64}

--- INTERNAL ---
Helper function for greedy_consensus_tree that counts the occurrences of each
bit vector representing a cluster. Returns a priority queue.
"""
function count_cluster_occurences(bit_vectors::Vector{BitVector}
                                 )::PriorityQueue{BitVector, Int64}

    cluster_queue = PriorityQueue{BitVector, Int64}()
    start = bit_vectors[1]
    count = 1
    for bit_vector in bit_vectors[2:end]
        if bit_vector == start
            count += 1
        else
            cluster_queue[start] = -count
            start = bit_vector
            count = 1
        end # if/else
    end # for
    cluster_queue[start] = -count
    return cluster_queue
end # count_cluster_occurences


"""
    set_node_stats!(tree::T, trees::Vector{T}, majority::Bool)
        ::Nothing where T<:GeneralNode

--- INTERNAL ---
Helper function for the construction of a consensus tree. Calculates the
inc_lengths and statistics of the nodes in the consensus tree. If dealing with a
(in-progress) majority consensus tree, this function will also delete its
non-majority clusters (handled by the 'majority' boolean).
"""
function set_node_stats!(main_tree::T, trees::Vector{T}, majority::Bool,
                         percentage::Float64=0.0)::Nothing where T <: GeneralNode

    nodes = level_order(main_tree)
    count_dict = Dict(zip([n.num for n in nodes], zeros(Int64, length(nodes))))
    inc_length_dict = Dict{Int64, Vector{Float64}}()
    for tree in trees
        is_common_cluster = find_common_clusters(tree, main_tree)
        for node in nodes
            if is_common_cluster[node.num][1]
                count_dict[node.num] += 1
                try
                    push!(inc_length_dict[node.num], is_common_cluster[node.num][2])
                catch KeyError
                    inc_length_dict[node.num] = [is_common_cluster[node.num][2]]
                end # try/catch
            end # if
        end # for
    end # for
    half = length(trees) * percentage
    # delete non-majority clusters
    for node in nodes
        if majority && count_dict[node.num] <= half
            delete_node!(node)
        else
            node.inc_length = mean(inc_length_dict[node.num])
            node.stats["sd"] = std(inc_length_dict[node.num])
            node.stats["median"] = median(inc_length_dict[node.num])
            node.stats["frequency"] = count_dict[node.num] / length(trees)
        end # if
    end # for
end # set_node_stats


"""
    find_common_clusters(ref_tree, tree:T)
        ::Dict{Int64, Tuple{Bool, Union{Float64, Missing}}}

--- INTERNAL ---
Use Day's algorithm to create a dictionary, that tells us for each node of the
second input tree, if its corresponding cluster is a common cluster of the trees.

Returns dictionary.

* `ref_tree` : Tree used to create dictionary.

* `tree` : Tree used to create dictionary.

"""
function find_common_clusters(ref_tree::T, tree::T) where T<:GeneralNode
    ref_nodes = post_order(ref_tree)
    leaves_dict = Dict{String, Tuple{Int64, Float64}}()
    clusters = Dict{T, Vector{String}}()
    cluster_dict = Dict{Tuple{Int64, Int64}, Int64}()
    dict_count = 1
    for ref_node in ref_nodes
        if ref_node.nchild == 0
            leaves_dict[ref_node.name] = (dict_count, ref_node.inc_length)
            dict_count += 1
        else
            cluster = Vector{String}()
            for child in ref_node.children
                child.nchild == 0 ? push!(cluster, child.name) :
                                    append!(cluster, clusters[child])
            end # for
            clusters[ref_node] = cluster
            if ref_node.root == true || ref_node == ref_node.mother.children[1]
                last_index = leaves_dict[last(cluster)][1]
                first_index = leaves_dict[first(cluster)][1]
                cluster_dict[(first_index, last_index)] = last_index
            else
                last_index = leaves_dict[last(cluster)][1]
                first_index = leaves_dict[first(cluster)][1]
                cluster_dict[(first_index, last_index)] = first_index
            end # if/else
        end # if/else
    end # for
    for value in values(clusters)
        sort!(value)
    end # for
    clusters_reverse = Dict(value => key.inc_length for (key, value) in clusters)
    is_common_cluster = Dict{Int64, Tuple{Bool, Union{Float64, Missing}}}()
    clusters = Dict{T, Vector{String}}()
    leaf_count = 0
    nodes = post_order(tree)
    for node in nodes
        if node.nchild == 0
            leaf_count += 1
            try
                leaves_dict[node.name]
            catch KeyError
                throw(ArgumentError(
                      "The leafs sets of the trees need to be identical"
                      ))
            end # try
            is_common_cluster[node.num] = (true, leaves_dict[node.name][2])
        else
            cluster = Vector{String}()
            for child in node.children
                child.nchild == 0 ? push!(cluster, child.name) :
                                    append!(cluster, clusters[child])
            end # for
            clusters[node] = cluster
            cluster_indeces = [leaves_dict[leaf][1] for leaf in cluster]
            if length(cluster) != maximum(cluster_indeces) - minimum(cluster_indeces) + 1
                is_common_cluster[node.num] = (false, missing)
            elseif haskey(cluster_dict, (minimum(cluster_indeces), maximum(cluster_indeces)))
                is_common_cluster[node.num] = (true, clusters_reverse[sort!(cluster)])
            else
                is_common_cluster[node.num] = (false, missing)
            end # if/else
        end # if/else
    end # for
    if leaf_count != length(keys(leaves_dict))
        throw(ArgumentError("The leafs sets of the trees need to be identical"))
    end # if
    return is_common_cluster
end # find_common_clusters


"""
    one_way_compatible(ref_tree::T, tree::T)::T where T<:GeneralNode

--- INTERNAL ---
Takes two trees and returns a copy of the first one, where all the clusters that
are not compatible with the second tree are removed.

Returns root node of edited tree.

* `ref_tree` : tree used to determine clusters.

* `tree` : tree used to determine clusters.
"""
function one_way_compatible(ref_tree::T, tree::T)::T where T <: GeneralNode
    ref_tree_copy = deepcopy(ref_tree)
    ref_nodes = post_order(ref_tree_copy)
    cluster_start_indeces = get_cluster_start_indeces(ref_nodes, tree)
    leaves::Vector{T} = order_tree!(tree, cluster_start_indeces)
    leaf_ranks_reverse = Dict(node.name => ind for (ind, node) in enumerate(leaves))
    xleft_dict, xright_dict = depth_dicts(leaves)
    marked_nodes = Dict{Int64, Bool}()
    for ref_node in ref_nodes
        check_node!(ref_node, leaves, leaf_ranks_reverse, xleft_dict,
                    xright_dict, marked_nodes)
    end # for
    for node in level_order(ref_tree_copy)
        if node.nchild != 0 && marked_nodes[node.num] == true
            delete_node!(node)
        end # if
    end # for
    set_binary!(ref_tree_copy)
    number_nodes!(ref_tree_copy)
    return ref_tree_copy
end # function one_way_compatible


"""
    merge_trees!(ref_tree::T, tree::T)::Vector{T}} where T<:GeneralNode

--- INTERNAL ---
Merge two compatible trees, i.e. inserts all cluster of the first tree, which
aren't already in the second tree, into the second tree

Returns a Tuple containing the root node of the merged tree, as well as a vector
of nodes that were added.

* `ref_tree` : tree used to determine common clusters.

* `tree` : tree used to determine common clusters.
"""
function merge_trees!(ref_tree::T, tree::T)::Vector{T} where T<:GeneralNode
    ref_nodes = post_order(ref_tree)
    cluster_start_indeces = get_cluster_start_indeces(ref_nodes, tree)
    leaves::Vector{T} = order_tree!(tree, cluster_start_indeces)
    leaf_ranks_reverse = Dict(node.name => ind for (ind, node) in enumerate(leaves))
    xleft_dict, xright_dict = depth_dicts(leaves)
    inserted_nodes = Vector{T}()
    count = -1
    for ref_node in ref_nodes
        tuple = check_node!(ref_node, leaves, leaf_ranks_reverse, xleft_dict,
                            xright_dict)
        isnothing(tuple) && continue
        d, e, r, start_node, stop_node, left_path, right_path = tuple
        left = d == r.children[1]
        right = e == r.children[end]
        if !(left && right)
            index_d = findfirst(x -> x == d, r.children)
            index_e = findfirst(x -> x == e, r.children)
            inserted_node = insert_node!(r, r.children[index_d:index_e])
            push!(inserted_nodes, inserted_node)
            # ensures correct depth
            inserted_node.binary = string(r.binary, ",z")
            # give unique number to avoid false positive "==" statements
            inserted_node.num = count
            count -= 1
            inserted_depth = node_depth(inserted_node)
            if !left
                left_path[inserted_depth] = inserted_node
                right_path[inserted_depth] = inserted_node
                xleft_dict[start_node] = (inserted_node, left_path)
            end
            if !right
                right_path[inserted_depth] = inserted_node
                left_path[inserted_depth] = inserted_node
                xright_dict[stop_node] = (inserted_node, right_path)
            end # if
        end # if
    end # for
    set_binary!(tree)
    return inserted_nodes
end # function merge_trees!


"""
    check_node!(
        ref_node::T, leaves::Vector{T},
        leaf_ranks_reverse::Dict{String, Int64},
        xleft_dict::Dict{T, Tuple{T, Dict{Int64, T}}},
        xright_dict::Dict{T, Tuple{T, Dict{Int64, T}}},
        marked_nodes::Union{Dict{Int64, Bool, Nothing}}=nothing
        )::Union{Nothing, Tuple{T, T, T, Dict{Int64, T}, Dict{Int64, T}}} where T<:GeneralNode

--- INTERNAL ---
Helper function that handles major chunk of code that one_way_compatible and
merge_tree share
"""
function check_node!(ref_node::T, leaves::Vector{T},
                     leaf_ranks_reverse::Dict{String, Int64},
                     xleft_dict::Dict{T, Tuple{T, Dict{Int64, T}}},
                     xright_dict::Dict{T, Tuple{T, Dict{Int64, T}}},
                     marked_nodes::Union{Dict{Int64, Bool}, Nothing}=nothing
                     )::Union{Nothing, Tuple{T, T, T, T, T, Dict{Int64, T}, Dict{Int64, T}}} where T<:GeneralNode
    if ref_node.nchild != 0
        start = min_leaf_rank(leaf_ranks_reverse, ref_node)
        stop = max_leaf_rank(leaf_ranks_reverse, ref_node)
        start_node = leaves[start]
        stop_node = leaves[stop]
        xleft, left_path = xleft_dict[start_node]
        xright, right_path = xright_dict[stop_node]
        if intersect(values(left_path), values(right_path)) == []
            try marked_nodes[ref_node.num] = true catch; end # try
            return
        end # if
        !xleft.root && delete!(left_path, node_depth(xleft.mother))
        !xright.root && delete!(right_path, node_depth(xright.mother))
        depth_left = node_depth(xleft)
        depth_right = node_depth(xright)
        depth = depth_left >= depth_right
        depth ? p2 = right_path[depth_left] : p2 = left_path[depth_right]
        depth ? p1 = xleft : p1 = xright
        if p2 == p1
            r = p1
            if depth
                d = left_path[depth_left + 1]
                e = right_path[depth_left + 1]
            else
                d = left_path[depth_right + 1]
                e = right_path[depth_right + 1]
            end # if/else
        elseif p1.mother == p2.mother
            r = p1.mother
            depth ? d = p1 : d = p2
            depth ? e = p2 : e = p1
        else
            try marked_nodes[ref_node.num] = true catch; end # try
            return
        end # if/else
        if !(node_depth(d.mother) <= node_depth(r) && node_depth(e.mother) <= node_depth(r))
            try marked_nodes[ref_node.num] = true catch; return end # try
        else
            try marked_nodes[ref_node.num] = false
            catch
                isnothing(marked_nodes) && return (d, e, r, start_node, stop_node, left_path, right_path)
            end # try
            return
        end # if/else
    end # if
end # check_node


"""
    get_cluster_start_indeces(ref_nodes::T, tree::T)
        ::Dict{T, Int64} where T<:GeneralNode

--- INTERNAL ---
Helper function to obtain the cluster start indeces for a tree (tree), based
on the nodes of another tree (ref_nodes).
"""
function get_cluster_start_indeces(ref_nodes::Vector{T}, tree::T
                                  )::Dict{T, Int64} where T<:GeneralNode

    leaf_ranks = get_leaf_ranks(ref_nodes)
    leaves = Vector{T}()
    cluster_start_indeces = Dict{T, Int64}()
    nodes = post_order(tree)
    for node in nodes
        if node.nchild != 0
            for leaf in leaves
                path = split(node.binary, ",")
                if node_depth(node) <= node_depth(leaf) && path == split(leaf.binary, ",")[1:length(path)]
                    cluster_start_indeces[node] = leaf_ranks[leaf.name]
                    break
                end # if
            end # for
        else
            cluster_start_indeces[node] = leaf_ranks[node.name]
            push!(leaves, node)
        end # if / else
    end # for
    return cluster_start_indeces
end


"""
    function get_leaf_ranks(nodes::Vector{T})
        ::Dict{String, Int64} where T<:GeneralNode

--- INTERNAL ---
Enumerate the leaf nodes in a tree. Returns a dictionary of this mapping.
"""
function get_leaf_ranks(nodes::Vector{T}
                       )::Dict{String, Int64} where T<:GeneralNode

    leaf_ranks = Dict{String, Int64}()
    count = 1
    for node in nodes
        if node.nchild == 0
            leaf_ranks[node.name] = count
            count += 1
        end # if
    end # for
    return leaf_ranks
end # get_leaf_ranks


"""
    order_tree!(root::T, cluster_start_indeces::Dict{T, Int64}, leaves=Vector{T}())
        ::Vector{T} where T<:GeneralNode

--- INTERNAL ---
Helper function to order a tree based on cluster indeces and return the leaves
of the ordered tree
"""
function order_tree!(root::T, cluster_start_indeces::Dict{T, Int64},
                     leaves=Vector{T}())::Vector{T} where T<:GeneralNode

    sort!(root.children, by = child -> cluster_start_indeces[child],
          alg=InsertionSort)
    for child in root.children
        if child.nchild == 0
            push!(leaves, child)
        else
            order_tree!(child, cluster_start_indeces, leaves)
        end # if/else
    end # for
    set_binary!(root)
    return leaves
end # order_tree!


"""
    min_leaf_rank(leaf_ranks::Dict{String, Int64}, node::T)
        ::Int64 where T<:GeneralNode

---- INTERNAL ---
Recursive helper function to find the lowest ranked leaf descendant of a node
"""
function min_leaf_rank(leaf_ranks::Dict{String, Int64}, node::T
                      )::Int64 where T<:GeneralNode

    if node.nchild == 0
        return leaf_ranks[node.name]
    else
        possible_minima = Vector{Int64}()
        for child in node.children
            if child.nchild == 0
                push!(possible_minima, leaf_ranks[child.name])
            else
                push!(possible_minima, min_leaf_rank(leaf_ranks, child))
            end # if/else
        end # for
    end # if/else
    minimum(possible_minima)
end # min_leaf_rank


"""
    max_leaf_rank(leaf_ranks::Dict{String, Int64}, node::T)
        ::Int64 where T<:GeneralNode

--- INTERNAL ---
Recursive helper function to find the highest ranked leaf descendant of a node
"""
function max_leaf_rank(leaf_ranks::Dict{String, Int64}, node::T
                      )::Int64 where T<:GeneralNode

    if node.nchild == 0
        return leaf_ranks[node.name]
    else
        possible_maxima = Vector{Int64}()
        for child in node.children
            if child.nchild == 0
                push!(possible_maxima, leaf_ranks[child.name])
            else
                push!(possible_maxima, max_leaf_rank(leaf_ranks, child))
            end # if/else
        end # for
    end # if/else
    maximum(possible_maxima)
end # max_leaf_rank


"""
    x_left(node::T)::Tuple{T,Vector{T}} where T<:GeneralNode

--- INTERNAL ---
Helper function to find ancestor of a leaf that has said leaf as leftmost
descendant. Also returns the path from the leaf to the mother of that node.
"""
function x_left(node::T)::Tuple{T, Vector{T}} where T<:GeneralNode
    path::Vector{T} = [node]
    while true
        if node.root
            return node, path
        else
            mother = node.mother
            if mother.children[1] != node
                push!(path, node.mother)
                return node, path
            end # if
        node = mother
        push!(path, node)
        end # if/else
    end # while
end # x_left


"""
    x_right(node::T)::Tuple{T,Vector{T}} where T<:GeneralNode

--- INTERNAL ---
Helper function to find ancestor of a leaf that has said leaf as rightmost
descendant. Also returns the path from the leaf to the mother of that node.
"""
function x_right(node::T)::Tuple{T, Vector{T}} where T<:GeneralNode
    path::Vector{T} = [node]
    while true
        if node.root
            return node, path
        else
            mother = node.mother
            if mother.children[end] != node
                push!(path, node.mother)
                return node, path
            end # if
        node = mother
        push!(path, node)
        end # if/else
    end # while
end # x_right


"""
    depth_dicts(leaves::Vector{T})
        ::Tuple{Dict{T, Tuple{T, Dict{Int64, T}}}, Dict{T, Tuple{T, Dict{Int64, T}}}} where T<:GeneralNode

--- INTERNAL ---
Helper function for one_way_compatible and merge_trees!. Creates a dictionary
based on a vector of leaf nodes, and stores the depth of each node, as well as
the left and right path leading to it. Based on section 6.1 of the paper.
"""
function depth_dicts(leaves::Vector{T})::Tuple{Dict{T, Tuple{T, Dict{Int64, T}}}, Dict{T, Tuple{T, Dict{Int64, T}}}} where T<:GeneralNode
    xleft_dict = Dict{T, Tuple{T, Dict{Int64, T}}}()
    xright_dict = Dict{T, Tuple{T, Dict{Int64, T}}}()
    for leaf in leaves
        x, path = x_left(leaf)
        xleft_dict[leaf] = (x, Dict(node_depth(node) => node for node in path))
        x, path = x_right(leaf)
        xright_dict[leaf] = (x, Dict(node_depth(node) => node for node in path))
    end
    return xleft_dict, xright_dict
end
