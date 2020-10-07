
#TODO: RSPR

"""
    NNI(root::T, target::T, lor::Bool)::Int64   where T<:AbstractNode

This function does a nearest neighbour interchange (NNI) move on the tree specified
by `root`. The parameter `target` specifies the node which performs the interchange
move using the left or right child of the target node. If the left child should
be used `lor=true`.
The function returns 1 if the move was successfull and 0 else.
"""
function NNI!(root::T, target::T, lor::Bool)::Int64  where T<:AbstractNode
    # NNI move would be illegal
    if target.nchild == 0 || target.root
        return 0
    end # if

    parent::T = get_mother(target)
    sister::T = get_sister(target)

    ychild::T = remove_child!(target, lor)
    xchild::T = remove_child!(parent, sister)

    add_child!(target, sister)
    add_child!(parent, ychild)

    set_binary!(root)

    return 1

end # function

"""
    NNI!(root::T, target::Int64)::Int64  where T<:AbstractNode

This function does a nearest neighbour interchange (NNI) move on the tree specified
by `root`. The target is identified by the number of the target node.
The function returns 1 if the move was successfull and 0 else.
"""
function NNI!(root::T, target::Int64)::Int64  where T<:AbstractNode
   tn::T = find_num(root, target)
   lor::Bool = 0.5 > rand()
   NNI!(root, tn, lor)
end #function

"""
    NNI!(root::T)::Int64  where T<:AbstractNode

This function does a nearest neighbour interchange (NNI) move on the tree specified
by `root`. The target is identified by the number of the target node.
The function returns 1 if the move was successfull and 0 else.
"""
function NNI!(root::T)::Int64  where T<:AbstractNode
    n = rand(1:size(root)[1])
    tn::T = find_num(root, n)
    lor::Bool = 0.5 > rand()
    NNI!(root, tn, lor)
end #function


"""
    slide!(root::Node)

This functin performs a slide move on an intermediate node. The node is moved
upwards or downwards on the path specified by its mother and one of its
daughters.
"""
function slide!(root::Node)
    throw("I need repair")
    target::Node = Node(1.0, [0.0], Node[], 0, true, 0.0, "0")
    while true
        target = random_node(root)
        # check if target is not a leave and that its grand daughters are also
        # no leaves
        if target.nchild != 0
            if target.child[1].nchild !=0
                if target.child[2].nchild !=0
                    break
                end
            end # if
        end # if
    end # end while

    # proportion of slide move is randomly selected
    proportion::Float64 = rand(Uniform(0,1))

    # pick a random child
    child::Node = target.child[rand([1,2])]

    # calculate and set new values
    move!(target, child, proportion)

end # function slide!

"""
    swing!(root::Node)

This function performs a swing node. A random non-leave node is selected and
moved along the path specified by its two children.
"""
function swing!(root::Node)
    throw("I need repair")
    target::Node = Node(1.0, [0.0], Node[], 0, true, 0.0, "0")
    while true
        target = random_node(root)
        # check if target is not a leave
        if target.nchild != 0
            break
        end # if
    end # end while

    proportion::Float64 = rand(Uniform(0,1))

    child1 = target.child[1]
    child2 = target.child[2]

    # calculate and set new values
    move!(child1, child2, proportion)
end # function swing!


"""
    randomize!(root::Node, num::Int64=100)::nothing

This function randomizes the tree topology by performing a number of nearest
neighbour interchange (NNI) moves. The number of NNI moves is specified in
the parameter num.
"""
function randomize!(root::T, num::Int64=100)::Nothing where T <:AbstractNode
    n_nodes = size(root)[1]
    i = 0
    while i < num
        n = rand(1:n_nodes)
        NNI!(root, n)
        i += 1
    end

end



"""
    move!(node1::Node, node2::Node, proportion::Float64)

Change the incomming length of node1 and node2 while keeping there combined length
constant.
"""
function slide!(node1::T, node2::T, proportion::Float64) where T <:AbstractNode
    total::Float64 = node1.inc_length + node2.inc_length
    fp::Float64 = total*proportion
    sp::Float64 = total-fp
    node1.inc_length = fp
    node2.inc_length = sp
end # function slide!


### Experimental rerooting



function reroot(root::T, new_root::String)::T where T<:Node

    new_tree = deepcopy(root)
    root_node = find_by_name(new_tree, new_root)

    mother = root_node.mother

    recursive_invert(mother, root_node)

    root_node.root = true
    new_tree.root = false


    set_binary!(root_node)
    number_nodes!(root_node)
    return root_node
end


function recursive_invert(old_mother::T, old_daughter::T)::T where T
    if old_mother.root == true
        # arrived at the root
        od = remove_child!(old_mother, old_daughter)
        add_child!(od, old_mother)
        return od
    end
        od1 = recursive_invert(old_mother.mother, old_mother)
        od = remove_child!(od1, old_daughter)
        add_child!(od, od1)
        return od
end


"""
    SPR(original_root::Node, binary::Bool)::AbstractNode

Performs SPR on tree; takes reference to root of tree, boolean value necessary to determine if tree should be treated as binary or not
Returns reference to root of altered tree
"""
function SPR(original_root::Node,binary::Bool)::AbstractNode
    root = deepcopy(original_root)
    if length(post_order(root)) <= 2
        error("The tree is too small for SPR")
    end #if

    spr_tree = binary ? perform_spr_binary(root) : perform_spr(root)
    set_binary!(spr_tree)
    return spr_tree
end
"""
    perform_spr(root::Node)::AbstractNode
performs SPR on non-binary tree

"""
function perform_spr(root::Node)::AbstractNode
    subtree_root, nodes_of_subtree = create_random_subtree(root) #returns reference to subtree to be pruned and reattached

    remove_child!(subtree_root.mother,subtree_root)
    spr_tree = merge_randomly(root,subtree_root) #returns reference to root of tree after reattachment
    number_nodes!(spr_tree)
    return spr_tree
end
"""
    perform_spr_binary(root::Node)::AbstractNode

performs SPR on binary tree

"""
function perform_spr_binary(root::Node)::AbstractNode
    subtree_root, nodes_of_subtree = create_random_subtree(root)#returns reference to subtree to be pruned and reattached
    subtree_mom = subtree_root.mother
    if subtree_mom.nchild > 0
        remove_child!(subtree_root.mother,subtree_root)
        other_child = subtree_mom.children[1]
        momlength = subtree_mom.inc_length
        otherlength = other_child.inc_length
        other_child.inc_length = otherlength + momlength
        grandmother = subtree_mom.mother
        remove_child!(grandmother,subtree_mom)
        add_child!(grandmother, other_child)
    else
        remove_child!(subtree_root.mother,subtree_root)
    end #ifelse

    spr_tree = merge_randomly_binary(root,subtree_root)#returns reference to root of tree after reattachment
    number_nodes!(spr_tree)
    return spr_tree
end

"""
    create_random_subtree(root::T) where T<:AbstractNode

selects random, non-root node from tree for use in SPR pruning
"""
function create_random_subtree(root::T)  where T<:AbstractNode
    subtree_root = random_node(root)
    while subtree_root.root || subtree_root.mother.root
        subtree_root = random_node(root)
    end #while
    nodes_of_subtree = post_order(subtree_root) #could be used in conjunction with Tree_Pruning.jl
    return subtree_root, nodes_of_subtree
end #function

"""
    merge_randomly_binary(root::T,subtree_root::T)::T where T<:AbstractNode

reattaches subtree to random, non-root node (binary tree)

"""
function merge_randomly_binary(root::T,subtree_root::T)::T  where T<:AbstractNode
    random_mother = random_node(root)
    while random_mother.root || random_mother.nchild == 0
        random_mother = random_node(root)
    end #while
    if random_mother.nchild > 1 #creates "placeholder node" in binary tree if necessary to preserve binarity
        other_child = random_mother.children[2]
        binary_placeholder_node = Node()
        binary_placeholder_node.name = "nameless" #standardizes name of node; constructor defaults to "no_name"

        incoming_length = random_mother.inc_length
        proportion = rand(Uniform(0,1)) #should be a number between 0 and 1
        half_inc_length = incoming_length*proportion
        other_half = incoming_length - half_inc_length #distributes length of node between placeholder node and child to preserve length
        add_child!(binary_placeholder_node,other_child)
        add_child!(binary_placeholder_node,subtree_root)
        random_mother.inc_length = half_inc_length
        binary_placeholder_node.inc_length = other_half

        add_child!(random_mother,binary_placeholder_node)
        remove_child!(random_mother,other_child)
    else
        add_child!(random_mother,subtree_root) #no need for placeholder node if given node has 1 or fewer children
    end #ifelse
    return root
end #function


"""
    merge_randomly(root::T,subtree_root::T)::T where T<:AbstractNode

reattaches subtree to random, non-root node (tree)

"""

function merge_randomly(root::T,subtree_root::T)::T  where T<:AbstractNode
    random_mother = random_node(root)
    while random_mother.root
        random_mother = random_node(root)
    end #while
        add_child!(random_mother,subtree_root) #no need for placeholder node if given node has 1 or fewer children
    return root
end #function


#TODO: TBR
#1 step: checks >4
#binary way:
##1. find the middle (walk to the middle https://cs.stackexchange.com/questions/42617/worst-case-bisection-of-binary-tree)
##2. remove the tree rooted on this node
##3. reuse merge_randomly_binary
#non_binary way:
##1. ALSO FINDING BISECTION POINT AS ABOVE
##2. raking the tree rooted
##3. merge_randomly

"""
    TBR(original_root::Node, binary::Bool)::AbstractNode

#
Returns reference to root of altered tree
"""
function TBR(original_root::Node,binary::Bool)::AbstractNode
    root = deepcopy(original_root)
    if length(post_order(root)) < 4
        error("The tree is too small for TBR")
    end #if

    tbr_tree = binary ? perform_tbr_binary(root) : perform_tbr(root)
    set_binary!(tbr_tree)
    return tbr_tree
end


function finding_bisection_point(root::Node)
    node_total = length(post_order(root))
    twothirds = node_total*.66
    onethird = node_total*.33
    if length(post_order(root)) < 4
        #TODO: ask Wahle
        print("too small")
    else
        for x in post_order(root)
            if length(post_order(x)) < (twothirds) && length(post_order(x)) > onethird
                goodnode = x
                return goodnode
            end #if
        end #for
    end #if
    error("something went wrong; make sure tree is formatted correctly")
    return root
end #function




"""
    perform_tbr(root::Node)::AbstractNode
performs TBR on non-binary tree

"""
function perform_tbr(root::Node)::AbstractNode
    subtree_root = finding_bisection_point(root) #returns reference to subtree to be pruned and reattached
    println("SUBTREE HERE")
    println(newick(subtree_root))
    remove_child!(subtree_root.mother,subtree_root)
    tbr_tree = merge_randomly(root,subtree_root) #returns reference to root of tree after reattachment
    number_nodes!(tbr_tree)
    return tbr_tree
end
"""
    perform_tbr_binary(root::Node)::AbstractNode

performs TBR on binary tree

"""
function perform_tbr_binary(root::Node)::AbstractNode
    subtree_root = finding_bisection_point(root)#returns reference to subtree to be pruned and reattached
    println("HERE IS SUBTREE ROOT")
    println(newick(subtree_root))
    subtree_mom = subtree_root.mother
    if subtree_mom.nchild > 0
        remove_child!(subtree_root.mother,subtree_root)
        other_child = subtree_mom.children[1]
        momlength = subtree_mom.inc_length
        otherlength = other_child.inc_length
        other_child.inc_length = otherlength + momlength
        grandmother = subtree_mom.mother
        remove_child!(grandmother,subtree_mom)
        add_child!(grandmother, other_child)
    else
        remove_child!(subtree_root.mother,subtree_root)
    end #ifelse

    tbr_tree = merge_randomly_binary(root,subtree_root)#returns reference to root of tree after reattachment
    number_nodes!(tbr_tree)
    return tbr_tree
end
