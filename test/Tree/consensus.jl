using MCPhylo
using Test

@testset "find_common_clusters" begin
    ref_tree = MCPhylo.parsing_newick_string("((A,(B,(C,(D,E)))),(F,(G,H)));")
    tree2 = MCPhylo.parsing_newick_string("((G,(C,(A,(F,E)))),(B,(D,H)));")
    MCPhylo.number_nodes!.([ref_tree, tree2])
    MCPhylo.set_binary!.([ref_tree, tree2])
    A, B, C, D, E, F, G, H = MCPhylo.find_by_name(tree2, "A"), MCPhylo.find_by_name(tree2, "B"),
                             MCPhylo.find_by_name(tree2, "C"), MCPhylo.find_by_name(tree2, "D"),
                             MCPhylo.find_by_name(tree2, "E"), MCPhylo.find_by_name(tree2, "F"),
                             MCPhylo.find_by_name(tree2, "G"), MCPhylo.find_by_name(tree2, "H")
     expected_dict = Dict{Int64,Tuple{Bool,Union{Missing, Float64}}}([(D.mother.num, (false, missing)), (C.mother.num, (false, missing)), (B.mother.num, (false, missing)),
                           (A.mother.num, (false, missing)), (G.mother.num, (false, missing)), (F.mother.num, (false, missing)),
                           (G.mother.mother.num, (true, 1.0)), (A.num, (true, 1.0)), (B.num, (true, 1.0)), (C.num, (true, 1.0)),
                           (D.num, (true, 1.0)), (E.num, (true, 1.0)), (F.num, (true, 1.0)), (G.num, (true, 1.0)), (H.num, (true, 1.0))])
    @test isequal(MCPhylo.find_common_clusters(ref_tree, tree2) , expected_dict)

    tree3 = MCPhylo.parsing_newick_string("((A,(C,(D,(B,E)))),(G,(F,H)));")
    MCPhylo.number_nodes!(tree3)
    MCPhylo.set_binary!(tree3)
    A, B, C, D, E, F, G, H = MCPhylo.find_by_name(tree3, "A"), MCPhylo.find_by_name(tree3, "B"),
                             MCPhylo.find_by_name(tree3, "C"), MCPhylo.find_by_name(tree3, "D"),
                             MCPhylo.find_by_name(tree3, "E"), MCPhylo.find_by_name(tree3, "F"),
                             MCPhylo.find_by_name(tree3, "G"), MCPhylo.find_by_name(tree3, "H")
     expected_dict = Dict([(D.mother.num, (false, missing)), (C.mother.num, (true, 1.0)), (B.mother.num, (false, missing)),
                           (A.mother.num, (true, 1.0)), (G.mother.num, (true, 1.0)), (F.mother.num, (false, missing)),
                           (A.mother.mother.num, (true, 1.0)), (A.num, (true, 1.0)), (B.num, (true, 1.0)), (C.num, (true, 1.0)),
                           (D.num, (true, 1.0)), (E.num, (true, 1.0)), (F.num, (true, 1.0)), (G.num, (true, 1.0)), (H.num, (true, 1.0))])
    @test isequal(MCPhylo.find_common_clusters(ref_tree, tree3), expected_dict)

    tree4 = MCPhylo.parsing_newick_string("((A,(B,(C,(D,E)))),(F,(G,H)));")
    MCPhylo.number_nodes!(tree4)
    MCPhylo.set_binary!(tree4)
    A, B, C, D, E, F, G, H = MCPhylo.find_by_name(tree4, "A"), MCPhylo.find_by_name(tree4, "B"),
                             MCPhylo.find_by_name(tree4, "C"), MCPhylo.find_by_name(tree4, "D"),
                             MCPhylo.find_by_name(tree4, "E"), MCPhylo.find_by_name(tree4, "F"),
                             MCPhylo.find_by_name(tree4, "G"), MCPhylo.find_by_name(tree4, "H")
    expected_dict = Dict([(D.mother.num, (true, 1.0)), (C.mother.num, (true, 1.0)), (B.mother.num, (true, 1.0)),
                          (A.mother.num, (true, 1.0)), (G.mother.num, (true, 1.0)), (F.mother.num, (true, 1.0)),
                          (A.mother.mother.num, (true, 1.0)), (A.num, (true, 1.0)), (B.num, (true, 1.0)), (C.num, (true, 1.0)),
                          (D.num, (true, 1.0)), (E.num, (true, 1.0)), (F.num, (true, 1.0)), (G.num, (true, 1.0)), (H.num, (true, 1.0))])
    @test isequal(MCPhylo.find_common_clusters(ref_tree, tree4), expected_dict)

    tree5 = MCPhylo.parsing_newick_string("((G,(X,(A,(F,E)))),(B,(D,H)));")
    MCPhylo.number_nodes!(tree5)
    MCPhylo.set_binary!(tree5)
    @test_throws ArgumentError MCPhylo.find_common_clusters(ref_tree, tree5)
    
    tree6 = MCPhylo.parsing_newick_string("(X,(G,(C,(A,(F,E)))),(B,(D,H)));")
    MCPhylo.number_nodes!(tree5)
    MCPhylo.set_binary!(tree5)
    @test_throws ArgumentError MCPhylo.find_common_clusters(ref_tree, tree6)
end

@testset "one_way_compatible" begin
    tree = MCPhylo.parsing_newick_string("((A,C,E),(B,D));")
    tree2 = MCPhylo.parsing_newick_string("((A,C),(B,D,E));")
    expected_tree = MCPhylo.parsing_newick_string("(A,C,E,(B,D));")
    MCPhylo.number_nodes!.([tree, tree2, expected_tree])
    MCPhylo.set_binary!.([tree, tree2, expected_tree])
    @test MCPhylo.newick(MCPhylo.one_way_compatible(tree, tree2)) == MCPhylo.newick(expected_tree)
end

@testset "get_leaf_ranks" begin
    ref_tree = MCPhylo.parsing_newick_string("((A,(B,(C,(D,E)))),(F,(G,H)));")
    nodes = post_order(ref_tree)
    @test MCPhylo.get_leaf_ranks(nodes) == Dict([("A", 1), ("B", 2), ("C", 3),
                                                 ("D", 4), ("E", 5), ("F", 6),
                                                 ("G", 7), ("H", 8)])
end

@testset "order_tree!" begin
    tree = MCPhylo.parsing_newick_string("(A,B,(C,(D,E)F)G)H;")
    MCPhylo.number_nodes!(tree)
    MCPhylo.set_binary!(tree)
    A, B, C, D, E, F, G, H = MCPhylo.find_by_name(tree, "A"), MCPhylo.find_by_name(tree, "B"),
                             MCPhylo.find_by_name(tree, "C"), MCPhylo.find_by_name(tree, "D"),
                             MCPhylo.find_by_name(tree, "E"), MCPhylo.find_by_name(tree, "F"),
                             MCPhylo.find_by_name(tree, "G"), MCPhylo.find_by_name(tree, "H")
    cluster_start_indeces = Dict([(A, 3), (B, 7), (C, 2), (D, 8),
                                  (E, 5), (F, 1), (G, 4), (H, 6)])
    ordered_tree = MCPhylo.parsing_newick_string("(A,((E,D)F,C)G,B)H;")
    MCPhylo.number_nodes!(ordered_tree)
    MCPhylo.set_binary!(ordered_tree)
    @test MCPhylo.order_tree!(tree, cluster_start_indeces) == [A, E, D, C, B]
    MCPhylo.number_nodes!(tree)
    MCPhylo.set_binary!(tree)
    @test MCPhylo.newick(tree) == MCPhylo.newick(ordered_tree)
end

@testset "max/min_leaf_rank" begin
    tree = MCPhylo.parsing_newick_string("(A,B,(C,(D,E)F)G)H;")
    F, G, H, A = MCPhylo.find_by_name(tree, "F"), MCPhylo.find_by_name(tree, "G"),
                 MCPhylo.find_by_name(tree, "H"), MCPhylo.find_by_name(tree, "A")
    leaf_ranks = Dict([("A", 1), ("B", 5), ("C", 2), ("D", 4), ("E", 3)])

    @testset "min_leaf_rank" begin
        @test MCPhylo.min_leaf_rank(leaf_ranks, F) == 3
        @test MCPhylo.min_leaf_rank(leaf_ranks, G) == 2
        @test MCPhylo.min_leaf_rank(leaf_ranks, H) == 1
        @test MCPhylo.min_leaf_rank(leaf_ranks, A) == 1
    end

    @testset "max_leaf_rank" begin
        @test MCPhylo.max_leaf_rank(leaf_ranks, F) == 4
        @test MCPhylo.max_leaf_rank(leaf_ranks, G) == 4
        @test MCPhylo.max_leaf_rank(leaf_ranks, H) == 5
        @test MCPhylo.max_leaf_rank(leaf_ranks, A) == 1
    end
end

@testset "x_left_right" begin
    tree = MCPhylo.parsing_newick_string("(A,B,(C,(D,E)F)G)H;")
    MCPhylo.set_binary!(tree)
    MCPhylo.number_nodes!(tree)
    A, B, C, D, E, F, G, H = MCPhylo.find_by_name(tree, "A"), MCPhylo.find_by_name(tree, "B"),
                             MCPhylo.find_by_name(tree, "C"), MCPhylo.find_by_name(tree, "D"),
                             MCPhylo.find_by_name(tree, "E"), MCPhylo.find_by_name(tree, "F"),
                             MCPhylo.find_by_name(tree, "G"), MCPhylo.find_by_name(tree, "H")

    @testset "x_left" begin
        @test MCPhylo.x_left(A) == (H, [A,H])
        @test MCPhylo.x_left(B) == (B, [B,H])
        @test MCPhylo.x_left(C) == (G, [C,G,H])
        @test MCPhylo.x_left(D) == (F, [D,F,G])
        @test MCPhylo.x_left(E) == (E, [E,F])
    end

    @testset "x_right" begin
        @test MCPhylo.x_right(A) == (A, [A,H])
        @test MCPhylo.x_right(B) == (B, [B,H])
        @test MCPhylo.x_right(C) == (C, [C,G])
        @test MCPhylo.x_right(D) == (D, [D,F])
        @test MCPhylo.x_right(E) == (H, [E,F,G,H])
    end
end

@testset "majority_consensus_tree" begin
    tree1 = MCPhylo.parsing_newick_string("(((A,B),C),(D,E));")
    tree2 = MCPhylo.parsing_newick_string("((A,C),(B,D,E));")
    tree3 = MCPhylo.parsing_newick_string("(((B,C),A),D,E);")
    trees = [tree1, tree2, tree3]
    MCPhylo.number_nodes!.(trees)
    MCPhylo.set_binary!.(trees)
    result = MCPhylo.newick(MCPhylo.parsing_newick_string("((A,B,C),D,E);"))
    @test MCPhylo.newick(MCPhylo.majority_consensus_tree(trees)) == result

    tree4 = MCPhylo.parsing_newick_string("(((B,C),A),D,F);")
    push!(trees, tree4)
    @test_throws ArgumentError MCPhylo.majority_consensus_tree(trees)

    tree1 = MCPhylo.parsing_newick_string("(((A,B)AB,C)ABC,(D,E)DE)no_name;")
    tree2 = MCPhylo.parsing_newick_string("((A,C)AC,(B,D,E)BDE)no_name;")
    tree3 = MCPhylo.parsing_newick_string("(((B,C)BC,A)BCA,D,E)no_name;")
    trees = [tree1, tree2, tree3]
    MCPhylo.number_nodes!.(trees)
    MCPhylo.set_binary!.(trees)
    result = newick(MCPhylo.parsing_newick_string("((A,B,C),D,E);"))
    @test newick(MCPhylo.majority_consensus_tree(trees)) == result
end

@testset "loose_consensus_tree" begin
    tree1 = MCPhylo.parsing_newick_string("(((A,B),C),(D,E));")
    tree2 = MCPhylo.parsing_newick_string("((A,C),(B,D,E));")
    tree3 = MCPhylo.parsing_newick_string("(((B,C),A),D,E);")
    trees = [tree1, tree2, tree3]
    MCPhylo.number_nodes!.(trees)
    MCPhylo.set_binary!.(trees)
    result = newick(MCPhylo.parsing_newick_string("(A,B,C,(D,E));"))
    @test newick(MCPhylo.loose_consensus_tree(trees)) == result

    tree4 = MCPhylo.parsing_newick_string("(((B,C),A),D,F);")
    push!(trees, tree4)
    @test_throws ArgumentError MCPhylo.loose_consensus_tree(trees)

    tree1 = MCPhylo.parsing_newick_string("(((A,B)AB,C)ABC,(D,E)DE)no_name;")
    tree2 = MCPhylo.parsing_newick_string("((A,C)AC,(B,D,E)BDE)no_name;")
    tree3 = MCPhylo.parsing_newick_string("(((B,C)BC,A)BCA,D,E)no_name;")
    trees = [tree1, tree2, tree3]
    MCPhylo.number_nodes!.(trees)
    MCPhylo.set_binary!.(trees)
    result = newick(MCPhylo.parsing_newick_string("(A,B,C,(D,E)no_name);"))
    @test newick(MCPhylo.loose_consensus_tree(trees)) == result
end

@testset "greedy_consensus_tree" begin
    tree1 = MCPhylo.parsing_newick_string("(((A,B),C),(D,E));")
    tree2 = MCPhylo.parsing_newick_string("((A,C),(B,D,E));")
    tree3 = MCPhylo.parsing_newick_string("(((B,C),A),D,E);")
    trees = [tree1, tree2, tree3]
    MCPhylo.number_nodes!.(trees)
    MCPhylo.set_binary!.(trees)
    result = newick(MCPhylo.parsing_newick_string("(((A,B),C),(D,E));"))
    # result = newick(MCPhylo.parsing_newick_string("(((A,C),B),(D,E))"))
    # result = newick(MCPhylo.parsing_newick_string("(((B,C),A),(D,E))"))
    @test newick(MCPhylo.greedy_consensus_tree(trees)) == result
end

@testset "count_cluster_occurences" begin
    bit_vec = BitArray.([[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0],
              [0,0,0,0,1], [1,1,0,0,0], [1,1,1,0,0], [0,0,0,1,1], [1,0,0,0,0],
              [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1], [1,0,1,0,0],
              [0,1,0,1,1], [0,1,1,0,0], [1,1,1,0,0], [0,0,0,1,1], [1,0,0,0,0],
              [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]])
    sort!(bit_vec, alg=QuickSort)
    queue = MCPhylo.PriorityQueue{BitVector, Int64}()
    queue[BitArray([0,0,0,0,1])] = -3
    queue[BitArray([0,0,0,1,0])] = -3
    queue[BitArray([0,1,0,0,0])] = -3
    queue[BitArray([0,0,1,0,0])] = -3
    queue[BitArray([1,0,0,0,0])] = -3
    queue[BitArray([0,0,0,1,1])] = -2
    queue[BitArray([1,1,0,0,0])] = -1
    queue[BitArray([1,0,1,0,0])] = -1
    queue[BitArray([0,1,0,1,1])] = -1
    queue[BitArray([0,1,1,0,0])] = -1
    queue[BitArray([1,1,1,0,0])] = -2

    @test MCPhylo.count_cluster_occurences(bit_vec) == queue
end


"""
# Additional test with a big file of trees
trees = MCPhylo.ParseNewick("./doc/Tree/Drav_mytrees_1.nwk")
MCPhylo.set_binary!.(trees)
MCPhylo.number_nodes!.(trees)
majority_tree = MCPhylo.majority_consensus_tree(trees)
println(newick(majority_tree))
"""
