include("../../src/MCPhylo.jl")
using .MCPhylo
using Test

@testset "find_common_clusters" begin

    ref_tree = MCPhylo.parsing_newick_string("((A,(B,(C,(D,E)))),(F,(G,H)))")
    MCPhylo.number_nodes!(ref_tree)
    MCPhylo.set_binary!(ref_tree)

    tree2 = MCPhylo.parsing_newick_string("((G,(C,(A,(F,E)))),(B,(D,H)))")
    MCPhylo.number_nodes!(tree2)
    MCPhylo.set_binary!(tree2)

    A, B, C, D, E, F, G, H = find_by_name(tree2, "A"), find_by_name(tree2, "B"),
                             find_by_name(tree2, "C"), find_by_name(tree2, "D"),
                             find_by_name(tree2, "E"), find_by_name(tree2, "F"),
                             find_by_name(tree2, "G"), find_by_name(tree2, "H")

    expected_matches = [G.mother.mother]
    expected_mis_matches = [F.mother, A.mother, C.mother, G.mother, D.mother, B.mother]
    matches, mis_matches = MCPhylo.find_common_clusters(ref_tree, tree2)
    @test matches == expected_matches
    @test mis_matches == expected_mis_matches

    tree3 = MCPhylo.parsing_newick_string("((A,(C,(D,(B,E)))),(G,(F,H)))")
    MCPhylo.number_nodes!(tree3)
    MCPhylo.set_binary!(tree3)

    A, B, C, D, E, F, G, H = find_by_name(tree3, "A"), find_by_name(tree3, "B"),
                             find_by_name(tree3, "C"), find_by_name(tree3, "D"),
                             find_by_name(tree3, "E"), find_by_name(tree3, "F"),
                             find_by_name(tree3, "G"), find_by_name(tree3, "H")

    expected_matches = [C.mother, A.mother, G.mother, A.mother.mother]
    expected_mis_matches = [B.mother, D.mother, F.mother]
    matches, mis_matches = MCPhylo.find_common_clusters(ref_tree, tree3)
    @test matches == expected_matches
    @test mis_matches == expected_mis_matches

    tree4 = MCPhylo.parsing_newick_string("((A,(B,(C,(D,E)))),(F,(G,H)))")
    MCPhylo.number_nodes!(tree4)
    MCPhylo.set_binary!(tree4)

    A, B, C, D, E, F, G, H = find_by_name(tree4, "A"), find_by_name(tree4, "B"),
                             find_by_name(tree4, "C"), find_by_name(tree4, "D"),
                             find_by_name(tree4, "E"), find_by_name(tree4, "F"),
                             find_by_name(tree4, "G"), find_by_name(tree4, "H")

    expected_matches = [D.mother, C.mother, B.mother, A.mother, G.mother, F.mother,
                        A.mother.mother]
    expected_mis_matches = Vector{Node}()
    matches, mis_matches = MCPhylo.find_common_clusters(ref_tree, tree4)
    @test matches == expected_matches
    @test mis_matches == expected_mis_matches

    tree5 = MCPhylo.parsing_newick_string("((G,(X,(A,(F,E)))),(B,(D,H)))")
    MCPhylo.number_nodes!(tree5)
    MCPhylo.set_binary!(tree5)
    @test_throws ArgumentError MCPhylo.find_common_clusters(ref_tree, tree5)

    tree6 = MCPhylo.parsing_newick_string("(X,(G,(C,(A,(F,E)))),(B,(D,H)))))")
    MCPhylo.number_nodes!(tree5)
    MCPhylo.set_binary!(tree5)
    @test_throws ArgumentError MCPhylo.find_common_clusters(ref_tree, tree6)
end


@testset "order_tree!" begin
    tree = MCPhylo.parsing_newick_string("(A,B,(C,(D,E)F)G)H;")
    MCPhylo.set_binary!(tree)
    MCPhylo.number_nodes!(tree)
    A, B, C, D, E, F, G, H = find_by_name(tree, "A"), find_by_name(tree, "B"),
                             find_by_name(tree, "C"), find_by_name(tree, "D"),
                             find_by_name(tree, "E"), find_by_name(tree, "F"),
                             find_by_name(tree, "G"), find_by_name(tree, "H")

    cluster_start_indeces = Dict([(A, 3), (B, 7), (C, 2), (D, 8),
                                  (E, 5), (F, 1), (G, 4), (H, 6)])

    ordered_tree = MCPhylo.parsing_newick_string("(A,((E,D)F,C)G,B)H;")
    MCPhylo.number_nodes!(ordered_tree)
    MCPhylo.set_binary!(ordered_tree)

    @test MCPhylo.order_tree!(tree, cluster_start_indeces) == [A, E, D, C, B]
    MCPhylo.set_binary!(tree)
    MCPhylo.number_nodes!(tree)
    @test tree == ordered_tree
end

@testset "max/min_leaf_rank" begin

    tree = MCPhylo.parsing_newick_string("(A,B,(C,(D,E)F)G)H;")
    F, G, H, A = find_by_name(tree, "F"), find_by_name(tree, "G"),
                 find_by_name(tree, "H"), find_by_name(tree, "A")
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
    A, B, C, D, E, F, G, H = find_by_name(tree, "A"), find_by_name(tree, "B"),
                             find_by_name(tree, "C"), find_by_name(tree, "D"),
                             find_by_name(tree, "E"), find_by_name(tree, "F"),
                             find_by_name(tree, "G"), find_by_name(tree, "H")

    @testset "x_left" begin
        @test MCPhylo.x_left(A) == H
        @test MCPhylo.x_left(B) == B
        @test MCPhylo.x_left(C) == G
        @test MCPhylo.x_left(D) == F
        @test MCPhylo.x_left(E) == E
    end

    @testset "x_right" begin
        @test MCPhylo.x_right(A) == A
        @test MCPhylo.x_right(B) == B
        @test MCPhylo.x_right(C) == C
        @test MCPhylo.x_right(D) == D
        @test MCPhylo.x_right(E) == H
    end

end

@testset "are_compatible" begin

    tree = MCPhylo.parsing_newick_string("(A,B,(C,(D,E)F)G)H;")
    MCPhylo.number_nodes!(tree)
    MCPhylo.set_binary!(tree)

    @testset "are_compatible with nodes" begin
        cluster = [find_by_name(tree, "A")]
        @test MCPhylo.are_compatible(cluster, tree)
        cluster = [find_by_name(tree, "A"), find_by_name(tree, "B")]
        @test MCPhylo.are_compatible(cluster, tree)
        cluster = [find_by_name(tree, "A"), find_by_name(tree, "C")]
        @test !MCPhylo.are_compatible(cluster, tree)
        cluster = [find_by_name(tree, "D"), find_by_name(tree, "E")]
        @test MCPhylo.are_compatible(cluster, tree)
        cluster = [find_by_name(tree, "C"), find_by_name(tree, "F")]
        @test MCPhylo.are_compatible(cluster, tree)
        cluster = [find_by_name(tree, "A"), find_by_name(tree, "B"),
                   find_by_name(tree, "C"), find_by_name(tree, "D"),
                   find_by_name(tree, "E")]
        @test MCPhylo.are_compatible(cluster, tree)
        cluster = [find_by_name(tree, "C"), find_by_name(tree, "D"),
                   find_by_name(tree, "E")]
        @test MCPhylo.are_compatible(cluster, tree)
    end

    @testset "are_compatible with node names" begin
        cluster = ["C"]
        @test MCPhylo.are_compatible(cluster,tree)
        cluster = ["A", "B"]
        @test MCPhylo.are_compatible(cluster,tree)
        cluster = ["A", "C"]
        @test !MCPhylo.are_compatible(cluster,tree)
        cluster = ["D", "E"]
        @test MCPhylo.are_compatible(cluster,tree)
        cluster = ["C", "F"]
        @test MCPhylo.are_compatible(cluster, tree)
        cluster = ["A", "B", "C", "D", "E"]
        @test MCPhylo.are_compatible(cluster,tree)
        cluster = ["C", "D", "E"]
        @test MCPhylo.are_compatible(cluster,tree)
    end
end

@testset "merge_trees" begin


end
@testset "majority_consensus_tree" begin

    tree1 = MCPhylo.parsing_newick_string("((A,(B,(C,(D,E)))),(F,(G,H)))")
    MCPhylo.number_nodes!(tree1)
    MCPhylo.set_binary!(tree1)
    tree2 = MCPhylo.parsing_newick_string("((G,(C,(A,(F,E)))),(B,(D,H)))")
    MCPhylo.number_nodes!(tree2)
    MCPhylo.set_binary!(tree2)
    tree3 = MCPhylo.parsing_newick_string("((B,(F,(C,(D,G)))),(H,(A,E)))")
    MCPhylo.number_nodes!(tree3)
    MCPhylo.set_binary!(tree3)
    tree4 = MCPhylo.parsing_newick_string("((A,(B,(C,(D,T)))),(F,(G,H)))")
    MCPhylo.number_nodes!(tree4)
    MCPhylo.set_binary!(tree4)

    trees = [tree1, tree2, tree3]
    # TODO: needs to be updated, when majority_consensus_tree method is finished
    @test MCPhylo.majority_consensus_tree(trees) == tree1

    trees = [tree1, tree2, tree3, tree4]
    @test_throws ArgumentError MCPhylo.majority_consensus_tree(trees)
end
