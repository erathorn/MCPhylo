using MCPhylo
using Test

@testset "NewickParser basic" begin
    filename = "newicktester.nwk"
    filepath = joinpath(@__DIR__,filename)
    testlist = MCPhylo.load_newick(filepath)
    @test testlist == ["(A,B)C;","((A,B)C,(D,E)F)G;"]
    testlist = ["(A,B)C;","((A,B)C,(D,E)F)G;"]
    for x in testlist
        @test MCPhylo.is_valid_newick_string(x) == true
    end
    @test MCPhylo.is_valid_newick_string("();") == true

    @test MCPhylo.is_valid_newick_string("(A,B));") == false
    @test MCPhylo.is_valid_newick_string("invalid") == false

    tuz = MCPhylo.ParseNewick(filepath)
    tuz1 = Node()
    tuz1.name = "C"
    A = Node()
    A.name = "A"
    B = Node()
    B.name = "B"
    MCPhylo.add_child!(tuz1,A)
    MCPhylo.add_child!(tuz1,B)
    @test newick(tuz[1]) == newick(tuz1)
    @test tuz[1] != MCPhylo.newick(B)
    nodes = MCPhylo.parsing_newick_string(testlist[1])
    @test nodes.children[1].name == "A"
    nodes2 = MCPhylo.parsing_newick_string(testlist[2])
    @test nodes2.children[2].name == "F"
    nodes3 = MCPhylo.parsing_newick_string("(A:0.1, B:0.2);")
    @test nodes3.children[1].name == "A"

end
@testset "NewickParser Special" begin
    node = MCPhylo.parse_name_length("A:0.5")
    @test node.name == "A"
    @test node.inc_length == 0.5
    
    node = MCPhylo.parse_name_length("A")
    @test node.name == "A"
    @test node.inc_length == 1.0
    
    node = MCPhylo.parse_name_length("")
    @test node.name == "no_name"
    @test node.inc_length == 1.0
    
    node = MCPhylo.parse_name_length("A:0.5;")
    @test node.name == "A"
    @test node.inc_length == 0.5
    
    node = MCPhylo.parse_name_length(" A:0.5")
    @test node.name == "A"
    @test node.inc_length == 0.5
    
    node = MCPhylo.parse_name_length("A:0.5 ")
    @test node.name == "A"
    @test node.inc_length == 0.5
    
end
