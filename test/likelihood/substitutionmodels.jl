@testset "SubstitutionModels.jl" begin
    @testset "Restriction" begin
        target = ([-0.9191450300180579 -0.7071067811865475; 0.3939192985791677 -0.7071067811865476], 
                  [-1.0, 0.0], [-0.7615773105863908 0.7615773105863908; -0.4242640687119285 -0.9899494936611665], 
                  2.380952380952381)
        result = Restriction([0.3, 0.7], [0.0])
        @test result[1] ≈ target[1]
        @test result[2] ≈ target[2]
        @test_throws AssertionError MCPhylo.Restriction(ones(3) / 3, [0.1])
    end  
    
    @testset "JC" begin
        target = ([0.7071067811865476 -0.408248290463863 -0.5773502691896257; -0.7071067811865475 -0.40824829046386313 -0.577350269189626; 0.0 0.816496580927726 -0.5773502691896258],
                  [-1.0, -0.9999999999999994, 1.1102230288049144e-16], 
                  [0.7071067811865476 -0.7071067811865475 0.0; -0.4082482904638629 -0.408248290463863 0.816496580927726; -0.5773502691896256 -0.5773502691896257 -0.5773502691896258], 1.5)
        result = JC([0.15, 0.45, 0.4],[0.0])
        @test result[1] ≈ target[1]
        @test result[2] ≈ target[2] 
    end
end