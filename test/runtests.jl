using Distributed
@everywhere using Random, Test
include("utils.jl")

const tutorialtests = [
   "line"
]

const samplertests = [
  "abc",
  "amm",
  "amwg",
  "bhmc",
  "bia",
  "bmc3",
  "bmg",
  "hmc",
  "mala",
  "nuts",
  "rwm",
  "slice",
  "slicesimplex"
]

const mcmctests = [
  "discretediag",
  "readcoda"
]

const extensiontests = [
  "newunivardist",
  "newmultivardist"
]

const treetests = [
  "basics",
  "consensus",
  "ladderize",
  "pruning",
  "spr"
]

const parsertests = [
  "newick"
]


println("Running tests:")
#
@testset "All tests" begin
@testset "Tutorial" begin
  @everywhere Random.seed!(123)
  for t in tutorialtests
    @runtest "mcmc/" t
end
end

@testset "Samplertest" begin
  @everywhere Random.seed!(123)
for t in samplertests
    @runtest "samplers/" t
  end
end
@testset "mcmctests" begin
  for t in mcmctests
    @runtest "mcmc/" t
  end
end
#
@testset "extensions" begin
for t in extensiontests
  @everywhere Random.seed!(123)
    @runtest "mcmc/" t
  end
end

@testset "treetests" begin
for t in treetests
  @everywhere Random.seed!(123)
    @runtest "Tree/" t
  end
end

@testset "parsers" begin
for t in parsertests
  @everywhere Random.seed!(123)
    @runtest "parsers/" t
  end
end

end
#all sets test set
