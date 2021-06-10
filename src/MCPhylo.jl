module MCPhylo

using Reexport
@reexport using Distributions

#################### Imports ####################

using DelimitedFiles
using SpecialFunctions
using Serialization
using Distributed
using Printf: @sprintf
using LinearAlgebra
using Plots
@reexport using Plots
using Plots.PlotMeasures
using StatsPlots
@reexport using StatsPlots
using RecipesBase
using StatsBase
using Zygote
using FiniteDiff
using ChainRules
using ChainRulesCore
using Showoff: showoff
using Markdown
using Random
using DataStructures
using ProgressMeter
using CUDA
if has_cuda()
  using GPUArrays
else
  @warn "The Julia CUDA library is installed, but no CUDA device detected.
         Computation is performed without CUDA functionality."
end

import Base: Matrix, names, summary, iterate
import Base.Threads.@spawn
import LinearAlgebra: cholesky, dot, BlasFloat
import Statistics: cor
import Distributions:
       ## Generic Types
       Continuous, ContinuousUnivariateDistribution, Distribution,
       MatrixDistribution, Multivariate, MultivariateDistribution, PDiagMat,
       PDMat, ScalMat, Truncated, Univariate, UnivariateDistribution,
       ValueSupport,
       ## ContinuousUnivariateDistribution Types
       Arcsine, Beta, BetaPrime, Biweight, Cauchy, Chi, Chisq, Cosine,
       Epanechnikov, Erlang, Exponential, FDist, Frechet, Gamma, Gumbel,
       InverseGamma, InverseGaussian, Kolmogorov, KSDist, KSOneSided, Laplace,
       Levy, Logistic, LogNormal, NoncentralBeta, NoncentralChisq,
       NoncentralF, NoncentralT, Normal, NormalCanon, Pareto, Rayleigh,
       SymTriangularDist, TDist, TriangularDist, Triweight, Uniform, VonMises,
       Weibull,
       ## DiscreteUnivariateDistribution Types
       Bernoulli, Binomial, Categorical, DiscreteUniform, Geometric,
       Hypergeometric, NegativeBinomial, NoncentralHypergeometric, Pareto,
       PoissonBinomial, Skellam,
       ## MultivariateDistribution Types
       Dirichlet, Multinomial, MvNormal, MvNormalCanon, MvTDist,
       VonMisesFisher,
       ## MatrixDistribution Types
       InverseWishart, Wishart,
       ## Methods
       cdf, dim, gradlogpdf, insupport, isprobvec, logpdf, logpdf!, maximum,
       minimum, pdf, quantile, rand, sample!, support, length
using LightGraphs: DiGraph, add_edge!, outneighbors,
       topological_sort_by_dfs, vertices
import StatsBase: autocor, autocov, countmap, counts, describe, predict,
       quantile, sample, sem, summarystats

include("distributions/pdmats2.jl")
using .PDMats2
include("Tree/Node_Type.jl") # We need this to get the Node type in

#################### Types ####################

ElementOrVector{T} = Union{T, Vector{T}}


#################### Variate Types ####################

abstract type ScalarVariate <: Real end
abstract type ArrayVariate{N} <: DenseArray{Float64, N} end
abstract type TreeVariate <: AbstractNode end

const AbstractVariate = Union{ScalarVariate, ArrayVariate, TreeVariate}
const NumericalVariate = Union{ScalarVariate, ArrayVariate}
const VectorVariate = ArrayVariate{1}
const MatrixVariate = ArrayVariate{2}



#################### Distribution Types ####################

const DistributionStruct = Union{Distribution,
                                 Array{UnivariateDistribution},
                                 Array{MultivariateDistribution}}


#################### Dependent Types ####################

mutable struct ScalarLogical <: ScalarVariate
  value::Float64
  symbol::Symbol
  monitor::Vector{Int}
  eval::Function
  sources::Vector{Symbol}
  targets::Vector{Symbol}
end

mutable struct ArrayLogical{N} <: ArrayVariate{N}
  value::Array{Float64, N}
  symbol::Symbol
  monitor::Vector{Int}
  eval::Function
  sources::Vector{Symbol}
  targets::Vector{Symbol}
end

mutable struct TreeLogical{T} <: TreeVariate where T<:GeneralNode
  value::T
  symbol::Symbol
  monitor::Vector{Int}
  eval::Function
  sources::Vector{Symbol}
  targets::Vector{Symbol}
end

mutable struct ScalarStochastic <: ScalarVariate
  value::Float64
  symbol::Symbol
  monitor::Vector{Int}
  eval::Function
  sources::Vector{Symbol}
  targets::Vector{Symbol}
  distr::UnivariateDistribution
end

mutable struct ArrayStochastic{N} <: ArrayVariate{N}
  value::Union{CuArray{Float64, N}, Array{Float64, N}}
  symbol::Symbol
  monitor::Vector{Int}
  eval::Function
  sources::Vector{Symbol}
  targets::Vector{Symbol}
  distr::DistributionStruct
end


mutable struct TreeStochastic{T} <: TreeVariate where T<: GeneralNode
    value::T
    symbol::Symbol
    monitor::Vector{Int}
    eval::Function
    sources::Vector{Symbol}
    targets::Vector{Symbol}
    distr::DistributionStruct
end

const AbstractLogical = Union{ScalarLogical, ArrayLogical}
const AbstractStochastic = Union{ScalarStochastic, ArrayStochastic}
const AbstractTreeStochastic = Union{TreeLogical, TreeStochastic}
const AbstractDependent = Union{AbstractLogical, AbstractStochastic, AbstractTreeStochastic}


#################### Sampler Types ####################

mutable struct Sampler{T}
  params::Vector{Symbol}
  eval::Function
  tune::T
  targets::Vector{Symbol}
end


abstract type SamplerTune end

struct SamplerVariate{T<:SamplerTune} <: VectorVariate
  value::Union{Vector{Float64}, Vector{S}} where S<:GeneralNode
  tune::T

  function SamplerVariate{T}(x::AbstractVector, tune::T) where T<:SamplerTune
    v = new{T}(x, tune)
    validate(v)
  end

  function SamplerVariate{T}(x::AbstractVector, pargs...; kargs...) where T<:SamplerTune
    if !isa(x[1], GeneralNode)
      value = convert(Vector{Float64}, x)
    else
      mt = typeof(x[1])
      value = convert(Vector{mt}, x)
    end
    new{T}(value, T(value, pargs...; kargs...))
  end
end


#################### Model Types ####################

struct ModelGraph
  graph::DiGraph
  keys::Vector{Symbol}
end

struct ModelState
  value::Vector
  tune::Vector{Any}
end

mutable struct Model
  nodes::Dict{Symbol, Any}
  samplers::Vector{Sampler}
  states::Vector{ModelState}
  iter::Int
  burnin::Int
  hasinputs::Bool
  hasinits::Bool
  likelihood::Float64
end


############## Additional Structs ################

struct SimulationParameters
  burnin::Int64
  thin::Int64
  chains::Int64
  verbose::Bool
  trees::Bool
  asdsf::Bool
  freq::Int64
  min_splits::Float64
end

struct ConvergenceStorage
  splitsQueue::Vector{Accumulator{Tuple{String, String}, Int64}}
  splitsQueues::Vector{Vector{Accumulator{Tuple{String, String}, Int64}}}
  run::Int64
end

#################### Chains Type ####################

abstract type AbstractChains end

struct Chains <: AbstractChains
  value::Array{Float64, 3}
  range::StepRange{Int, Int}
  names::Vector{AbstractString}
  chains::Vector{Int}
  trees::Array{AbstractString, 3}
  moves::Array{Int, 1}
  tree_names::Vector{AbstractString}
end

struct ModelChains <: AbstractChains
  value::Array{Float64, 3}
  range::StepRange{Int, Int}
  names::Vector{AbstractString}
  chains::Vector{Int}
  model::Model
  trees::Array{AbstractString, 3}
  moves::Array{Int, 1}
  tree_names::Vector{AbstractString}
  stats::Array{Float64, 2}
  stat_names::Vector{AbstractString}
  sim_params::SimulationParameters
  conv_storage::Union{Nothing, ConvergenceStorage}
end

#################### Includes ####################

include("progress.jl")
include("utils.jl")
include("variate.jl")

include("distributions/constructors.jl")
include("distributions/distributionstruct.jl")
include("distributions/extensions.jl")
include("distributions/pdmatdistribution.jl")
include("Likelihood/SubstitutionModels.jl")
include("distributions/Phylodist.jl")
include("distributions/TreeConstraints.jl")
include("distributions/transformdistribution.jl")

include("model/dependent.jl")
include("model/dependent_tree.jl")
include("model/graph.jl")
include("model/initialization.jl")
include("model/mcmc.jl")
include("model/model.jl")
include("model/simulation.jl")

include("output/chains.jl")
include("output/chainsummary.jl")
include("output/discretediag.jl")
include("output/fileio.jl")
include("output/gelmandiag.jl")
include("output/gewekediag.jl")
include("output/heideldiag.jl")
include("output/mcse.jl")
include("output/modelchains.jl")
include("output/modelstats.jl")
include("output/rafterydiag.jl")
include("output/stats.jl")
include("output/plot.jl")
include("output/tree_plot.jl")

include("samplers/sampler.jl")

include("samplers/abc.jl")
include("samplers/amm.jl")
include("samplers/amwg.jl")
include("samplers/bhmc.jl")
include("samplers/bia.jl")
include("samplers/bmc3.jl")
include("samplers/bmg.jl")
include("samplers/dgs.jl")
include("samplers/dmh.jl")
include("samplers/empirical.jl")
include("samplers/hmc.jl")
include("samplers/mala.jl")
include("samplers/miss.jl")
include("samplers/nuts.jl")
include("samplers/pnuts.jl")
include("samplers/rwm.jl")
include("samplers/rwmc.jl")
include("samplers/slice.jl")
include("samplers/slicesimplex.jl")


include("Tree/Tree_Basics.jl")
include("Tree/Converter.jl")
include("Tree/Tree_moves.jl")
include("Tree/Tree_Distance.jl")
include("Tree/Tree_Traversal.jl")
include("Tree/Tree_Search.jl")
include("Tree/Tree_Legacy.jl")
include("Tree/Tree_Clustering.jl")
include("Tree/Tree_Ladderizing.jl")
include("Tree/Tree_Pruning.jl")
include("Tree/Tree_Consensus.jl")
include("Tree/Tree_Statistics.jl")


include("Parser/Parser.jl")
include("Parser/ParseCSV.jl")
include("Parser/ParseNexus.jl")
include("Parser/ParseNewick.jl")


include("Utils/FileIO.jl")

include("Likelihood/LikelihoodCalculator_Node.jl")
include("Likelihood/Parsimony.jl")
include("Likelihood/Prior.jl")
include("Likelihood/Rates.jl")
#################### Exports ####################

export
  AbstractChains,
  AbstractDependent,
  AbstractLogical,
  AbstractStochastic,
  AbstractVariate,
  ArrayLogical,
  ArrayStochastic,
  ArrayVariate,
  ConvergenceStorage,
  TreeLogical,
  TreeVariate,
  TreeStochastic,
  GeneralNode,
  AbstractNode,
  Node,
  Chains,
  Logical,
  MatrixVariate,
  Model,
  ModelChains,
  Sampler,
  SamplerTune,
  SamplerVariate,
  ScalarLogical,
  ScalarStochastic,
  ScalarVariate,
  SimulationParameters,
  Stochastic,
  VectorVariate

export
  BDiagNormal,
  Flat,
  SymUniform,
  CompoundDirichlet,
  PhyloDist,
  MultiplePhyloDist,
  exponentialBL

export
  autocor,
  changerate,
  contourplot,
  cor,
  describe,
  dic,
  discretediag,
  draw,
  gelmandiag,
  gettune,
  gewekediag,
  gradlogpdf,
  gradlogpdf!,
  graph,
  graph2dot,
  heideldiag,
  hpd,
  insupport,
  invlink,
  invlogit,
  link,
  logit,
  logpdf,
  logpdf!,
  mcmc,
  mcse,
  plot,
  plot_asdsf,
  predict,
  quantile,
  rafterydiag,
  rand,
  readcoda,
  relist,
  relist!,
  sample!,
  setinits!,
  setinputs!,
  setmonitor!,
  setsamplers!,
  summarystats,
  unlist,
  update!

@static if VERSION >= v"1.0.0"
  export showall
end

export
  ABC,
  AMM, AMMVariate,
  AMWG, AMWGVariate,
  BHMC, BHMCVariate,
  BMC3, BMC3Variate,
  BMG, BMGVariate,
  DiscreteVariate,
  DGS, DGSVariate,
  HMC, HMCVariate,
  BIA, BIAVariate,
  MALA, MALAVariate,
  MISS,
  NUTS, NUTSVariate,
  RWM, RWMVariate,
  Slice, SliceMultivariate, SliceUnivariate,
  SliceSimplex, SliceSimplexVariate,
  DMH, DMHVariate,
  PNUTS, PNUTSVariate,
  Empirical, EmpiricalVariate

export
  make_tree_with_data,
  make_tree_with_data_cu,
  to_file, drop_samples,
  to_df,
  to_covariance,
  to_distance_matrix,
  node_distance,
  get_path,
  cut_samples,
  NNI!, NNI,
  SPR!, SPR,
  slide!, slide,
  swing!, swing,
  RF, randomize!,
  BHV_bounds,
  get_branchlength_vector,
  set_branchlength_vector!,
  post_order,
  pre_order,
  level_order,
  add_child!,
  remove_child!,
  delete_node!,
  insert_node!,
  find_lca,
  find_by_binary,
  find_by_name,
  find_root,
  create_tree_from_leaves,
  create_tree_from_leaves_cu,
  newick,
  tree_length,
  tree_height,
  path_length,
  get_sister,
  get_leaves,
  check_leafsets,
  neighbor_joining,
  upgma,
  prune_tree!, prune_tree,
  ladderize_tree!, ladderize_tree,
  majority_consensus_tree,
  discrete_gamma_rates,
  Restriction, JC, GTR, freeK,
  ASDSF, parsimony, ParseNewick


export
  cm,
  inch,
  mm,
  pt,
  px

#################### Deprecated ####################

include("deprecated.jl")

end # module
