using Revise
#using Distributed#, CUDA
#gpus = length(devices())
#addprocs(2)
#@everywhere using Pkg
#@everywhere Pkg.activate(".")
#@everywhere Pkg.instantiate()
using Pkg
Pkg.activate(".")
using MCPhylo
using CUDA
using LinearAlgebra
import Distributions: DiscreteMatrixDistribution, logpdf, gradlogpdf
using CSV,DataFramesMeta, DataFrames
using Zygote
using StatsFuns
using ChainRules
using Base.Threads
includet("BrownianDistr.jl")
includet("helper.jl")
includet("browniantree.jl")
CUDA.allowscalar(false)


data, langs = read_binary("untracked_files/data-ie-42-208.paps.nex");

mt = MCPhylo.create_tree_from_leaves_bin(langs,1);

lnum = [n.num for n in get_leaves(mt)];
lnames = [n.name for n in get_leaves(mt)];
blv_length = length(get_branchlength_vector(mt))

#dfn = zeros(size(df)[1], size(df)[2], length(lnum))
mt2 = deepcopy(mt);
randomize!(mt2);

mu = zeros(size(data))
sig = rand(size(data,2))
latent = rand(blv_length,size(data,2))

my_data = Dict{Symbol, Any}(
  :mtree => mt,
  :df => data,
  :nnodes => size(data)[1],
  :nsites => size(data)[2],
);




# model setup
model =  Model(
    df = Stochastic(2, (mtree, mu, sig, latent) ->
                        BrownianTreeDistr(mu, sig, mtree, latent), false, true),
    mu = Stochastic(2, ()-> Normal(), false, true),
    sig = Stochastic(1, ()-> LogNormal(0,0.1), true, true),
    latent = Stochastic(2, ()-> Normal(), false, true),
    mtree = Stochastic(Node(), () -> CompoundDirichlet(1.0,1.0,0.100,1.0), true)
   );

inits = [ Dict{Symbol, Union{Any, Real}}(
   :mtree => mt,
   :df => my_data[:df],
   :nnodes => my_data[:nnodes],
   :mu => zeros(size(my_data[:df])),
   :sig => rand(my_data[:nsites]),
   :latent => rand(blv_length,size(my_data[:df],2)),
   ),
   Dict{Symbol, Union{Any, Real}}(
   :mtree => mt2,
   :df => my_data[:df],
   :nnodes => my_data[:nnodes],
   :mu => zeros(size(my_data[:df])),
   :sig => rand(my_data[:nsites]),
   :latent => rand(blv_length, size(my_data[:df],2)),
   )
    ];

scheme = [#RWM(:mtree, [:NNI, :SPR]),
          #Slice(:mtree, 0.5),
          #PNUTS(:mtree, target=0.8, targetNNI=8),
          NUTS([:sig, :latent], dtype=:Zygote)
          #Slice([:sig, :latent], 0.5)
         ];

setsamplers!(model, scheme);

sim = mcmc(model, my_data, inits, 5, burnin=0,thin=1, chains=1, trees=true)

#psrf = max_psrf(sim)
#@show psrf
#to_file(sim, "Brownian_IE_")
#while psrf > 1.1
#    global sim, psrf#, bi, gd, gd_values, indices
#    @show psrf
#    sim = mcmc(sim, 500000)
#    psrf = max_psrf(sim)
#    to_file(sim, "Brownian_IE_")
#end

