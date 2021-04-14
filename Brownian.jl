using Revise
using Pkg
Pkg.activate(".")
Pkg.instantiate()
using MCPhylo
using LinearAlgebra
import Distributions: DiscreteMatrixDistribution, logpdf
using CSV,DataFramesMeta, DataFrames
includet("BrownianDistr.jl")
includet("helper.jl")

#mt, df = make_tree_with_data("notebook/Dravidian.cc.phy.nex",binary=true); # load your own nexus file

#lnum = [n.num for n in get_leaves(mt)]

#dfn = df[:, :, lnum]
#dfn1 = zeros(size(dfn,3), size(dfn,2))

#dfn1 .= transpose(dfn[1, :, :])
data, langs = read_cldf("untracked_files/data-ie-42-208.tsv")

mt = MCPhylo.create_tree_from_leaves_bin(langs,1)

lnum = [n.num for n in get_leaves(mt)]
lnames = [n.name for n in get_leaves(mt)]
lnames == langs
x = zeros(Int,length(lnames))
for (ind,i) in enumerate(langs)
   x[ind] = findfirst(y -> y == i, lnames)
end

#dfn = zeros(size(df)[1], size(df)[2], length(lnum))
mt2 = deepcopy(mt)
randomize!(mt2)
data =data[x,:]
my_data = Dict{Symbol, Any}(
  :mtree => mt,
  :df => data,
  :nnodes => size(data)[1],
  :nsites => size(data)[2],
);


nclasses_vec = length.(unique.(eachcol(data))) .- 1

# model setup
model =  Model(
    df = Stochastic(2, (Σ, mu, sig, latent) ->
                        BrownianCatDistr(mu, sig, Σ, latent, nclasses_vec), false, false),

    Σ = Logical(2,(mtree)->to_covariance(mtree), false),
    mu = Stochastic(2, ()->Normal(),false),
    sig = Stochastic(1, ()->Exponential(),false),
    latent = Stochastic(1,()->Normal(),false),
    mtree = Stochastic(Node(), () -> CompoundDirichlet(1.0,1.0,0.100,1.0), true)
   )

inits = [ Dict{Symbol, Union{Any, Real}}(
   :mtree => mt,
   :df => my_data[:df],
   :nnodes => my_data[:nnodes],
   :mu => zeros(my_data[:nnodes], my_data[:nsites]),
   :sig => rand(my_data[:nsites]),
   :latent => randn(30514)
   ),
   Dict{Symbol, Union{Any, Real}}(
   :mtree => mt2,
   :df => my_data[:df],
   :nnodes => my_data[:nnodes],
   :mu => zeros(my_data[:nnodes], my_data[:nsites]),
   :sig => rand(my_data[:nsites]),
   :latent => randn(30514)
   )
    ]

scheme = [RWM(:mtree, [:NNI]),
          Slice(:mtree, [1.0]),
          Slice(:sig, 1.0),
          Slice(:latent, 1.0)
         ]

setsamplers!(model, scheme);

sim = mcmc(model, my_data, inits, 100, burnin=50,thin=5, chains=2, trees=true)
sim = mcmc(sim, 500, trees=true)
