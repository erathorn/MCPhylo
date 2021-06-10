#################### Model-Based Posterior Statistics ####################
"""
    dic(mc::ModelChains)

Compute the Deviance Information Criterion (DIC) of Spiegelhalter et al. and Gelman et al. from MCMC sampler output.

Returns a `ChainSummary` type object with DIC results from the methods of Spiegelhalter and Gelman in the first and second rows of the `value` field, and the DIC value and effective numbers of parameters in the first and second columns; where

``\\text{DIC} = -2 \\mathcal{L}(\\bar{\\Theta}) + 2 p,``

such that ``\\mathcal{L}(\\bar{\\Theta})`` is the log-likelihood of model outputs given the expected values of model parameters ``\\Theta``, and ``p`` is the effective number of parameters. The latter is defined as ``p_D = -2 \\bar{\\mathcal{L}}(\\Theta) + 2 \\mathcal{L}(\\bar{\\Theta})`` for the method of Spiegelhalter and as ``p_V = \\frac{1}{2} \\operatorname{var}(-2 \\mathcal{L}(\\Theta))`` for the method of Gelman. Results are for all chains combined.


* `mc` : sampler output from a model fit with the `mcmc()` function.
"""
function dic(mc::ModelChains)
  nodekeys = keys(mc.model, :output)

  Dhat = -2.0 * logpdf(mc, mean, nodekeys)
  D = -2.0 * logpdf(mc, nodekeys).value
  p = [mean(D) - Dhat, 0.5 * var(D)]

  ChainSummary([Dhat .+ 2.0 .* p  p], ["pD", "pV"],
               ["DIC", "Effective Parameters"], header(mc))
end

"""
    logpdf(mc::ModelChains, f::Function, nodekeys::Vector{Symbol})

Compute the sum of log-densities at each iteration of MCMC output for stochastic nodes.

Returns a `ModelChains` object of resulting summed log-densities at each MCMC iteration of the supplied chain.

* `mc` : sampler output from a model fit with the `mcmc()`` function.

*  `nodekey/nodekeys` : stochastic model node(s) over which to sum densities (default: all).

* `f` : ??
"""
function logpdf(mc::ModelChains, f::Function, nodekeys::Vector{Symbol})
  m = mc.model

  relistkeys, updatekeys = getsimkeys(mc, nodekeys)
  relistkeys = union(relistkeys, intersect(nodekeys, keys(m, :block)))
  inds = names2inds(mc, relistkeys)
  for key in relistkeys
    isa(m[key], TreeStochastic) && throw("not possible with tree objects")
  end

  m[relistkeys] = relist(m, map(i -> f(mc.value[:, i, :]), inds), relistkeys)
  update!(m, updatekeys)
  mapreduce(key -> logpdf(m[key]), +, nodekeys)
end


logpdf(mc::ModelChains, nodekey::Symbol) = logpdf(mc, [nodekey])
"""
    logpdf(mc::ModelChains,
            nodekeys::Vector{Symbol}=keys(mc.model, :stochastic))

"""
function logpdf(mc::ModelChains,
                nodekeys::Vector{Symbol}=keys(mc.model, :stochastic))
  N = length(mc.range)
  K = size(mc, 3)

  relistkeys, updatekeys = getsimkeys(mc, nodekeys)
  relistkeys = union(relistkeys, intersect(nodekeys, keys(mc.model, :block)))
  inds = names2inds(mc, relistkeys)

  frame = ChainProgressFrame(
    "MCMC Processing of $N Iterations x $K Chain" * "s"^(K > 1), true
  )

  lsts = [
    Any[mc[:, :, [k]], nodekeys, relistkeys, inds, updatekeys,
        ChainProgress(frame, k, N)]
    for k in 1:K
  ]
  sims = pmap2(logpdf_modelchains_worker, lsts)

  ModelChains(cat(sims..., dims=3), mc.model, mc.stats, mc.stat_names)
end


function logpdf_modelchains_worker(args::Vector)
  mc, nodekeys, relistkeys, inds, updatekeys, meter = args
  m = mc.model

  sim = Chains(size(mc, 1), 1, start=first(mc), thin=step(mc), names=["logpdf"])

  for i in 1:size(mc.value, 1)
    m[relistkeys] = relist(m, mc.value[i, inds, 1], relistkeys)
    update!(m, updatekeys)
    sim.value[i, 1, 1] = mapreduce(key -> logpdf(m[key]), +, nodekeys)
    next!(meter)
  end

  sim
end


predict(mc::ModelChains, nodekey::Symbol) = predict(mc, [nodekey])
"""
    predict(mc::ModelChains,
             nodekeys::Vector{Symbol}=keys(mc.model, :output))

Generate MCMC draws from a posterior predictive distribution.

Returns a `ModelChains` object of draws simulated at each MCMC iteration of the supplied chain. For observed data node ``y``, simulation is from the posterior predictive distribution

``p(\\tilde{y} | y) = \\int p(\\tilde{y} | \\Theta) p(\\Theta | y) d\\Theta,``

where ``\\tilde{y}`` is an unknown observation on the node, ``p(\\tilde{y} | \\Theta)`` is the data likelihood, and ``p(\\Theta | y)`` is the posterior distribution of unobserved parameters ``\\Theta``.

* `mc` : sampler output from a model fit with the `mcmc()`` function.

* `nodekey/nodekeys` : observed Stochastic model node(s) for which to generate draws from the predictive distribution (default: all observed data nodes).
"""
function predict(mc::ModelChains,
                 nodekeys::Vector{Symbol}=keys(mc.model, :output))
  m = mc.model

  outputs = keys(m, :output)
  all(key -> key in outputs, nodekeys) ||
    throw(ArgumentError(string(
      "nodekeys are not all observed Stochastic nodess : ",
      join(map(string, outputs), ", ")
    )))

  nodenames = names(m, nodekeys)
  relistkeys, updatekeys = getsimkeys(mc, nodekeys)
  for key in relistkeys
    isa(m[key], TreeStochastic) && throw("not possible with tree objects")
  end

  inds = names2inds(mc, relistkeys)

  c = Chains(size(mc, 1), length(nodenames), chains=size(mc, 3),
             start=first(mc), thin=step(mc), names=nodenames)

  iters, _, chains = size(c.value)
  for k in 1:chains
    for i in 1:iters
      m[relistkeys] = relist(m, mc.value[i, inds, k], relistkeys)
      update!(m, updatekeys)
      f = key -> unlist(m[key], rand(m[key]))
      c.value[i, :, k] = vcat(map(f, nodekeys)...)
    end
  end

  ModelChains(c, m, c.stats, c.stat_names)
end


#################### Auxiliary Functions ####################

function getsimkeys(mc::ModelChains, nodekeys::Vector{Symbol})
  relistkeys = Symbol[]
  updatekeys = Symbol[]

  m = mc.model
  dag = ModelGraph(m)

  nodekeys = intersect(nodekeys, keys(m, :stochastic))
  blockkeys = keys(m, :block)
  dynamickeys = union(blockkeys, keys(m, :target, blockkeys))
  terminalkeys = union(keys(m, :stochastic), keys(mc, :dependent))

  for v in vertices(dag.graph)
    vkey = dag.keys[v]
    if vkey in dynamickeys
      if any(key -> key in nodekeys, gettargets(dag, v, terminalkeys))
        vkey in terminalkeys ?
          push!(relistkeys, vkey) :
          push!(updatekeys, vkey)
      end
    end
  end
  if !isempty(relistkeys) append!(updatekeys, nodekeys) end

  relistkeys, intersect(keys(m, :dependent), updatekeys)
end
