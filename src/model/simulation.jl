#################### Model Simulation ####################
"""
    gettune(m::Model, block::Integer)
"""
function gettune(m::Model, block::Integer)
  block == 0 && return gettune(m)
  m.samplers[block].tune
end
"""
    gettune(m::Model)

Get block-sampler tuning parameters.

Returns a `Vector{Any}` of all block-specific tuning parameters without `block` input, and turning parameters for the specified `block` otherwise.
"""
function gettune(m::Model)
  Any[gettune(m, i) for i in 1:length(m.samplers)]
end
"""
    settune!(m::Model, tune, block::Integer)
"""
function settune!(m::Model, tune, block::Integer)
  block == 0 && return settune!(m, tune)
  m.samplers[block].tune = tune
end
"""
    settune!(m::Model, tune::Vector{Any})

Set tuning parameters for one or all blocks.

Assigns desired tune values to model.

* `m` : model containing the nodes of interest.

* `tune` : tune values to be assigned to models; if no `block` value is input, `tune` must be a Vector with length equal to `m.samplers`.

* `block` : Integer denoting which block's tune value is to be reassigned.
"""
function settune!(m::Model, tune::Vector{Any})
  nsamplers = length(m.samplers)
  ntune = length(tune)
  nsamplers == ntune ||
    throw(DimensionMismatch(
      "tried to assign $ntune tune elements to $nsamplers samplers"
    ))

  for i in 1:nsamplers
    settune!(m, tune[i], i)
  end
end

"""
    gradlogpdf(m::Model, block::Integer=0, transform::Bool=false;
                    dtype::Symbol=:forward)

"""
function gradlogpdf(m::Model, block::Integer=0, transform::Bool=false;
                    dtype::Symbol=:forward)
  x0 = unlist(m, block, transform)
  value = gradlogpdf!(m, x0, block, transform, dtype=dtype)
  relist!(m, x0, block, transform)
  value
end

"""
    gradlogpdf!(m::Model, x::AbstractVector{T}, block::Integer=0,
                      transform::Bool=false; dtype::Symbol=:forward) where {T<:Real}

"""
function gradlogpdf!(m::Model, x::AbstractVector{T}, block::Integer=0,
                      transform::Bool=false; dtype::Symbol=:forward) where {T<:Real}
  f = x -> logpdf!(m, x, block, transform)
  if dtype == :Zygote
    Zygote.gradient(f, x)
  else
    FiniteDiff.finite_difference_gradient(f, x)
  end

end

"""
    logpdf(m::Model, block::Integer=0, transform::Bool=false)
"""
function logpdf(m::Model, block::Integer=0, transform::Bool=false)
  params = keys(m, :block, block)
  targets = keys(m, :target, block)
  logpdf(m, params, transform) + logpdf(m, setdiff(targets, params))
end
"""
    logpdf(m::Model, nodekeys::Vector{Symbol}, transform::Bool=false)

Compute the sum of log-densities for stochastic nodes.

Returns the resulting numeric value of summed log-densities.

* `m`: model containing the stochastic nodes for which to evaluate log-densities.

* `block` : sampling block of stochastic nodes over which to sum densities (default: all stochastic nodes).

* `nodekeys` : nodes over which to sum densities.

* `x` : value (possibly different than the current one) at which to evaluate densities.

* `transform` : whether to evaluate evaluate log-densities of block parameters on the link–transformed scale.
"""
function logpdf(m::Model, nodekeys::Vector{Symbol}, transform::Bool=false)
  lp = 0.0
  for key in nodekeys
    lp += logpdf(m[key], transform)
    isfinite(lp) || break
  end
  m.likelihood= isnan(lp) ? -Inf : lp
  m.likelihood
end

function pseudologpdf(m::Model, nodekeys::Vector{Symbol}, y, transform::Bool=false)
  lp = 0.0
  for key in nodekeys
    lp += pseudologpdf(m[key],y, transform)
    isfinite(lp) || break
  end
  lp
end

function conditional_likelihood(m::Model, nodekeys::Vector{Symbol}, args...)
  #println(nodekeys)
  lp = conditional_likelihood(m[nodekeys[1]], args...)
  #for key in nodekeys[2:end]
  #  lp .+= conditional_likelihood(m[key], args...)
  #end
  lp
end


function rand(m::Model, nodekeys::Vector{Symbol}, x::Int64)
  rand(m[nodekeys[1]], x)
end

"""
    logpdf(m::Model, x::AbstractArray{T}, block::Integer=0,
            transform::Bool=false) where {T<:Real}
"""
function logpdf(m::Model, x::AbstractArray{T}, block::Integer=0,
                transform::Bool=false) where {T<:Real}
  x0 = unlist(m, block)
  lp = logpdf!(m, x, block, transform)
  relist!(m, x0, block)
  lp
end

"""
    logpdf!(m::Model, x::N, block::Integer=0,
              transform::Bool=false) where N<:GeneralNode
"""
function logpdf!(m::Model, x::N, block::Integer=0,
                  transform::Bool=false) where N<:GeneralNode
  params = keys(m, :block, block)
  targets = keys(m, :target, block)
  m[params] = relist(m, x, params, transform)
  lp = logpdf(m, setdiff(params, targets), transform)
  for key in targets
    isfinite(lp) || break
    node = m[key]
    update!(node, m)
    lp += key in params ? logpdf(node, transform) : logpdf(node)
  end
  lp
end
"""
    gradlogpdf!(m::Model, x::AbstractArray{T}, block::Integer=0,transform::Bool=false)
     where T<:GeneralNode

"""
function gradlogpdf!(m::Model, x::AbstractArray{T}, block::Integer=0,transform::Bool=false) where T<:GeneralNode
  gradlogpdf!(m, x, block, transform)
end

"""
    gradlogpdf(m::Model, targets::Array{Symbol, 1})::Tuple{Float64, Array{Float64}}

Compute the gradient of log-densities for stochastic nodes.

Returns the resulting gradient vector. Method `gradlogpdf!()` additionally updates model `m` with supplied values `x`.

* `m` : model containing the stochastic nodes for which to compute the gradient.

* `block` : sampling block of stochastic nodes for which to compute the gradient (default: all stochastic nodes).

* `x`: value (possibly different than the current one) at which to compute the gradient.

* `transform`: whether to compute the gradient of block parameters on the link–transformed scale.

* `dtype` : type of differentiation for gradient calculations. Options are
  * `:central` : central differencing.

  * `:forward` : forward differencing.
"""
function gradlogpdf(m::Model, targets::Array{Symbol, 1})::Tuple{Float64, Array{Float64}}
  vp = 0.0
  gradp = Array[]
  for key in targets
    node = m[key]
    update!(node, m)
    v, grad = gradlogpdf(node)
    vp += v
    push!(gradp, grad)
  end
  m.likelihood = vp
  vp, .+(gradp...)
end
"""
    gradlogpdf!(m::Model, x::N, block::Integer=0,transform::Bool=false)::Tuple{Float64, Vector{Float64}}
        where N<:GeneralNode

Returns the resulting gradient vector. Method `gradlogpdf!()` additionally updates model `m` with supplied values `x`.
"""
function gradlogpdf!(m::Model, x::N, block::Integer=0,transform::Bool=false)::Tuple{Float64, Vector{Float64}} where N<:GeneralNode
  params = keys(m, :block, block)
  targets = keys(m, :target, block)
  m[params] = relist(m, x, params, transform)
  lp = logpdf(m, setdiff(params, targets), transform)


  # use thread parallelism
  # likelihood
  lik_res = @spawn gradlogpdf(m, targets)

  # prior
  vp, gradp =  gradlogpdf(m[params[1]], x)

  # get results from threads
  v, grad = fetch(lik_res)

  vp+v, gradp.+grad
end
"""
    logpdf!(m::Model, x::AbstractArray{T}, block::Integer=0,
                  transform::Bool=false) where {T<:Real}

Compute the sum of log-densities for stochastic nodes.

The resulting numeric value of summed log-densities. Method `logpdf!()` additionally updates model `m` with supplied values `x`.
"""
function logpdf!(m::Model, x::AbstractArray{T}, block::Integer=0,
                  transform::Bool=false) where {T<:Real}
  params = keys(m, :block, block)
  targets = keys(m, :target, block)
  m[params] = relist(m, x, params, transform)
  lp = logpdf(m, setdiff(params, targets), transform)
  lp = isnan(lp) ? -Inf : lp
  for key in targets
    isfinite(lp) || break
    node = m[key]
    update!(node, m)
    lp += key in params ? logpdf(node, transform) : logpdf(node)
  end
  lp
end

function rand!(m::Model, x::Int64, block::Integer=0)
  params = keys(m, :block, block)
  res = rand(m, params, x)
  res
end

"""
    sample!(m::Model, block::Integer=0)

Generate one MCMC sample of values for a specified model.

Returns the model updated with the MCMC sample and, in the case of `block=0`, the `iter` field incremented by 1.

* `m` : model specification.

* `block` : block for which to sample values (default: all blocks).


"""
function sample!(m::Model, block::Integer=0)
  m.iter += 1
  isoneblock = block != 0
  blocks = isoneblock ? block : 1:length(m.samplers)
  for b in blocks
    sampler = m.samplers[b]
    value = sampler.eval(m::Model, b::Int)
    if value != nothing
      m[sampler.params] = value
      update!(m, b)
    end
  end
  m.iter -= isoneblock
  m.likelihood = final_likelihood(m)
  m
end

function pseudologpdf!(m::Model, x::AbstractArray{T}, y::AbstractArray, block::Integer=0,
                  transform::Bool=false) where {T<:Real}
  params = keys(m, :block, block)
  targets = keys(m, :target, block)
  m[params] = relist(m, x, params, transform)
  lp = 0.0
  for key in targets
    isfinite(lp) || break
    node = m[key]
    update!(node, m)
    lp += key in params ? pseudologpdf(node, y, transform) : pseudologpdf(node,y)
  end
  lp
end


function conditional_likelihood!(m::Model, x::AbstractArray{T}, block::Integer=0,
                  args...) where {T<:Real}
  params = keys(m, :block, block)
  targets = keys(m, :target, block)
  m[params] = relist(m, x, params)

  lp = conditional_likelihood(m, targets, args...)

  lp
end


function final_likelihood(model::Model)::Float64
  logpdf(model, keys_output(model))
end
"""
    unlist(m::Model, block::Integer=0, transform::Bool=false)
"""
function unlist(m::Model, block::Integer=0, transform::Bool=false)
  unlist(m, keys(m, :block, block), transform)
end

function samparas(m::Model)
  moves = 0
  for i in m.samplers
    if typeof(i.tune) == PNUTSTune
      moves = i.tune.moves
    end
  end
 return moves
end
"""
    unlist(m::Model, monitoronly::Bool)
"""
function unlist(m::Model, monitoronly::Bool)
  f = function(key)
    node = m[key]
    lvalue = isa(node, AbstractTreeStochastic) ? unlist_tree(node) : unlist(node)
    monitoronly ? lvalue[node.monitor] : lvalue
  end
  r = vcat(map(f, keys(m, :dependent))..., m.likelihood)
  r
end
"""
    unlist(m::Model, nodekeys::Vector{Symbol}, transform::Bool=false)

Convert (unlist) sets of logical and/or stochastic node values to vectors.

Returns vectors of concatenated node values.

* `m` : model containing nodes to be unlisted or relisted.

* `block` : sampling block of nodes to be listed (default: all blocks).

* `nodekeys` : node(s) to be listed.

* `transform` : whether to apply a link transformation in the conversion.
"""
function unlist(m::Model, nodekeys::Vector{Symbol}, transform::Bool=false)
  vcat(map(key -> unlist(m[key], transform), nodekeys)...)
end

"""
    relist(m::Model, x::AbstractArray{T}, block::Integer=0,
            transform::Bool=false) where {T<:Real}
"""
function relist(m::Model, x::AbstractArray{T}, block::Integer=0,
                transform::Bool=false) where {T<:Real}
  relist(m, x, keys(m, :block, block), transform)
end
"""
    relist(m::Model, x::AbstractArray{T}, block::Integer=0,
            transform::Bool=false) where {T<:GeneralNode}
"""
function relist(m::Model, x::AbstractArray{T}, block::Integer=0,
               transform::Bool=false) where {T<:GeneralNode}

 relist(m, x, keys(m, :block, block), transform)
end

"""
    relist(m::Model, x::AbstractArray{T},
            nodekeys::Vector{Symbol}, transform::Bool=false) where {T<:Any}
"""
function relist(m::Model, x::AbstractArray{T},
                nodekeys::Vector{Symbol}, transform::Bool=false) where {T<:Any}
  values = Dict{Symbol,Union{Any, Real}}()

  N = length(x)
  offset = 0
  for key in nodekeys

    value, n = relistlength(m[key], view(x, (offset + 1):N), transform)

    values[key] = value
    offset += n
  end
  offset == length(x) ||
    throw(ArgumentError("incompatible number of values to put in nodes"))
  values
end


"""
    relist(m::Model, x::N, nodekeys::Vector{Symbol}, transform::Bool=false) where N<:GeneralNode

Reverse of unlist; ie. Converts vectors to sets of logical and/or stochastic node values. Same inputs and return values as unlist.
"""
function relist(m::Model, x::N, nodekeys::Vector{Symbol}, transform::Bool=false) where N<:GeneralNode
  values = Dict{Symbol,Any}()
  offset = 0
  for key in nodekeys
    value, n = relistlength(m[key], x, transform)
    values[key] = value
    offset += n
  end

  values
end


"""
    relist!(m::Model, x::AbstractArray{T}, block::Integer=0,
              transform::Bool=false) where {T<:Any}
"""
function relist!(m::Model, x::AbstractArray{T}, block::Integer=0,
                 transform::Bool=false) where {T<:Any}
  nodekeys = keys(m, :block, block)
  values = relist(m, x, nodekeys, transform)
  for key in nodekeys
    assign!(m, key, values[key])
  end
  update!(m, block)
end

function assign!(m::Model, key::Symbol, value::T) where T <: Real
  m[key].value = value
end

function assign!(m::Model, key::Symbol, value::T) where T <:GeneralNode
  m[key].value = value
end

function assign!(m::Model, key::Symbol, value::T) where T <: Array
  if isa(m[key], TreeVariate)
    @assert (2,) == size(value)
    d = Array{Float64, 1}(undef, 2)
  else
    @assert size(m[key].value) == size(value)
    d = similar(m[key].value)
  end
  for (ind, elem) in enumerate(value)
    d[ind] = elem
  end
  m[key].value = d
end


"""
    relist!(m::Model, x::AbstractArray{T}, nodekey::Symbol,
                  transform::Bool=false) where {T<:Real}

Reverse of unlist; ie. Converts vectors to sets of logical and/or stochastic node values. Same inputs as unlist.

Returns `m`, with values copied to the nodes.

"""
function relist!(m::Model, x::AbstractArray{T}, nodekey::Symbol,
                  transform::Bool=false) where {T<:Real}
  node = m[nodekey]
  m[nodekey] = relist(node, x, transform)
  update!(m, node.targets)
end

"""
    update!(m::Model, block::Integer=0)
"""
function update!(m::Model, block::Integer=0)
  nodekeys = block == 0 ? keys(m, :dependent) : m.samplers[block].targets
  update!(m, nodekeys)
end
"""
    update!(m::Model, nodekeys::Vector{Symbol})

Update values of logical and stochastic model node according to their relationship with others in a model.

Returns the model with updated nodes.

* `m` : mode with nodes to be updated.

* `block` : sampling block of nodes to be updated (default: all blocks).

* `nodekeys` : nodes to be updated in the given order.
"""
function update!(m::Model, nodekeys::Vector{Symbol})
  for key in nodekeys
    update!(m[key], m)
  end
  m
end
