#################### Chains ####################

#################### Constructors ####################
"""
    Chains(iters::Integer, params::Integer;
    start::Integer=1, thin::Integer=1, chains::Integer=1,
    names::Vector{T}=AbstractString[]) where {T<:AbstractString}
"""
function Chains(iters::Integer, params::Integer;
               start::Integer=1, thin::Integer=1, chains::Integer=1,
               names::Vector{T}=AbstractString[], ntrees::Integer=1, tree_names::Vector{Symbol}=Symbol[]) where {T<:AbstractString}
  value = Array{Float64}(undef, length(start:thin:iters), params, chains)
  value2 = Array{AbstractString}(undef, length(start:thin:iters), ntrees, chains)
  fill!(value, NaN)

  Chains(value, value2, start=start, thin=thin, names=names, tree_names=string.(tree_names))
end

"""
    Chains(value::Array{T, 3},
    start::Integer=1, thin::Integer=1,
    names::Vector{W}=AbstractString[], chains::Vector{V}=Int[], moves::Vector{V}=Int[0])
    where {T<:Real, U<:AbstractString, V<:Integer, W <: AbstractString}
"""
function Chains(value::Array{T, 3},
               start::Integer=1, thin::Integer=1,
               names::Vector{W}=AbstractString[], chains::Vector{V}=Int[], moves::Vector{V}=Int[0]) where {T<:Real, U<:AbstractString, V<:Integer, W <: AbstractString}
   Chains(value, Array{String, 3}(undef, size(value)), start=start,
          thin=thin, names=names, chains=chains, moves=moves)
end
"""
    Chains(value::Array{T, 3},
    value2::Array{U,3};
    start::Integer=1, thin::Integer=1,
    names::Vector{W}=AbstractString[], chains::Vector{V}=Int[], moves::Vector{V}=Int[0])
    where {T<:Real, U<:AbstractString, V<:Integer, W <: AbstractString}
"""
function Chains(value::Array{T, 3},
                value2::Array{U,3};
               start::Integer=1, thin::Integer=1,
               names::Vector{W}=AbstractString[], chains::Vector{V}=Int[],
               moves::Vector{V}=Int[0], tree_names::Vector{X}=AbstractString[]) where {T<:Real, U<:AbstractString, V<:Integer, W <: AbstractString, X <: AbstractString}
  n, p, m = size(value)

  if isempty(names)
    names = map(i -> "Param$i", 1:p)
  elseif length(names) != p
    throw(DimensionMismatch("size(value, 2) and names length differ"))
  end

  if isempty(chains)
    chains = collect(1:m)
  elseif length(chains) != m
    throw(DimensionMismatch("size(value, 3) and chains length differ"))
  end

  v = convert(Array{Float64, 3}, value)
  Chains(v, range(start, step=thin, length=n), AbstractString[names...], Int[chains...], value2, moves, tree_names)
end
"""
    Chains(value::Matrix{T};
    start::Integer=1, thin::Integer=1,
    names::Vector{U}=AbstractString[], chains::Integer=1)
    where {T<:Real, U<:AbstractString}
"""
function Chains(value::Matrix{T};
               start::Integer=1, thin::Integer=1,
               names::Vector{U}=AbstractString[], chains::Integer=1) where {T<:Real, U<:AbstractString}
  cont_vals = reshape(value, size(value, 1), size(value, 2), 1)
  tree_vals = Array{String, 3}(undef, size(cont_vals))
  Chains(cont_vals, tree_vals, start=start,
         thin=thin, names=names, chains=Int[chains])
end
"""
    Chains(value::Vector{T};
    start::Integer=1, thin::Integer=1,
    names::U="Param1", chains::Integer=1) where {T<:Real, U <: AbstractString}

Construct a `Chains` object that stores MCMC sampler output.

Returns an object of type `Chains`.

* `iters`: total number of iterations in each sampler run, of which `length(start:thin:iters)` outputted iterations will be stored in the object.

* `params`: number of parameters to store.

* `value`: array whose first, second (optional), and third (optional) dimensions index outputted iterations, parameter elements, and runs of an MCMC sampler, respectively.

* `start`: number of the first iteration to be stored.

* `thin`: number of steps between consecutive iterations to be stored.

* `chains`: number of simulation runs for which to store output, or indices to the runs (default: 1, 2, …).

* `names`: names to assign to the parameter elements (default: `"Param1"`, `"Param2"`, …).
"""
function Chains(value::Vector{T};
               start::Integer=1, thin::Integer=1,
               names::U="Param1", chains::Integer=1) where {T<:Real, U <: AbstractString}
  Chains(reshape(value, length(value), 1, 1), start=start, thin=thin,
         names=U[names], chains=Int[chains])
end


#################### Indexing ####################
"""
    Base.getindex(c::Chains, window, names, chains)
Subset MCMC sampler output. The syntax `c[i, j, k]` is converted to `getindex(c, i, j, k)`.

Subsetted sampler output stored in the same type of object as that supplied in the call.

* `c` : sampler output to subset.

* `window` : indices of the form `start:stop` or `start:thin:stop` can be used to subset iterations, where `start` and `stop` define a range for the subset and `thin` will apply additional thinning to existing sampler output.

* `names` : indices for subsetting of parameters that can be specified as strings, integers, or booleans identifying parameters to be kept. `ModelChains` may additionally be indexed by model node symbols.

* `chains` : indices for chains can be integers or booleans.

A value of `:` can be specified for any of the dimensions to indicate no subsetting.
"""
function Base.getindex(c::Chains, window, names, chains)
  inds1 = window2inds(c, window)
  inds2 = names2inds(c, names)
  if !isdefined(c.trees, 1)
    newsize=size(c.value[inds1, inds2, chains])
    Chains(c.value[inds1, inds2, chains], Array{String, length(newsize)}(undef, newsize),
           start = first(c) + (first(inds1) - 1) * step(c),
           thin = step(inds1) * step(c), names = c.names[inds2],
           chains = c.chains[chains], moves = c.moves)
  else
    Chains(c.value[inds1, inds2, chains], c.trees[inds1, :, chains],
           start = first(c) + (first(inds1) - 1) * step(c),
           thin = step(inds1) * step(c), names = c.names[inds2],
           chains = c.chains[chains], moves = c.moves)
  end
end

Base.lastindex(c::AbstractChains, i) = size(c, i)
"""
    Base.setindex!(c::AbstractChains, value, iters, names, chains)

Store MCMC sampler output at a given index. The syntax `c[i, j, k] = value` is converted to `setindex!(c, value, i, j, k)`.

Returns an object of the same type as c with the sampler output stored in the specified indices.

* `c` : object within which to store sampler output.

* `value` : sampler output.

* `iters` : iterations can be indexed as a `start:stop` or `start:thin:stop` range, a single numeric index, or a vector of indices; and are taken to be relative to the index range store in the `c.range` field.

* `names` : indices for subsetting of parameters can be specified as strings, integers, or booleans.

* `chains` : indices for chains can be integers or booleans.

A value of `:` can be specified for any of the dimensions to indicate no subsetting.
"""
function Base.setindex!(c::AbstractChains, value, iters, names, chains)
  setindex!(c.value, value, iters2inds(c, iters), names2inds(c, names), chains)
end

macro mapiters(iters, c)
  return esc(quote
    ($iters .- first($c)) ./ step($c) .+ 1.0
  end)
end

window2inds(c::AbstractChains, window) =
  throw(ArgumentError("$(typeof(window)) iteration indexing is unsupported"))
window2inds(c::AbstractChains, ::Colon) = window2inds(c, 1:size(c, 1))
window2inds(c::AbstractChains, window::Int) = window2inds(c, window:window)
window2inds(c::AbstractChains, window::AbstractRange) = begin
  range = @mapiters(window, c)
  a = max(ceil(Int, first(range)), 1)
  b = step(window)
  c = min(floor(Int, last(range)), size(c.value, 1))
  a:b:c
end

iters2inds(c::AbstractChains, iters) = iters
iters2inds(c::AbstractChains, ::Colon) = 1:size(c.value, 1)
iters2inds(c::AbstractChains, iters::AbstractRange) =
  convert(StepRange{Int, Int}, @mapiters(iters, c))
iters2inds(c::AbstractChains, iter::Real) = Int(@mapiters(iter, c))
iters2inds(c::AbstractChains, iters::Vector{T}) where {T<:Real} =
  Int[@mapiters(i, c) for i in iters]

names2inds(c::AbstractChains, names) = names
names2inds(c::AbstractChains, ::Colon) = 1:size(c.value, 2)
names2inds(c::AbstractChains, name::Real) = [name]
names2inds(c::AbstractChains, name::AbstractString) = names2inds(c, [name])
names2inds(c::AbstractChains, names::Vector{T}) where {T<:AbstractString} =
  indexin(names, c.names)


#################### Concatenation ####################
"""
    Base.cat(c1::AbstractChains, args::AbstractChains...; dims::Integer)

Concatenate input MCMC chains along a specified dimension. For dimensions other than the specified one, all input chains must have the same sizes, which will also be the sizes of the output chain. The size of the output chain along the specified dimension will be the sum of the sizes of the input chains in that dimension. vcat concatenates vertically along dimension 1, and has the alternative syntax [chain1; chain2; ...]. hcat concatenates horizontally along dimension 2, and has the alternative syntax [chain1 chain2 ...].

Returns a `Chains` object containing the concatenated input.

* `dim` : dimension (1, 2, or 3) along which to concatenate the input chains.

* `c1`, `args...` : Chains to concatenate.

"""
function Base.cat(c1::AbstractChains, args::AbstractChains...; dims::Integer)
  dims == 1 ? cat1(c1, args...) :
  dims == 2 ? cat2(c1, args...) :
  dims == 3 ? cat3(c1, args...) :
    throw(ArgumentError("cannot concatenate along dimension $dim"))
end

function cat1(c1::A, args::A...) where A <: AbstractChains

  range = c1.range
  for c in args
    last(range) + step(range) == first(c) ||
      throw(ArgumentError("noncontiguous chain iterations"))
    step(range) == step(c) ||
      throw(ArgumentError("chain thinning differs"))
    range = first(range):step(range):last(c)
  end

  names = c1.names
  tree_names = c1.tree_names
  all(c -> c.names == names, args) ||
    throw(ArgumentError("chain names differ"))
  all(c -> c.tree_names == tree_names, args) ||
    throw(ArgumentError("chain tree names differ"))

  chains = c1.chains
  all(c -> c.chains == chains, args) ||
    throw(ArgumentError("sets of chains differ"))

  value = cat(c1.value, map(c -> c.value, args)..., dims=1)

  if isassigned(c1.trees,1)
      trees = cat(c1.trees, map(c -> c.trees, args)..., dims=1)
  else
      trees = c1.trees
  end
  moves = +(c1.moves, map(c -> c.moves, args)...)

  Chains(value, trees, start=first(range), thin=step(range), names=names,
         chains=chains, moves=moves, tree_names=tree_names)
end

function cat2(c1::A, args::A...) where A <: AbstractChains
  range = c1.range
  all(c -> c.range == range, args) ||
    throw(ArgumentError("chain ranges differ"))

  names = c1.names
  n = length(names)
  for c in args
    names = union(names, c.names)
    n += length(c.names)
    n == length(names) ||
      throw(ArgumentError("non-unique chain names"))
  end

  chains = c1.chains
  all(c -> c.chains == chains, args) ||
    throw(ArgumentError("sets of chains differ"))

  value = cat(c1.value, map(c -> c.value, args)..., dims=2)
  if isassigned(c1.trees,1)
      trees = cat(c1.trees, map(c -> c.trees, args)..., dims=2)
  else
      trees = c1.trees
  end
  moves = +(c1.moves, map(c -> c.moves, args)...)
  Chains(value, trees, start=first(range), thin=step(range), names=names,
         chains=chains, moves=moves)
end

function cat3(c1::A, args::A...) where A <: AbstractChains
  range = c1.range
  all(c -> c.range == range, args) ||
    throw(ArgumentError("chain ranges differ"))

  names = c1.names
  all(c -> c.names == names, args) ||
    throw(ArgumentError("chain names differ"))
  tree_names = c1.tree_names
  all(c -> c.tree_names == tree_names, args) ||
    throw(ArgumentError("chain names differ"))

  value = cat(c1.value, map(c -> c.value, args)..., dims=3)

  if isassigned(c1.trees,1)
      value2 = cat(c1.trees, map(c -> c.trees, args)..., dims=3)
  else
      value2 = c1.trees
  end

  moves = +(c1.moves, map(c -> c.moves, args)...)
  Chains(value, value2, start=first(range), thin=step(range), names=names, moves=moves, tree_names=tree_names)
end

Base.hcat(c1::AbstractChains, args::AbstractChains...) = cat(c1, args..., dims=2)

Base.vcat(c1::AbstractChains, args::AbstractChains...) = cat(c1, args..., dims=1)


#################### Base Methods ####################
"""
    Base.keys(c::AbstractChains)

Returns names of parameter elements.

* `c` : Chain to return names of.

"""
function Base.keys(c::AbstractChains)
  c.names
end
"""
    Base.show(io::IO, c::AbstractChains)

Prints header and values of Chain.

* `io` : IO stream on which to print.

* `AbstractChains` : Chain to print.
"""
function Base.show(io::IO, c::AbstractChains)
  print(io, "Object of type \"$(summary(c))\"\n\n")
  print(io, header(c))
  isa(c, ModelChains) && show(io, c.sim_params; short=true)
  println()
  show(io, c.value)
end
"""
    Base.size(c::AbstractChains)

Returns Tuple containing last iteration of MCMC sampler output and dimensions of Chain dimensions of c.value.

* `c` : Chain object of interest.

"""
function Base.size(c::AbstractChains)
  dim = size(c.value)
  last(c), dim[2], dim[3]
end
"""
    Base.size(c::AbstractChains, ind)

Returns last iteration of MCMC sampler output, or dimension derived from C, according to value of ind.

* `c` : Chain object of interest.

* `ind` : index of tuple to return; 1 returns last iteration of MCMC sampler output, 2 and 3 return dimensions of c.value.
"""
function Base.size(c::AbstractChains, ind)
  size(c)[ind]
end

Base.first(c::AbstractChains) = first(c.range)
Base.step(c::AbstractChains) = step(c.range)
Base.last(c::AbstractChains) = last(c.range)


#################### Auxilliary Functions ####################

function combine(c::AbstractChains)
  n, p, m = size(c.value)
  value = Array{Float64}(undef, n * m, p)
  for j in 1:p
    idx = 1
    for i in 1:n, k in 1:m
      value[idx, j] = c.value[i, j, k]
      idx += 1
    end
  end
  value
end

function header(c::AbstractChains)
  string(
    "Iterations = $(first(c)):$(last(c))\n",
    "Thinning interval = $(step(c))\n",
    "Chains = $(join(map(string, c.chains), ","))\n",
    "Samples per chain = $(length(c.range))\n"
    #,
    #"NNI moves = $(c.moves)\n"
  )
end

function indiscretesupport(c::AbstractChains,
                           bounds::Tuple{Real, Real}=(0, Inf))
  nrows, nvars, nchains = size(c.value)
  result = Array{Bool}(undef, nvars * (nrows > 0))
  for i in 1:nvars
    result[i] = true
    for j in 1:nrows, k in 1:nchains
      x = c.value[j, i, k]
      if !isinteger(x) || x < bounds[1] || x > bounds[2]
        result[i] = false
        break
      end
    end
  end
  result
end

function link(c::AbstractChains)
  cc = copy(c.value)
  for j in 1:length(c.names)
    x = cc[:, j, :]
    if minimum(x) > 0.0
      cc[:, j, :] = maximum(x) < 1.0 ? logit.(x) : log.(x)
    end
  end
  cc
end
