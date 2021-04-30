#################### Model Expression Operators ####################

function modelfx(literalargs::Vector{Tuple{Symbol, DataType}}, f::Function)
  modelfxsrc(literalargs, f)[1]
end

function modelfxsrc(literalargs::Vector{Tuple{Symbol, DataType}}, f::Function)
  args = Expr(:tuple, map(arg -> Expr(:(::), arg[1], arg[2]), literalargs)...)
  expr, src = modelexprsrc(f, literalargs)
  fx = Core.eval(Main, Expr(:function, args, expr))
  (fx, src)
end


function modelexprsrc(f::Function, literalargs::Vector{Tuple{Symbol, DataType}})

  m = first(methods(f).ms) # returns a mehtod datatype
  argnames = Vector{Any}(undef, m.nargs)
  #ccall(:jl_fill_argnames, Cvoid, (Any, Any), m.source, argnames)
  argnames = Base.method_argnames(m)
  fkeys = Symbol[argnames[2:end]...]
  ftypes = DataType[m.sig.parameters[2:end]...]
  n = length(fkeys)

  literalinds = Int[]
  for (key, T) in literalargs
    i = findfirst(fkey -> fkey == key, fkeys)
    if i !== nothing && ftypes[i] == T
      push!(literalinds, i)
    end
  end
  nodeinds = setdiff(1:n, literalinds)

  all(T -> T == Any, ftypes[nodeinds]) ||
    throw(ArgumentError("model node arguments are not all of type Any"))

  modelargs = Array{Any}(undef, n)
  for i in nodeinds
    modelargs[i] = Expr(:ref, :model, QuoteNode(fkeys[i]))
  end
  for i in literalinds
    modelargs[i] = fkeys[i]
  end
  expr = Expr(:block, Expr(:(=), :f, f), Expr(:call, :f, modelargs...))

  (expr, fkeys[nodeinds])
end


#################### Mathematical Operators ####################

#isprobvec(p::AbstractVector) = isprobvec(convert(Vector{Float64}, p))

cummean(x::AbstractArray) = mapslices(cummean, x, dims=1)

function cummean(x::AbstractVector{T}) where {T<:Real}
  y = similar(x, Float64)
  xs = 0.0
  for i in 1:length(x)
    xs += x[i]
    y[i] = xs / i
  end
  y
end

dot(x) = dot(x, x)

@inline logit(x::Real) = log(x / (1.0 - x))
@inline invexpit(x::Real) = 1.0 / (x - x^2)
@inline invlogit(x::Real, λ::Real=1.0) = 1.0 / (exp(-λ*x) + 1.0)
@inline loginvlogit(x::Real, λ::Real=1.0) = -log(exp(-λ*x)+1.0)


function loginvlogitder(x::T, λ::T)::T where T <: Real
    λ/(exp(λ*x)+1)
end

function invlogder(x::T, λ::T)::T where T <: Real
    invlogit(x, λ)*invlogit(-x, λ)
end

Zygote.@adjoint function invlogit(x::Real, λ::Real)
    invlogit(x, λ), Δ -> (nothing, invlogder(Δ, λ))
end

Zygote.@adjoint function loginvlogit(x::Real, λ::Real)
    loginvlogit(x, λ), Δ -> (nothing, loginvlogitder(Δ, λ))
end



## Csorgo S and Faraway JJ. The exact and asymptotic distributions of the
## Cramer-von Mises statistic. Journal of the Royal Statistical Society,
## Series B, 58: 221-234, 1996.
function pcramer(q::Real)
  p = 0.0
  for k in 0:3
    c1 = 4.0 * k + 1.0
    c2 = c1^2 / (16.0 * q)
    p += gamma(k + 0.5) / factorial(k) * sqrt(c1) * exp(-c2) * besselk(0.25, c2)
  end
  p / (pi^1.5 * sqrt(q))
end


#################### Auxiliary Functions ####################
## pmap2 is a partial work-around for the pmap issue in julia 0.4.0 of worker
## node errors being blocked.  In single-processor mode, pmap2 calls map
## instead to avoid the error handling issue.  In multi-processor mode, pmap is
## called and will apply its error processing.

function pmap2(f::Function, lsts::AbstractArray)
  if (nprocs() > 1)
    pmap(f, lsts)
  else
    map(f, lsts)
  end
end


"""
  assign_mcmc_work(f::Function, lsts::AbstractArray
                   )::Tuple{Vector{Tuple{Chains, Model, ModelState}},
                      Array{Float64, 2}, Vector{AbstractString}}

--- INTERNAL ---
Starts the MCMC chain generation (on multiple workers if possible) and also
starts parallel ASDSF - if possible and requested by the user.
"""
function assign_mcmc_work(f::Function, lsts::AbstractArray
                          )::Tuple{Vector{Tuple{Chains, Model, ModelState}}, Array{Float64, 2}, Vector{AbstractString}}

  # count the number of trees per step per chain
  params = lsts[1][1].sim_params
  ASDSF::Bool = params.asdsf
  statnames::Vector{AbstractString} = []
  tree_dim::Int64 = 0
  for i in lsts[1][1].nodes
    if isa(i[2], TreeStochastic)
      push!(statnames, string("asdsf_", string(i[1])))
      tree_dim += 1
    end # if
  end # for
  ll::Int64 = length(lsts)
  # set up remote channels for communication across workers for ASDSF-on-the-fly
  r_channels = [RemoteChannel(()->Channel{Vector{AbstractString}}(100)) for x in 1:ll]
  ASDSF_vals::Vector{Vector{Float64}} = []
  n_trees::Int64 = ASDSF ? floor((last(lsts[1][3]) - lsts[1][4]) / params.freq) : 0
  results = Dict{Int64, Tuple{Chains, Model, ModelState}}()
  if nworkers() > 1
    # @sync and @async asure parallel computation
    @sync begin
      # assign each chain a free worker
      for (ind, list) in enumerate(lsts)
        @async results[ind] = ASDSF ? (@fetchfrom default_worker_pool() f(list, params.freq, r_channels[ind])) :
                                      (@fetchfrom default_worker_pool() f(list))
      end # for
      # assign a free worker the ASDSF computation
      @async ASDSF && append!(ASDSF_vals, @fetchfrom default_worker_pool() calculate_convergence(params, r_channels, n_trees, 1:tree_dim))
    end # begin
  else
    for (ind, list) in enumerate(lsts)
      results[ind] = ASDSF ? f(list, params.freq, r_channels[ind]) : f(list)
    end # for
    ASDSF && append!(ASDSF_vals, calculate_convergence(params, r_channels,
                                                       n_trees, 1:tree_dim))
  end # if/else
  close.(r_channels)
  if ASDSF
    stats = Array{Float64,2}(undef, length(ASDSF_vals[1]), length(ASDSF_vals))
    for i = 1:length(ASDSF_vals)
      stats[:, i] = ASDSF_vals[i]
    end # for
  else
    stats = zeros(Float64, 0, tree_dim)
  end # if/else
  return [results[i] for i in 1:ll], stats, statnames
end # assign_mcmc_work


ind2sub(dims, ind) = Tuple(CartesianIndices(dims)[ind])

showall(v) = showall(stdout, v)


"""
    lcp(str1::T, str2::T)::T where T <: AbstractString

Get the longest common prefix.
"""
function lcp(str1::T, str2::T)::T where T <: AbstractString
    minl::Int64 = min(length(str1), length(str2))
    outs::T = ""
    minl == 0 && return outs
    for i in 1:minl
      str1[i] == str2[i] ? outs *= str1[i] : return outs
    end # for
    return outs
end
