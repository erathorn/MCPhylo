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
    if i != nothing && ftypes[i] == T
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

isprobvec(p::AbstractVector) = isprobvec(convert(Vector{Float64}, p))

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
"""
  assign_mcmc_work(f::Function, lsts::AbstractArray, asdsf::Bool,
                   ASDSF_freq::Int64, ASDSF_min_splits::Float64)

--- INTERNAL ---
Starts the MCMC chain generation (on multiple workers if possible) and also
starts parallel ASDSF - if possible and requested by the user.
"""
function assign_mcmc_work(f::Function, lsts::AbstractArray, asdsf::Bool, ASDSF_freq::Int64,
               ASDSF_min_splits::Float64)
  tree_dim::Int64 = 0
  for i in lsts[1][1].nodes
    if isa(i[2], TreeStochastic)
      tree_dim += 1
    end # if
  end # for
  ll::Int64 = length(lsts)
  r_channels = [RemoteChannel(()->Channel{Vector{AbstractString}}(10)) for x in 1:ll]
  ASDSF_results::Vector{Vector{Float64}} = []
    if ll <= nworkers()
      results = Dict{Int64, Tuple{Chains, Model, ModelState}}()
      @sync begin
        if ll == nworkers() && asdsf
          asdsf = false
          @warn "Not enough workers to run ASDSF on-the-fly parallel to the chains. Generating chains normally without ASDSF"
        end # if
        if asdsf
          n_trees::Int64 = floor((last(lsts[1][3]) - lsts[1][4]) / ASDSF_freq)
          @async ASDSF_results = @fetchfrom workers()[end] ASDSF(r_channels, n_trees,
                                                                 1:tree_dim, ASDSF_min_splits)
        end # if
        for (ind, list) in enumerate(lsts)
          if asdsf
            @async results[ind] =
              @fetchfrom workers()[ind] f(list, ASDSF_freq, r_channels[ind])
          else
            @async results[ind] = @fetchfrom workers()[ind] f(list)
          end # if/else
        end # for
      end # begin
      close.(r_channels)
      println(ASDSF_results)
      return [results[i] for i in 1:ll], ASDSF_results
    else
      return map(f, lsts), [Float64[]]
    end # if/else
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
