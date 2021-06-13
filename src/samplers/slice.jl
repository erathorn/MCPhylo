#################### Slice Sampler ####################

#################### Types and Constructors ####################

const SliceForm = Union{Univariate, Multivariate}

mutable struct SliceTune{F<:SliceForm} <: SamplerTune
  logf::Union{Function, Missing}
  width::Union{Float64, Vector{Float64}}

  SliceTune{F}() where {F<:SliceForm} = new{F}()

  SliceTune{F}(x::DenseVector, width) where {F<:SliceForm} =
    SliceTune{F}(x, width, missing)

  SliceTune{F}(x::DenseVector, width::Real, logf::Union{Function, Missing}) where
    {F<:SliceForm} = new{F}(logf, Float64(width))

  SliceTune{F}(x::DenseVector, width::Vector, logf::Union{Function, Missing}) where
    {F<:SliceForm} = new{F}(logf, convert(Vector{Float64}, width))
end


const SliceUnivariate = SamplerVariate{SliceTune{Univariate}}
const SliceMultivariate = SamplerVariate{SliceTune{Multivariate}}

validate(v::SamplerVariate{SliceTune{F}}) where {F<:SliceForm} =
  validate(v, v.tune.width)

validate(v::SamplerVariate{SliceTune{F}}, width::Float64) where {F<:SliceForm} = v

function validate(v::SamplerVariate{SliceTune{F}}, width::Vector) where {F<:SliceForm}
  n = length(v)
  length(width) == n ||
    throw(ArgumentError("length(width) differs from variate length $n"))
  v
end


#################### Sampler Constructor ####################
"""
Slice(params::ElementOrVector{Symbol},
                width::ElementOrVector{T},
                ::Type{F}=Multivariate;
                transform::Bool=false) where {T<:Real, F<:SliceForm}

Construct a `Sampler` object for Slice sampling. Parameters are assumed to be
continuous, but may be constrained or unconstrained.

Returns a `Sampler{SliceTune{Univariate}}` or `Sampler{SliceTune{Multivariate}}`
type object if sampling univariately or multivariately, respectively.

  * `params`: stochastic node(s) to be updated with the sampler.

  * `width`: scaling value or vector of the same length as the combined elements of nodes `params`, defining initial widths of a hyperrectangle from which to simulate values.

  * `F` : sampler type. Options are
      * `:Univariate` : sequential univariate sampling of parameters.
      * `:Multivariate` : joint multivariate sampling.

  * `transform`: whether to sample parameters on the link-transformed scale (unconstrained parameter space). If `true`, then constrained parameters are mapped to unconstrained space according to transformations defined by the Stochastic `unlist()` function, and `width` is interpreted as being relative to the unconstrained parameter space. Otherwise, sampling is relative to the untransformed space.
"""
function Slice(params::ElementOrVector{Symbol},
                width::ElementOrVector{T},
                ::Type{F}=Multivariate;
                transform::Bool=false) where {T<:Real, F<:SliceForm}
  samplerfx = function(model::Model, block::Integer)
    block = SamplingBlock(model, block, transform)
    v = SamplerVariate(block, width)
    sample!(v, x -> logpdf!(block, x))
    relist(block, v.value)
  end
  Sampler(params, samplerfx, SliceTune{F}())
end


#################### Sampling Functions ####################

sample!(v::Union{SliceUnivariate, SliceMultivariate}) = sample!(v, v.tune.logf)
"""
    sample!(v::Union{SliceUnivariate, SliceMultivariate}, logf::Function)

Draw one sample from a target distribution using the Slice univariate or
multivariate sampler. Parameters are assumed to be continuous, but may be
constrained or unconstrained.

Returns `v` updated with simulated values and associated tuning parameters.
"""
function sample!(v::Union{SliceUnivariate, SliceMultivariate}, logf::Function)
    eltype(v.value) <:GeneralNode ? sample_node!(v, logf) : sample_number!(v, logf)
end


function sample_node!(v::SliceUnivariate, logf::Function)
  tree = v.value[1]

  logf0 = logf(tree)
  blv = get_branchlength_vector(tree)

  n = length(blv)
  lower = blv - v.tune.width .* rand(n)
  lower[lower .< 0.0] .= 0.0
  upper = lower .+ v.tune.width

  for i in 1:n
    p0 = logf0 + log(rand())

    x = blv[i]
    blv[i] = rand(Uniform(lower[i], upper[i]))
    while true
      set_branchlength_vector!(tree, blv)
      logf0 = logf(tree)
      logf0 < p0 || break
      value = blv[i]
      if value < x
        lower[i] = value
      else
        upper[i] = value
      end
      blv[i] = rand(Uniform(lower[i], upper[i]))
    end
  end

  v
end


function sample_node!(v::SliceMultivariate, logf::Function)
  tree = v.value[1]

  p0 = logf(tree) + log(rand())
  blv = get_branchlength_vector(tree)
  org = deepcopy(blv)


  n = length(blv)
  lower = blv - v.tune.width .* rand(n)
  lower[lower .< 0.0] .= 0.0
  upper = lower .+ v.tune.width

  blv = v.tune.width .* rand(n) + lower
  set_branchlength_vector!(tree, blv)
  while logf(tree) < p0
    for i in 1:n
      value = blv[i]
      if value < org[i]
        lower[i] = value
      else
        upper[i] = value
      end
      blv[i] = rand(Uniform(lower[i], upper[i]))
    end
    set_branchlength_vector!(tree, blv)
  end

  v.value[1] = tree
  v
end



function sample_number!(v::SliceUnivariate, logf::Function)
  logf0 = logf(v.value)

  n = length(v)
  lower = v - v.tune.width .* rand(n)
  upper = lower .+ v.tune.width

  for i in 1:n
    p0 = logf0 + log(rand())

    x = v[i]
    v[i] = rand(Uniform(lower[i], upper[i]))
    while true
      logf0 = logf(v.value)
      logf0 < p0 || break
      value = v[i]
      if value < x
        lower[i] = value
      else
        upper[i] = value
      end
      v[i] = rand(Uniform(lower[i], upper[i]))
    end
  end

  v
end


function sample_number!(v::SliceMultivariate, logf::Function)
 
  p0 = logf(v.value) + log(rand())
 
  n = length(v)
  lower = v.value - convert(typeof(v.value),v.tune.width .* rand(n))
  #lower = convert(typeof(v.value), lower)
  upper = lower .+ v.tune.width
  pr = convert(typeof(v.value),rand(n))
  x = v.tune.width .* pr + lower
  ln = length(v.value)
  
  threads = min(ln, MAX_THREADS_PER_BLOCK)
  threads_x = floor(Int, sqrt(threads))
  threads_y = threads ÷ threads_x
  thds = (threads_x, threads_y)
  blo = ceil.(Int, ln ./ threads)
  while logf(x) < p0
    
    #for i in 1:n
      #value = x[i]
      linds = x .< v.value
      uinds = x .> v.value
      
      @cuda blocks = blo threads = thds kernel_place!(lower, x, linds)
      @cuda blocks = blo threads = thds kernel_place!(upper, x, uinds)
      #synchronize()
      #if value < v[i]
      #  lower[i] = value
      #else
      #  upper[i] = value
      #end
      #x[i] = rand(Uniform(lower[i], upper[i]))
      #@show x[1:5]
      Random.rand!(x)
    
      @cuda blocks = blo threads = thds kernel_uniform(x, upper, lower)
      #synchronize()
    
  end
  CUDA.@allowscalar(v[:] = x)
  v
end

const MAX_THREADS_PER_BLOCK = CUDA.attribute(
   device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
)


function kernel_uniform(v, upper, lower)
  bId = (blockIdx().x-1) + (blockIdx().y-1) * gridDim().x
  i =  bId * (blockDim().x * blockDim().y) + ((threadIdx().y-1) * blockDim().x) + threadIdx().x
  if i <= length(v)
      v[i] = v[i] * (upper[i] - lower[i]) + lower[i]
  end
  nothing
end

function kernel_place!(a, b, c)
  bId = (blockIdx().x-1) + (blockIdx().y-1) * gridDim().x
  i =  bId * (blockDim().x * blockDim().y) + ((threadIdx().y-1) * blockDim().x) + threadIdx().x
  if i <= length(c) && c[i]    
      a[i] = b[i]
  end
  return nothing
end