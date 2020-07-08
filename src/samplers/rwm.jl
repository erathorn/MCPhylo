#################### Random Walk Metropolis ####################

#################### Types and Constructors ####################

mutable struct RWMTune <: SamplerTune
  logf::Union{Function, Missing}
  scale::Union{Float64, Vector{Float64}}
  proposal::SymDistributionType

  RWMTune() = new()

  function RWMTune(x::Vector, scale::Real, logf::Union{Function, Missing};
                   proposal::SymDistributionType=Normal)
    new(logf, Float64(scale), proposal)
  end

  function RWMTune(x::Vector, scale::Vector{T},
                  logf::Union{Function, Missing};
                  proposal::SymDistributionType=Normal) where {T<:Real}
    new(logf, convert(Vector{Float64}, scale), proposal)
  end
end

RWMTune(x::Vector, scale::ElementOrVector{T}; args...) where {T<:Real} =
  RWMTune(x, scale, missing; args...)


const RWMVariate = SamplerVariate{RWMTune}

validate(v::RWMVariate) = validate(v, v.tune.scale)

validate(v::RWMVariate, scale::Float64) = v

function validate(v::RWMVariate, scale::Vector)
  n = length(v)
  length(scale) == n ||
    throw(ArgumentError("length(scale) differs from variate length $n"))
  v
end


#################### Sampler Constructor ####################

function RWM(params::ElementOrVector{Symbol},
              scale::ElementOrVector{T}; args...) where {T<:Real}
  samplerfx = function(model::Model, block::Integer)
    block = SamplingBlock(model, block, true)
    v = SamplerVariate(block, scale; args...)
    sample!(v, x -> logpdf!(block, x))
    relist(block, v)
  end
  Sampler(params, samplerfx, RWMTune())
end


#################### Sampling Functions ####################

sample!(v::RWMVariate) = sample!(v, v.tune.logf)

function sample!(v::RWMVariate, logf::Function)
  typeof(v[1]) <:AbstractNode ? sample_node!(v, logf) : sample_number!(v, logf)
end


function sample_node!(v::RWMVariate, logf::Function)
  tree = v[1]
  tc = deepcopy(tree)
  move = NNI!(tree)
  while move == 0
    move = NNI!(tree)
  end
  if rand() < exp(logf(tc) - logf(tree))
    v[1] = tree
  else
    v[1] = tc
  end
  v
end

function sample_number!(v::RWMVariate, logf::Function)
  x = v + v.tune.scale .* rand(v.tune.proposal(0.0, 1.0), length(v))
  if rand() < exp(logf(x) - logf(v.value))
    v[:] = x
  end
  v
end
