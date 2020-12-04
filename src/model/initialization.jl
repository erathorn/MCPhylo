#################### Model Initialization ####################

function setinits!(m::Model, inits::Dict{Symbol})::Model
  m.hasinputs || throw(ArgumentError("inputs must be set before inits"))
  m.iter = 0
  for key in keys(m, :dependent)
    node = m[key]
    if isa(node, AbstractStochastic) || isa(node, AbstractTreeStochastic)
      haskey(inits, key) || throw(ArgumentError("missing initial value for node : $key"))
      setinits!(node, m, inits[key])
    else
      setinits!(node, m)
    end
  end
  m.hasinits = true
  m
end

function setinits!(m::Model, inits::Vector{V} where V<:Dict{Symbol})::Model
  n = length(inits)
  m.states = Array{ModelState}(undef, n)
  for i in n:-1:1
    setinits!(m, inits[i])
    m.states[i] = ModelState(unlist(m), deepcopy(gettune(m)))
  end
  m
end

function setinputs!(m::Model, inputs::Dict{Symbol})::Model
  for key in keys(m, :input)
    haskey(inputs, key) ||
      throw(ArgumentError("missing inputs for node : $key"))
    isa(inputs[key], AbstractDependent) &&
      throw(ArgumentError("inputs cannot be Dependent types"))
    m.nodes[key] = deepcopy(inputs[key])
  end
  m.hasinputs = true
  m
end

function setsamplers!(m::Model, samplers::Vector{V} where V<:Sampler)::Model
  m.samplers = deepcopy(samplers)
  for sampler in m.samplers
    sampler.targets = keys(m, :target, sampler.params)
  end
  m
end
