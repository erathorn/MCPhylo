
mutable struct DMHTune <: SamplerTune
	logf::Union{Function, Missing}
	pseudolog::Union{Function, Missing} # AuxLog
	condlike::Union{Function, Missing}
	datakeys::Vector{Symbol}
	m::Int64
	scale::Float64
	link::Function
	invlink::Function
  
	DMHTune() = new()
  
	function DMHTune(f::Union{Function, Missing}, ps, cl, m::Int64, s::Float64;
						link::Function=identity, inverselink::Function=identity)
	  new(f, ps, cl, Vector{Symbol}(), m, s, link, inverselink)
	end
	function DMHTune(f::Union{Function, Missing}, ps, cl, dk::Vector{Symbol}, m::Int64, s::Float64;
						link::Function=identity, inverselink::Function=identity)
	  new(f, ps, cl, dk, m, s, link, inverselink)
	end
  end
  
  const DMHVariate = SamplerVariate{DMHTune}
  
  DMHTune(x::Vector, f, ps, cl, dk, m, s; args...) = DMHTune(f, ps, cl, dk, m, s; args...)
  
  #################### Sampler Constructor ####################
  
  function DMH(params::ElementOrVector{Symbol}, m::Int64, scale::Float64=1.0,
			   transform::Bool=true; args...)
  
		samplerfx = function(model::Model, block::Integer)
  
	  tune = gettune(model, block)
	  params = asvec(params)
  
	  targets = keys(model, :target, params)
	  tune.datakeys = targets
	  block = SamplingBlock(model, block, transform)
  
	  f = x -> logpdf!(block, x)
	  fp = (x, y) -> pseudologpdf!(block, x, y)
	  cl = (x, args...) -> conditional_likelihood!(block, x, args...)
	  v = SamplerVariate(block, f, fp, cl, targets, m, scale; args...)
  
	  sample!(v::DMHVariate, model)
  
	  relist(block, v)
	end
	Sampler(params, samplerfx, DMHTune())
  end
  
  sample!(v::DMHVariate, model) = DMH_sample!(v, v.tune, model)
  
  function DMH_sample!(v::DMHVariate, tune::DMHTune, model::Model)
	  # 1. propose theta prime
	  # 2. Generate the auxiliary variable using theta prime
  
	  data = unlist(model[tune.datakeys[1]])
	  data = reshape(data, size(getindex(model, tune.datakeys[1])))
	  nfeatures, nlang = size(data)
	  #xobs = Array{Any}(undef, nfeatures)
	  #nobs = Array{Float64}(undef, nfeatures)
  
	  # Separate missing from observed indices
	  #for f in 1:nfeatures
	  #	xmis = findall(x -> x .== -10, data[f,:])
	  #	xobs[f,:] = findall(x -> x .≠ -10, data[f,:])
	  #	nobs[f] = length(xobs[f,:])
	  #end
  
	  @assert v.tune.m >= nlang
  
	  # store old value for possible future reference
	  θ = deepcopy(v.value)
	  # this way of generating theta_prime from the current values of theta
	  # takes care of the transition probability from theta_prime to theta and vice versa
	  # the values equal and will cancel out.
	  θ_prime = θ + v.tune.scale .* rand(Normal(0.0, 1.0), length(v))
  
	  #println("real")
	  # calculate logpdf values
	  lf_xt = tune.logf(v.value)
  
	  #println("θ_prime $θ_prime")
	  lf_xtp = tune.logf(θ_prime)
  
	  # prematurely assign, to allow computations to go through properly
	  v[:] = θ_prime
  
	  # Sample new pseudo observations based on the θ_prime
	  y = inner_sampler(v, deepcopy(data)) #xobs
  
	  # calculate logpdfs based on pseudo observations
	  #println("pseudo")
	  lf_ytp = tune.pseudolog(v.value, y) # -> Do I calculate the prior here??? This should not happen!!!
	  lf_yt = tune.pseudolog(θ, y)
  
	  # calculate acceptance probability (proposal distribution?)
	  r = exp((lf_yt + lf_xtp) - (lf_xt + lf_ytp))
	  #println("r $r")
  
	  # RWM acceptance logic is a bit reversed here. Above the new value is
	  # prematurely assigned to allow computations in the inner sampler to go through.
	  # If the sample is accepted nothing needs to be done anymore, otherwise the
	  # old value will be reassigned.
  
	  if rand() > r
		  #println("rejected")
		  # sample is rejected, so use the old value.
		  v[:] = θ
	  end
	  #println("v ", v[:])
	  v
  
	  # here?
  end
  
  function inner_sampler(v::DMHVariate, X::Array{N,2})::Array{N, 2} where N <: Real
	  nfeatures, nlangs = size(X)
	  counter = zero(Int64)
  
	  while true
		  random_language_idx = shuffle(1:nlangs)
		  random_feature_idx = shuffle(1:nfeatures)
		  @inbounds for i in random_language_idx
			  @inbounds for f in random_feature_idx
				  
				  probs = v.tune.condlike(v, X, i, f)	
				  new_x = sample(1:length(probs), Weights(probs))
					X[f, i] = new_x
			  #	end
			  end
			  counter += 1
			  if counter > v.tune.m
				  return X
			  end
		  end
	  end
  end
  