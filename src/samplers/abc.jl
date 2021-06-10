#################### Approximate Bayesian Computation ####################

#################### Types and Constructors ####################

mutable struct ABCTune <: SamplerTune
    params::Vector{Symbol}
    datakeys::Vector{Symbol}
    Tsim::Vector{Vector{Float64}}
    Tobs::Vector{Float64}
    epsilon::Float64
    epsilon_vec::Vector{Float64}
    epsilonprime::Vector{Float64}
    kernel::Function
    dist::Function
    proposal::Function
    maxdraw::Integer
    nsim::Integer
    decay::Real
    randeps::Bool
    obsdata::Function
    simdata::Function
    summarizenodes::Function
    pi_epsilon0::Float64


    function ABCTune()
        new()
    end

    function ABCTune(
        x,
        params::Vector{Symbol},
        datakeys::Vector{Symbol},
        epsilon::Float64,
        kernel::Function,
        dist::Function,
        proposal::Function,
        maxdraw::Integer,
        nsim::Integer,
        decay::Real,
        randeps::Bool,
        summarizenodes::Function,
        model::Model,
    )
        new(
            params,
            datakeys,
            Vector{Vector{Float64}}(),
            Vector{Vector{Float64}}(),
            epsilon,
            Array{Float64}(undef, nsim),
            Array{Float64}(undef, nsim),
            kernel,
            dist,
            proposal,
            maxdraw,
            nsim,
            decay,
            randeps,
            (model, key) -> unlist(model[key]),
            (model, key) -> unlist(model[key], rand(model[key])),
            summarizenodes,
            0.0,
        )
    end

end

const ABCVariate = SamplerVariate{ABCTune}

#################### Sampler Constructor ####################

"""
    ABC(params::ElementOrVector{Symbol},
    scale::ElementOrVector{T}, summary::Function,
    epsilon::Real; kernel::KernelDensityType=SymUniform,
    dist::Function=(Tsim, Tobs) -> sqrt(sum(abs2, Tsim - Tobs)),
    proposal::SymDistributionType=Normal, maxdraw::Integer=1,
    nsim::Integer=1, decay::Real=1.0, randeps::Bool=false,
    args...)
    
    Construct a `Sampler` object for ABC sampling. Parameters are assumed to be continuous, but may be constrained or unconstrained.
    Returns a `Sampler{ABCTune}` type object.
    * `params`: stochastic node(s) to be updated with the sampler.  Constrained parameters are mapped to unconstrained space according to transformations defined by the Stochastic `unlist()` function.
    * `scale` : scaling value or vector of the same length as the combined elements of nodes `params` for the `proposal` distribution.  Values are relative to the unconstrained parameter space, where candidate draws are generated.
    * `summary` : function that takes a vector of observed or simulated data and returns a summary statistic or vector of statistics.
    * `epsilon` : target tolerance for determining how similar observed and simulated data summary statistics need to be in order to accept a candidate draw.
    * `kernel` : weighting kernel density of type `Biweight`, `Cosine`, `Epanechnikov`, `Normal`, `SymTriangularDist`, `SymUniform`, or `Triweight` to use in measuring similarity between observed and simulated data summary statistics.  Specified `epsilon` determines the standard deviation of Normal kernels and widths of the others.
    * `dist` : positive function for the kernel density to compute distance between vectors of observed (`Tobs`) and simulated (`Tsim`) data summary statistics (default: Euclidean distance).
    * `proposal` : symmetric distribution of type `Biweight`, `Cosine`, `Epanechnikov`, `Normal`, `SymTriangularDist`, `SymUniform`, or `Triweight` to be centered around current parameter values and used to generate proposal draws.  Specified `scale` determines the standard deviations of Normal proposals and widths of the others.
    * `maxdraw` : maximum number of unaccepted candidates to draw in each call of the sampler.  Draws are generated until one is accepted or the maximum is reached.  Larger values increase acceptance rates at the expense of longer runtimes.
    * `nsim` : number of data sets to simulate in deciding whether to accept a candidate draw.  Larger values lead to closer approximations of the target distribution at the expense of longer runtimes.
    * `decay` : if `0 < decay <= 1`, the rate at which internal tolerances are monotonically decreased from the initial distance between observed and simulated summary statistics toward the maximum of each subsequent distance and `epsilon`; if `decay = 0`, internal tolerances are fixed at `epsilon`.
    * `randeps` : whether to perturb internal tolerances by random exponential variates.
    * `args...` : additional keyword arguments to be passed to the `dist` function.
"""
function ABC(
    params::ElementOrVector{Symbol},
    scale::ElementOrVector{T},
    summary::Function,
    epsilon::Real;
    kernel::KernelDensityType=SymUniform,
    dist::Function=(Tsim, Tobs) -> sqrt(sum(abs2, Tsim - Tobs)),
    proposal::SymDistributionType=Normal,
    maxdraw::Integer=1,
    nsim::Integer=1,
    decay::Real=1.0,
    randeps::Bool=false,
    args...,
) where {T <: Real}
    0 <= decay <= 1 || throw(ArgumentError("decay is not in [0, 1]"))

    params = asvec(params)

    kernelpdf = (epsilon, d) -> pdf(kernel(0.0, epsilon), d)


    samplerfx = function (model::Model, block::Integer)

        sblock = SamplingBlock(model, block, true)
        tune = gettune(model, block)
        targets = keys(model, :target, params)
        stochastics = keys(model, :stochastic)
        datakeys = intersect(setdiff(targets, params), stochastics)

        summarizenodes =
            length(datakeys) > 1 ?
            (mo, data) -> vcat(map(key -> summary(data(mo, key)), datakeys)...) :
            (mo, data) -> asvec(summary(data(mo, datakeys[1])))

        proposal_fun = theta0 -> theta0 + scale * rand(proposal(0.0, 1.0), length(theta0))

        v = SamplerVariate(
            sblock,
            params,
            datakeys,
            epsilon,
            kernelpdf,
            dist,
            proposal_fun,
            maxdraw,
            nsim,
            decay,
            randeps,
            summarizenodes,
            model,
        )

        fp = x -> logpdf(model, x, true)

        sample!(v, fp, model, block, model.iter == 1)
    end

    Sampler(params, samplerfx, ABCTune())
end

"""
    ABC(params::ElementOrVector{Symbol},
    scale::ElementOrVector{T}, summary::Function,
    epsilon::Real, proposal::Function; kernel::KernelDensityType=SymUniform,
    dist::Function=(Tsim, Tobs) -> sqrt(sum(abs2, Tsim - Tobs)),
    maxdraw::Integer=1,
    nsim::Integer=1, decay::Real=1.0, randeps::Bool=false,
    args...)

    Construct a `Sampler` object for ABC sampling. Parameters are assumed to be continuous, but may be constrained or unconstrained.
    Returns a `Sampler{ABCTune}` type object.
    * `params`: stochastic node(s) to be updated with the sampler.  Constrained parameters are mapped to unconstrained space according to transformations defined by the Stochastic `unlist()` function.
    * `summary` : function that takes a vector of observed or simulated data and returns a summary statistic or vector of statistics.
    * `epsilon` : target tolerance for determining how similar observed and simulated data summary statistics need to be in order to accept a candidate draw.
    * `proposal` : a proposal function which proposes a new value. 
    * `kernel` : weighting kernel density of type `Biweight`, `Cosine`, `Epanechnikov`, `Normal`, `SymTriangularDist`, `SymUniform`, or `Triweight` to use in measuring similarity between observed and simulated data summary statistics.  Specified `epsilon` determines the standard deviation of Normal kernels and widths of the others.
    * `dist` : positive function for the kernel density to compute distance between vectors of observed (`Tobs`) and simulated (`Tsim`) data summary statistics (default: Euclidean distance).
    * `maxdraw` : maximum number of unaccepted candidates to draw in each call of the sampler.  Draws are generated until one is accepted or the maximum is reached.  Larger values increase acceptance rates at the expense of longer runtimes.
    * `nsim` : number of data sets to simulate in deciding whether to accept a candidate draw.  Larger values lead to closer approximations of the target distribution at the expense of longer runtimes.
    * `decay` : if `0 < decay <= 1`, the rate at which internal tolerances are monotonically decreased from the initial distance between observed and simulated summary statistics toward the maximum of each subsequent distance and `epsilon`; if `decay = 0`, internal tolerances are fixed at `epsilon`.
    * `randeps` : whether to perturb internal tolerances by random exponential variates.
    * `args...` : additional keyword arguments to be passed to the `dist` function.
"""
function ABC(
    params::ElementOrVector{Symbol},
    summary::Function,
    epsilon::Real,
    proposal::Function;
    kernel::KernelDensityType=SymUniform,
    dist::Function=(Tsim, Tobs) -> sqrt(sum(abs2, Tsim - Tobs)),
    maxdraw::Integer=1,
    nsim::Integer=1,
    decay::Real=1.0,
    randeps::Bool=false,
    args...,
)
    0 <= decay <= 1 || throw(ArgumentError("decay is not in [0, 1]"))

    params = asvec(params)
    kernelpdf = (epsilon, d) -> pdf(kernel(0.0, epsilon), d)

    samplerfx = function (model::Model, block::Integer)

        sblock = SamplingBlock(model, block, true)
        tune = gettune(model, block)
        targets = keys(model, :target, params)
        stochastics = keys(model, :stochastic)
        datakeys = intersect(setdiff(targets, params), stochastics)

        summarizenodes =
            length(datakeys) > 1 ?
            (mo, data) -> vcat(map(key -> summary(data(mo, key)), datakeys)...) :
            (mo, data) -> asvec(summary(data(mo, datakeys[1])))

        v = SamplerVariate(
            sblock,
            params,
            datakeys,
            epsilon,
            kernelpdf,
            dist,
            proposal,
            maxdraw,
            nsim,
            decay,
            randeps,
            summarizenodes,
            model,
        )

        fp = let model = model
             x -> logpdf(model, x, true)
         end
        sample!(v, fp, model, block, model.iter == 1)
    end



    Sampler(params, samplerfx, ABCTune())
    end

#################### Sampler Functions ####################


function sample!(v::ABCVariate, lp::Function, model::Model, block::Integer, gen1::Bool)

    tune = v.tune
    if gen1
        pi_epsilon0 = 0.0
        pi_error0 = 1.0

        v.tune.Tobs = v.tune.summarizenodes(model, v.tune.obsdata)
        v.tune.Tsim = Array{Vector{Float64}}(undef, v.tune.nsim)
        v.tune.epsilonprime = Array{Float64}(undef, v.tune.nsim)
        for i = 1:tune.nsim
            ## simulated data summary statistics for current parameter values
            tune.Tsim[i] = tune.summarizenodes(model, tune.simdata)
            d = tune.dist(tune.Tsim[i], tune.Tobs)

            # starting tolerance
            tune.epsilon_vec[i] = tune.decay > 0 ? max(tune.epsilon, d) : tune.epsilon
            
            if tune.randeps
                dexp = Exponential(tune.epsilon_vec[i])
                tune.epsilonprime[i] = rand(dexp)
                pi_error0 = pdf(dexp, tune.epsilonprime[i])
            else
                tune.epsilonprime[i] = tune.epsilon_vec[i]
            end

            ## kernel density evaluation
            pi_epsilon0 += tune.kernel(tune.epsilonprime[i], d) * pi_error0
            tune.pi_epsilon0 = pi_epsilon0
        end
    end
    ABC_sample(v, lp, model, block)
end


"""
    Do the ABC sampling. This version uses threads.
"""
function ABC_sample(v::ABCVariate, lp::Function, model::Model, block::Integer)

    tune = v.tune

    ## current parameter and density values
    theta0::Array{Float64} = v.value
    logprior0::Float64 = lp(tune.params)

    lo::Base.ReentrantLock = Base.ReentrantLock()
    flag = Threads.Atomic{Bool}(true)

    Threads.@threads for k = 1:tune.maxdraw
        if flag[]
            ## candidate draw and prior density value
            theta1::Array{Float64} = tune.proposal(theta0)
            other_model = deepcopy(model)

                logprior1::Float64 = mapreduce(
                keyval -> logpdf(other_model[keyval[1]], keyval[2]),
                +,
                relist(other_model, theta1, block, true),
            )

            if logprior1 == -Inf
                continue
            end

            relist!(other_model, theta1, block, true)

            ## tolerances and kernel density
            pi_epsilon1::Float64 = 0.0
            epsilon1::Vector{Float64} = similar(tune.epsilon_vec)
            epsilonprime1::Vector{Float64} = similar(tune.epsilonprime)
            Tsim1::Vector{Vector{Float64}} = similar(tune.Tsim)
            for i = 1:tune.nsim
                ## simulated data summary statistics for candidate draw
                Tsim1[i] = tune.summarizenodes(other_model, tune.simdata)
                d::Float64 = tune.dist(Tsim1[i], tune.Tobs)
                pi_error1::Float64 = 1.0
                ## monotonically decrease tolerance to target
                epsilon1[i] =
                    (1 - tune.decay) * tune.epsilon_vec[i] +
                    tune.decay * max(tune.epsilon, min(d, tune.epsilon_vec[i]))
                    
                if tune.randeps
                    dexp = Exponential(epsilon1[i])
                    epsilonprime1[i] = rand(dexp)

                    pi_error1 = pdf(dexp, tune.epsilonprime1[i])
                else
                    epsilonprime1[i] = epsilon1[i]
                end

                ## kernel density evaluation
                pi_epsilon1 += tune.kernel(epsilonprime1[i], d) * pi_error1
            end

            # accept/reject the candidate draw
            # check if simulation has been accepted in other thread
            if rand() < pi_epsilon1 / tune.pi_epsilon0 * exp(logprior1 - logprior0)
                # lock threads to set values
                if flag[]
                    Threads.atomic_xchg!(flag, false)
                    Base.lock(lo)
                    theta0 = theta1
                    tune.Tsim = Tsim1
                    tune.pi_epsilon0 = pi_epsilon1
                    tune.epsilon_vec = epsilon1
                    tune.epsilonprime = epsilonprime1
                    Base.unlock(lo)
                end
            end
        end
    end

    relist(model, theta0, block, true)
end
