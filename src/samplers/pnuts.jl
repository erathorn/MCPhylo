#################### Phylogenetic No-U-Turn Sampler ####################

#################### Types and Constructors ####################

mutable struct PNUTSTune <: SamplerTune
    logfgrad::Union{Function,Missing}
    adapt::Bool
    alpha::Float64
    epsilon::Float64
    epsilonbar::Float64
    gamma::Float64
    Hbar_acc::Float64
    kappa::Float64
    m::Int
    mu::Float64
    nalpha::Int
    t0::Float64
    delta::Float64
    target::Float64
    moves::Vector{Int}
    tree_depth::Int
    nniprime::Float64
    targetNNI::Int
    tree_depth_trace::Vector{Int}


    PNUTSTune() = new()

    function PNUTSTune(
        x::Vector{T},
        epsilon::Float64,
        logfgrad::Union{Function,Missing};
        target::Real = 0.6,
        tree_depth::Int = 10,
        targetNNI::Int = 5,
        delta::Float64 = 0.003,
        jitter::Float64 = 0.0,
    ) where {T<:GeneralNode}

        new(
            logfgrad,
            false,
            0.0,
            epsilon,
            1.0,
            0.05,
            0.0,
            0.75,
            0,
            NaN,
            0,
            10.0,
            delta,
            target,
            Int[],
            tree_depth,
            0,
            targetNNI,
            Int[],
        )
    end
end

PNUTSTune(
    x::Vector{T},
    logfgrad::Function,
    ::NullFunction,
    delta::Float64 = 0.003,
    target::Real = 0.6;
    args...,
) where {T<:GeneralNode} =
    PNUTSTune(x, nutsepsilon(x[1], logfgrad, delta, target), logfgrad; args...)

PNUTSTune(
    x::Vector{T},
    logfgrad::Function,
    delta::Float64,
    target::Real;
    args...,
) where {T<:GeneralNode} =
    PNUTSTune(x, nutsepsilon(x[1], logfgrad, delta, target), logfgrad; args...)

PNUTSTune(x::Vector; epsilon::Real, args...) = PNUTSTune(x, epsilon, missing, args...)

const PNUTSVariate = SamplerVariate{PNUTSTune}


#################### Sampler Constructor ####################


"""
    PNUTS(params::ElementOrVector{Symbol}; args...)

Construct a `Sampler` object for PNUTS sampling. The Parameter is assumed to be
a tree.

Returns a `Sampler{PNUTSTune}` type object.

* params: stochastic node to be updated with the sampler.

* args...: additional keyword arguments to be passed to the PNUTSVariate constructor.
"""
function PNUTS(params::ElementOrVector{Symbol}; args...)
    samplerfx = function (model::Model, block::Integer)
        block = SamplingBlock(model, block, true)

        f = let block = block
            (x, sz, ll, gr) -> mlogpdfgrad!(block, x, sz, ll, gr)
        end
        v = SamplerVariate(block, f, NullFunction(); args...)

        sample!(v::PNUTSVariate, f, adapt = model.iter <= model.burnin)

        relist(block, v)
    end
    Sampler(params, samplerfx, PNUTSTune())
end


#################### Sampling Functions ####################

function mlogpdfgrad!(
    block::SamplingBlock,
    x::FNode,
    sz::Int64,
    ll::Bool = false,
    gr::Bool = false,
)::Tuple{Float64,Vector{Float64}}
    grad = Vector{Float64}(undef, sz)
    lp = zero(Float64)

    if gr
        lp, grad = gradlogpdf!(block, x)::Tuple{Float64,Vector{Float64}}
    elseif ll
        lp = logpdf!(block, x)::Float64
    end
    lp, grad
end
sample!(v::PNUTSVariate; args...) = sample!(v, v.tune.logfgrad; args...)

function sample!(v::PNUTSVariate, logfgrad::Function; adapt::Bool = false)

    tune = v.tune
    setadapt!(v, adapt)
    if tune.adapt
        tune.m += 1
        tune.nniprime = 0

        nuts_sub!(v, tune.epsilon, logfgrad)
        adaptstat = tune.alpha / tune.nalpha
        adaptstat = adaptstat > 1 ? 1 : adaptstat
        HT = tune.target - adaptstat
        
        HT2 = tune.targetNNI - tune.nniprime / tune.nalpha
        HT -= abs_adapter(HT2)
        HT /= 2
        p = 1.0 / (tune.m + tune.t0)
        tune.Hbar_acc = (1.0 - p) * tune.Hbar_acc + p * HT
        
        tune.epsilon = exp(tune.mu - sqrt(tune.m) * tune.Hbar_acc / tune.gamma)
                
        
        p = tune.m^-tune.kappa
        tune.epsilonbar = exp(p * log(tune.epsilon) + (1.0 - p) * log(tune.epsilonbar))
    else
        if (tune.m > 0)
            tune.epsilon = tune.epsilonbar
        end

        nuts_sub!(v, tune.epsilon, logfgrad)
    end
    v
end

@inline function abs_adapter(x::R)::Float64 where R <:Real
    x / (1+abs(x))
end

function setadapt!(v::PNUTSVariate, adapt::Bool)
    tune = v.tune
    if adapt && !tune.adapt
        tune.m = 0
        tune.mu = log(10.0 * tune.epsilon)
    end
    tune.adapt = adapt
    v
end



function nuts_sub!(v::PNUTSVariate, epsilon::Float64, logfgrad::Function)
    x = deepcopy(v.value[1])
    nl = size(x)[1] - 1
    delta = v.tune.delta
    r = randn(nl)
    g = zeros(nl)
    #blv = get_branchlength_vector(x)
    #set_branchlength_vector!(x, molifier.(blv, delta))
    #logf, grad = logfgrad(x, nl, true, true)
    
    x, r, logf, grad, nni = refraction(x, r, g, epsilon, logfgrad, delta, nl)
    #@show nni
    lu = log(rand())
    logp0 = logf - 0.5 * dot(r)
    logu0 = logp0 + lu#log(rand())
    rminus = rplus = r
    gradminus = gradplus = grad

    xminus = xplus = x
    nni = 0
    j = 0
    n = 1
    s = true
    v.tune.nniprime = 0
    while s && j < v.tune.tree_depth
        pm = 2 * (rand() > 0.5) - 1

        if pm == -1

            xminus,
            rminus,
            gradminus,
            _,
            _,
            _,
            xprime,
            nprime,
            sprime,
            alpha,
            nalpha,
            nni1,
            lpp,
            nniprime = buildtree(
                xminus,
                rminus,
                gradminus,
                pm,
                j,
                epsilon,
                logfgrad,
                logp0,
                logu0,
                delta,
                nl,
                lu
            )

        else

            _,
            _,
            _,
            xplus,
            rplus,
            gradplus,
            xprime,
            nprime,
            sprime,
            alpha,
            nalpha,
            nni1,
            lpp,
            nniprime = buildtree(
                xplus,
                rplus,
                gradplus,
                pm,
                j,
                epsilon,
                logfgrad,
                logp0,
                logu0,
                delta,
                nl,
                lu
            )

        end#if pm

        v.tune.alpha, v.tune.nalpha = alpha, nalpha
        v.tune.nniprime = nniprime
        
        if !sprime
            break
        end
        # sprime is true so checking is not necessary
        if rand() < nprime / n
            v.value[1] = xprime
        end
        
        n += nprime
        nni += nni1


        j += 1
        s = nouturn(
            xminus,
            xplus,
            rminus,
            rplus,
            gradminus,
            gradplus,
            epsilon,
            logfgrad,
            delta,
            nl,
            j,
        )

    end

    push!(v.tune.moves, nni)
    push!(v.tune.tree_depth_trace, j)
    v
end


function refraction(
    v::T,
    r::Vector{Float64},
    grad::Vector{Float64},
    epsilon::Float64,
    logfgrad::Function,
    delta::Float64,
    sz::Int64,
) where {T<:FNode}

    v1 = deepcopy(v)

    blenvec = get_branchlength_vector(v1)
    fac = scale_fac.(blenvec, delta)
    
    @. r += (epsilon * 0.5) * grad * fac
    tmpB = @. blenvec + (epsilon * r)

    nni = 0

    if minimum(tmpB) <= 0
        v1, tmpB, r, nni =
            ref_NNI(v1, tmpB, r, abs(epsilon), blenvec, delta, logfgrad, sz)
    end

    blenvec = molifier.(tmpB, delta)

    set_branchlength_vector!(v1, blenvec)

    logf, grad = logfgrad(v1, sz, true, true)

    fac = scale_fac.(blenvec, delta)
    
    @. r += (epsilon * 0.5) * grad * fac

    return v1, r, logf, grad, nni
end


function ref_NNI(
    v::T,
    tmpB::Vector{Float64},
    r::Vector{Float64},
    epsilon::Float64,
    blv::Vector{Float64},
    delta::Float64,
    logfgrad::Function,
    sz::Int64,
) where {T<:FNode}

    intext = internal_external(v)
    t = 0.0
    nni = 0
    #epsilon = abs(epsilon)
    while minimum(tmpB) <= 0.0
        timelist = tmpB ./ abs.(r)
        ref_index = argmin(timelist)

        temp = epsilon-t+timelist[ref_index]
        blv = @. blv + (temp * r)
        
        r[ref_index] *= -1.0

        if intext[ref_index] == 1

            blv1 = molifier.(blv, delta)            
            set_branchlength_vector!(v, blv1)

            U_before_nni, _ = logfgrad(v, sz, true, false) # still with molified branch length
            
            v_copy = deepcopy(v)
            tmp_NNI_made = NNI!(v_copy, ref_index)
            
            if tmp_NNI_made != 0

                U_after_nni, _ = logfgrad(v_copy, sz, true, false)
                
                delta_U::Float64 = 2.0 * (U_before_nni-U_after_nni)
                my_v::Float64 = r[ref_index]^2
                if my_v > delta_U
                    nni += tmp_NNI_made
                    r[ref_index] = sqrt(my_v - delta_U)
                    v = v_copy
                end # if my_v
            end #if NNI
        end #non leave
        
        t = epsilon + timelist[ref_index]    
        tmpB = @. blv + (epsilon-t) * r

    end #while

    v, tmpB, r, nni
end



function buildtree(
    x::T,
    r::Vector{Float64},
    grad::Vector{Float64},
    pm::Int64,
    j::Integer,
    epsilon::Float64,
    logfgrad::Function,
    logp0::Real,
    logu0::Real,
    delta::Float64,
    sz::Int64,
    lu::Float64
) where {T<:FNode}


    if j == 0

        xprime, rprime, logfprime, gradprime, nni =
            refraction(x, r, grad, pm*epsilon, logfgrad, delta, sz)

        logpprime = logfprime - 0.5 * dot(rprime)

        nprime = lu + (logp0 - logpprime) < 0#Int(logu0 < logpprime)

        sprime = lu + (logp0 - logpprime) < 1000.0
        #logu0 < logpprime + 1000.0
        xminus = xplus = xprime
        rminus = rplus = rprime
        gradminus = gradplus = gradprime
        alphaprime = min(1.0, exp(logpprime - logp0))
        nniprime = nni
        nalphaprime = 1

    else
        xminus,
        rminus,
        gradminus,
        xplus,
        rplus,
        gradplus,
        xprime,
        nprime,
        sprime,
        alphaprime,
        nalphaprime,
        nni,
        logpprime,
        nniprime =
            buildtree(x, r, grad, pm, j - 1, epsilon, logfgrad, logp0, logu0, delta, sz,lu)
        if sprime

            if pm == -1

                xminus,
                rminus,
                gradminus,
                _,
                _,
                _,
                xprime2,
                nprime2,
                sprime2,
                alphaprime2,
                nalphaprime2,
                nni,
                logpprime,
                nniprime2 = buildtree(
                    xminus,
                    rminus,
                    gradminus,
                    pm,
                    j - 1,
                    epsilon,
                    logfgrad,
                    logp0,
                    logu0,
                    delta,
                    sz,
                    lu
                )
            else
                _,
                _,
                _,
                xplus,
                rplus,
                gradplus,
                xprime2,
                nprime2,
                sprime2,
                alphaprime2,
                nalphaprime2,
                nni,
                logpprime,
                nniprime2 = buildtree(
                    xplus,
                    rplus,
                    gradplus,
                    pm,
                    j - 1,
                    epsilon,
                    logfgrad,
                    logp0,
                    logu0,
                    delta,
                    sz,
                    lu
                )
            end # if pm

            if rand() < nprime2 / (nprime + nprime2)
                xprime = xprime2
            end
            nprime += nprime2
            sprime =
                sprime2 && nouturn(
                    xminus,
                    xplus,
                    rminus,
                    rplus,
                    gradminus,
                    gradplus,
                    epsilon,
                    logfgrad,
                    delta,
                    sz,
                    j,
                )
            alphaprime += alphaprime2
            nalphaprime += nalphaprime2
            nniprime += nniprime2
        end #if sprime
    end #if j

    xminus,
    rminus,
    gradminus,
    xplus,
    rplus,
    gradplus,
    xprime,
    nprime,
    sprime,
    alphaprime,
    nalphaprime,
    nni,
    logpprime,
    nniprime
end


function nouturn(
    xminus::T,
    xplus::T,
    rminus::Vector{Float64},
    rplus::Vector{Float64},
    gradminus::Vector{Float64},
    gradplus::Vector{Float64},
    epsilon::Float64,
    logfgrad::Function,
    delta::Float64,
    sz::Int64,
    j::Int64,
) where {T<:FNode}
    curr_l, curr_h = BHV_bounds(xminus, xplus)

    xminus_bar, _, _, _, _ = refraction(
        deepcopy(xminus),
        deepcopy(rminus),
        gradminus,
        -epsilon,
        logfgrad,
        delta,
        sz,
    )
    xplus_bar, _, _, _, _ = refraction(
        deepcopy(xplus),
        deepcopy(rplus),
        gradplus,
        epsilon,
        logfgrad,
        delta,
        sz,
    )

    curr_t_l, curr_t_h = BHV_bounds(xminus_bar, xplus_bar)
    return curr_h < curr_t_l
end




#################### Auxilliary Functions ####################

function nutsepsilon(x::FNode, logfgrad::Function, delta::Float64, target::Float64)

    x0 = deepcopy(x)
    n = size(x)[1] - 1

    # molifier is necessary!
    blv = get_branchlength_vector(x)
    set_branchlength_vector!(x, molifier.(blv, delta))
    
    logf0, gr = logfgrad(x, n, true, true)
    
    r0 = randn(n)
    epsilon = 1.0
    _, rprime, logfprime, _, _ = refraction(x0, r0, gr, epsilon, logfgrad, delta, n)
    Hp = logfprime - 0.5 * dot(rprime)
    H0 = logf0 - 0.5 * dot(r0)
    prob = Hp - H0
    direction = prob > target ? 1 : -1

    while direction == 1 ? prob > target : prob < target
        epsilon = direction == 1 ? 2 * epsilon : 0.5 * epsilon
        _, rprime, logfprime, _, _ = refraction(x0, r0, gr, epsilon, logfgrad, delta, n)
        Hp = logfprime - 0.5 * dot(rprime)
        prob = Hp - H0
    end
    epsilon
end

@inline function scale_fac(x::T, delta::T) where {T<:Float64}
    x < delta ? x/delta : 1.0
end

@inline function molifier(x::Float64, delta::Float64)::Float64
    x >= delta ? x : (x^2 + delta^2) / (2.0 * delta)
end # function
