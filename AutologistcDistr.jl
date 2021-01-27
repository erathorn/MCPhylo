mutable struct AutologisticDistr <: DiscreteMatrixDistribution
    ling_concordant::Array{Float64,3}
	ov_ling_concordant::Vector{Float64}
    ling_params::Vector{Float64}
	lingmat::Array{Int64,2}
    spatial_concordant::Array{Float64,3}
	ov_spatial_concordant::Vector{Float64}
    spatial_params::Vector{Float64}
	spmat::Array{Int64,2}
	ov_universality::Array{Float64,2}
    universality_params::Array{Float64,2}
	nvals::Vector{Int64} # for each feature, the number of possible values
	nlangs::Int64
	nfeat::Int64
end # mutable struct

Base.size(d::AutologisticDistr) = (d.nfeat, d.nlangs)

function logpdf(d::AutologisticDistr, X::Array{N,2}) where N <: Real
	res = 0
	maxval = maximum(d.nvals)
	ling_cond = ov_concordant_sums(X, d.lingmat)
	sp_cond = ov_concordant_sums(X, d.spmat)
	#res = @. ling_cond[f] * d.ling_params[f] + sp_cond[f] * d.spatial_params[f]
	for f in 1:d.nfeat
		res += d.ov_ling_concordant[f] * d.ling_params[f] +
		d.ov_spatial_concordant[f] * d.spatial_params[f]
		res += ling_cond[f] * d.ling_params[f] + sp_cond[f] * d.spatial_params[f]
		for k in 1:maxval
			res += d.ov_universality[f,k] * d.universality_params[f,k]
		end
	end
	return res
end

"""
P = exp(vertical_params[i] * vertical_sums[l,i,k] + ... + uni_params[i,k])
"""

function logcond(d::AutologisticDistr, X::Array{N, 2}, l::Int64, f::Int64) where N <: Real
	# l -> language_index
	# f -> index of feature value
	nv = d.nvals[f]
	probs = Vector{Float64}(undef, nv)
	for k in 1:nv
		p = d.spatial_params[f] * d.spatial_concordant[f,l,k] +
			d.ling_params[f] * d.ling_concordant[f,l,k] +
			d.universality_params[f,k]
		probs[k] = p
	end
	return probs
end
