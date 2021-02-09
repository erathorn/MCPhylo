mutable struct AutologisticDistr <: DiscreteMatrixDistribution
    cond_ling::Array{Float64,3}
#	ov_ling_concordant::Vector{Float64}
    ling_params::Vector{Float64}
	lingmat::Array{Float64,2}
    cond_space::Array{Float64,3}
#	ov_spatial_concordant::Vector{Float64}
    spatial_params::Vector{Float64}
	spmat::Array{Float64,2}
	ov_universality::Array{Float64,2}
    universality_params::Array{Float64,2}
	nvals::Vector{Int64} # for each feature, the number of possible values
	nlangs::Int64
	nfeat::Int64
end

Base.size(d::AutologisticDistr) = (d.nfeat, d.nlangs)

function logpdf(d::AutologisticDistr, X::Array{N,2}) where N <: Real
	res = 0
	maxval = maximum(d.nvals)
	ling_cond = ov_concordant_sums(X, d.lingmat)
	sp_cond = ov_concordant_sums(X, d.spmat)
	#res = @. ling_cond[f] * d.ling_params[f] + sp_cond[f] * d.spatial_params[f]
	#println("ling: ", d.ling_params)
	for f in 1:d.nfeat
		res += ling_cond[f] * d.ling_params[f] + sp_cond[f] * d.spatial_params[f]
		#println("---")
		for k in 1:maxval
			uni_cond = count(x -> x == k, X[f,:])
			#println(uni_cond, " ", d.universality_params[f,k])
			res += uni_cond * d.universality_params[f,k]
		end
		#println("---")
	end
	#println("res $res")
	return res
end

"""
P = exp(vertical_params[i] * vertical_sums[l,i,k] + ... + uni_params[i,k])
"""

function logcond(d::AutologisticDistr, X::Array{N, 2}, l::Int64, f::Int64) where N <: Real
	nv = d.nvals[f]
	probs = Vector{Float64}(undef, nv)
	spacesums = cond_concordant_sums(X, d.spmat)
	lingsums = cond_concordant_sums(X, d.lingmat)

	#println("language $l, feature $f")
	for k in 1:nv
		p = d.spatial_params[f] * spacesums[f,l,k] +
			d.ling_params[f] * lingsums[f,l,k] +
			d.universality_params[f,k]
		probs[k] = p
	end
	#println("raw $probs")
	return softmax(probs)
end
