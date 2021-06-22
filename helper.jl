function read_cldf(path)
    df = CSV.read(path, DataFrame)
    df = df[:, [:ISO_CODE, :CONCEPT, :COGNATE_CLASS]]
    dropmissing!(df)
    langs = [i for i in unique(df[!,:ISO_CODE]) if !ismissing(i)]
    concepts = unique(df[!,:CONCEPT])
    out = zeros(length(langs), length(concepts))
    for (ind1, concept) in enumerate(concepts)
        df1 = @linq df |> where(:CONCEPT .== concept) |> select(:COGNATE_CLASS, :ISO_CODE)
        classes = String[]

        for (ind2, lang) in enumerate(langs)
          # findfirst(x -> x == df1[(df1.ISO_CODE .== lang),:COGNATE_CLASS][1], classes)
            entry = df1[(df1.ISO_CODE .== lang),:COGNATE_CLASS]# [1]
      
            assignval = -1
            if length(entry) > 1
                cc = entry[1]
                if !(cc in classes)
                    push!(classes, cc)
                end
                assignval = findfirst(x -> x == cc, classes)
            end
            out[ind2, ind1] = assignval
        end
    end
    out, langs
end



function max_psrf(sim::ModelChains)::Float64
    bi = 1 + size(sim)[1] ÷ 2
    gd = gelmandiag(sim[bi:end,:,:])
    gd_values = gd.value
    indices = isnan.(gd_values)
    gd_values[indices] .= -1
    psrf = maximum(gd_values[:,1])
    psrf
end

function read_binary(path)
    _,_, g,m,syms,df,langs = MCPhylo.ParseNexus(path)
    of = zeros(size(df))
    for (i,s) in enumerate(df)
        if string(s) == g || string(s) == m
            of[i] = -1
        else
            of[i] = findfirst(isequal(string(s)), syms) - 1
        end
    end
    of, langs
end

function other_matrix(root::G, n::Int)::Matrix{Float64} where G <:GeneralNode
    leaves::Vector{G} = get_leaves(root)
    out = zeros(length(leaves),n)
    for (i,leave) in enumerate(leaves)
        mother = leave
        while !mother.root
            out[i,mother.num] = 1.0
            mother = MCPhylo.get_mother(mother)
        end
    end
    out
end

function covariance_fun(blv::AbstractArray{Float64}, om::AbstractArray{Float64})
    dm = Diagonal(sqrt.(blv))
    Φ = om * dm
    Φ * transpose(Φ)
end