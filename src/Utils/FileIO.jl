
function to_file(model::ModelChains, outpath::AbstractString, thin::Int64)

    for run in 1:size(model.value,3)

        df = DataFrame(model.value[:,:,run])
        rename!(df, Symbol.(model.names))
        if isassigned(model.trees, 1)

            tdf = DataFrame(model.trees[:,:,run])
            to_file(df, tdf, outpath, string(run), thin)
        else
            to_file(df, outpath, string(run), thin)
        end
    end

end

function to_file(df::DataFrame, outpath::AbstractString, run::AbstractString, thin::Int64)
    insertcols!(df,1, it=1:nrow(df))
    df[!, 1] .*= thin
    CSV.write(string(outpath, "params_"*run*".log"), df, writeheader=true, delim="\t")

end

function to_file(df::DataFrame, tdf::DataFrame, outpath::AbstractString, run::AbstractString, thin::Int64)

    insertcols!(df,1, it=1:nrow(df))
    df[!, 1] .*= thin
    CSV.write(string(outpath, "params_"*run*".log"), df, writeheader=true, delim="\t")
    io = open(string(outpath, "mytrees_"*run*".nwk"), "w")
    for x = 1:length(tdf[:,1])
        write(io, tdf[x,:][1])
        write(io, "\n")
    end
    close(io)
end


function drop_samples(model::ModelChains, thin::Int64)
    trees = model.trees
    range = model.range
    value = model.value[1:thin:size(model.value,1),:,:]
    if isassigned(trees, 1)
        trees =model.trees[1:thin:size(model.trees,1),:,:]
    end
    mrange = range[1]:step(range)*thin:range[end]
    ModelChains(value, mrange, model.names, model.chains, model.model, trees, model.moves)
end

function cut_samples(model::ModelChains, remove::Int64)
    trees = model.trees
    range = model.range
    interval = step(range)
    acct_remove = remove/interval
    #@assert isinteger(acct_remove)
    println(acct_remove)
    acct_remove = Int(acct_remove)
    value = model.value[1:size(model.value,1)-acct_remove,:,:]
    if isassigned(trees, 1)
        trees =model.trees[1:size(model.trees,1)-acct_remove,:,:]
    end
    mrange = range[1]:step(range):range[end]-remove
    ModelChains(value, mrange, model.names, model.chains, model.model, trees, model.moves)
end
