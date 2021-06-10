"""
    to_file(model::ModelChains, outpath::AbstractString)
This function writes the results of the MCMC runs into files. The destination of
the files is specified using `outpath`. It will create a files for each chain. A
`params_x.log` file storing each parameter sample. In this case `x` specifies the
index of the chain. The file is compatible with MCMC analysis tools like `Tracer`
(http://tree.bio.ed.ac.uk/software/tracer/). If in addition trees are sampled,
they are stored in newick format in a file called `trees_x.nwk`, where `x` again
specifies the index of the respective chain.
"""
function to_file(model::ModelChains, outpath::AbstractString)
    for run in 1:size(model.value,3)
        thin = model.range.step
        if isassigned(model.trees, 1)
            to_file(model.value[:, :, run], model.trees[:,:,run], outpath, string(run), thin, model.names, model.tree_names)
        else
            to_file(model.value[:, :, run], outpath, string(run), thin, model.names)
        end
    end

end

function to_file(df::Array, outpath::String, run::String, thin::Int64, names::Array)
    io = open(string(outpath, "params_"*run*".log"),"w")
    write(io, "it\t")
    join(io, names, "\t")
    write(io, "\n")
    for (ind,row) in enumerate(eachrow(df))
        write(io, "$(ind*thin)\t")
        join(io, row, "\t")
        write(io, "\n")
    end
    close(io)
end


function to_file(df::Array, tdf::Array, outpath::AbstractString,
                run::AbstractString, thin::Int64, names::Array, tree_names::Array{A}) where A <: AbstractString

    to_file(df, outpath, run, thin, names)
    for (ind, n_name) in enumerate(tree_names)
        io = open(string(outpath, "trees_"*n_name*"_"*run*".nwk"), "w")
        for x = 1:length(tdf[:,ind])
            write(io, tdf[x,ind])
            write(io, "\n")
        end
        close(io)
    end
end
