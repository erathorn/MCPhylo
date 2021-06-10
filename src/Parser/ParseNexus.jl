

"""
    ParseNexus(filename::String)
This function parses a NEXUS file which stores the input for the MCMC compuation.
The file should follow the conventions used for MrBayes.
Returns ntax, nchar, gap, and missing_representation values; returns Dataframe storing language names and data.
* `filename` : NEXUS file to be parsed.
"""
function ParseNexus(filename::String)
    open(filename, "r") do file
        global content = readlines(file)
    end # do

    if lowercase(content[1]) != "#nexus"
        throw("$filename is not a Nexus file!")
    end # if

    while true
        line = popfirst!(content)
        if lowercase(line) == "begin data;"
            # here comes the important information, so break the loop
            break
        end #if
    end # while
    # get meta info
    ntax, nchar, gap, missing_representation, symbols = extract_meta_info(content)
    langs, df = create_nexusdf(content)

    out_symbols = symbols == "NOSYMBOLS" ? get_alphabet(df, gap, missing_representation) : [string(s) for s in symbols]

    return ntax, nchar, gap, missing_representation, out_symbols, df, langs
end # function ParseNexus


function get_alphabet(df::Array, gap::String, missing_representation::String)
    alphabet = Set{String}()
    for row in df
        for entry in row
            entry = string(entry)
            if !(entry == gap) && !(entry == missing_representation)
                push!(alphabet, entry)
            end
        end
    end
    sort([i for i in alphabet])
end

"""
    extract_meta_info(content::Array{String})
This function extracts some meta information from the content of the nexus file.
It "eats-up" the stack.
Returns values derived from metadata (ntax, nchar, gap, missing_representation). Used in ParseNexus().
* `content` : Array of Strings; Strings are read from NEXUS file in ParseNexus().
"""
function extract_meta_info(content::Array{String})
    lct::Int64 = 0
    ntax::Int64 = 0
    nchar::Int64 = 0
    gap::String = "-"
    symbols::String= "NOSYMBOLS"
    missing_representation::String = "?"
    while true
        line = popfirst!(content)
        if lowercase(line)=="matrix"
            # meta info is done, data begins now
            break
        else
            splitted_line = split(line)
            for entry in splitted_line
                info = split(entry, "=")
                info = lowercase.(info)
                if length(info) != 1
                    choped = info[2]
                    if endswith(choped, ";")
                        choped = chop(choped)
                    end #if
                    k_word = info[1]
                    if k_word == "ntax"
                        ntax = parse(Int64, choped)
                    elseif k_word == "nchar"
                        nchar = parse(Int64, choped)
                    elseif k_word == "gap"
                        gap = choped
                    elseif k_word == "missing"
                        missing_representation = choped
                    elseif k_word == "symbols"
                        symbols = strip(String(choped), '\"')
                    else
                        @warn "Keyword $k_word not understood, will be ignored"
                    end # if
                end # if
            end # for
        end # if
    end # while
    return ntax, nchar, gap, missing_representation, symbols
end # function extract_meta_info

"""
    create_nexusdf(filecontent::Array{String})::Tuple{Array{SubString}, Array{Char}}
This function creates a DataFrame of the actual data. Used in ParseNexus().
Returns DataFrame of language names and data derived from NEXUS file.
* `filecontent` : Array of Strings; Strings are read from NEXUS file in ParseNexus().
"""
function create_nexusdf(filecontent::Array{String})::Tuple{Array{String}, Array{Char}}
    #df = DataFrame(Language=String[], Data=String[])
    languages = String[]
    data = Char[]
    nlangs = 0
    ntax = 0
    while true
        line = popfirst!(filecontent)
        if line == ""
            continue
        elseif line == ";"
            break
        else
            lang, raw = split(line)
            push!(languages,lang)
            ntax = length(raw)
            data = append!(data,raw)
            nlangs += 1

        end # if
    end # while
    return languages, permutedims(reshape(data, (ntax,nlangs)))
end # function create_nexusdf
