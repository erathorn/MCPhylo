"""
    ParseCSV(filename::String, gap::AbstractString, miss::AbstractString, header::Bool=true)
This function parses a CSV file containing input for the MCMC compuation.
The file should follow the conventions used for MrBayes. For example:
    Swedish_0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,?,0,0,?,0,0
    Welsh_N_0,0,0,0,0,0,0,?,0,0,0,0,?,?,0,0,?,0,0,0,1,?,?,0
    Sardinian_N_0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,?,0,0,0,0,0
 Set input for "header" to false if no header is present in the file.
Returns ntax, nchar as Integer values, gap as a String; df is returned as a DataFrame, and stores language names and data.
* `filename` : Name of CSV file to parse.
* `header` : Boolean value. "true" denotes there is a header to be skipped; input "false" if the file does not contain a header.
"""
function ParseCSV(filename::String, gap::AbstractString, miss::AbstractString, header::Bool=true)
    open(filename, "r") do file
        global content = readlines(file)
    end # do
    # remove the header if necessary
    header ||  popfirst!(content)
    langs,df = create_csvdf(content)
    dimensions = size(df)
    ntax = dimensions[1]
    nchar = dimensions[2]
    symbols = get_alphabet(df, gap, miss)
    return ntax, nchar, gap, miss, symbols, df, langs
end # function ParseCSV


"""
    function create_csvdf(filecontnt::Array{String}, separator::AbstractString=",")::Tuple{Array{SubString}, Array{Char}}
This function parses a CSV file and returns its content as a DataFrame.
NOTE: May later on be replaced by the respective DataFrames function.
"""
function create_csvdf(filecontent::Array{String}, separator::AbstractString=",")::Tuple{Array{String}, Array{Char}}
    language = String[]
    data = Char[]
    nlangs = 0
    ntax = 0
    #df's created after editing the filecontent into separate language/data objects
    for line in filecontent
        splitline = split(line, separator)
        lang, raw = splitline[1], splitline[2:end]
        push!(language,lang)
        ntax = length(raw)
        data = append!(data,[i[1] for i in raw])
        nlangs += 1
    end #for
    #df = DataFrame(Language = language,Data = data)
    return language, permutedims(reshape(data, (ntax,nlangs)))
end # function create_csvdf
