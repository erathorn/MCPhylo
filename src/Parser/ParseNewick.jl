"""
    load_newick(filename::String)

This function loads all tree representations from a file
"""

function load_newick(filename::String)

    open(filename, "r") do file
        global content = readlines(file)
    end
    content
end


"""
    is_valid_newick_string(newick::String)

This function checks if the given string is valid: is the brackets number matches and if the string ends with ";"
"""

function is_valid_newick_string(newick::String) #TODO: possibly not necessary; could be done as part of recursion, MAYBE
    # Step one: does the stripped string ends with ';'
    if endswith(strip(newick),";")
        # Step two: check for the equal amount of brackets
        bracket_level = 0
        for letter in newick # => char is a type in Julia
            if letter == '('
                bracket_level += 1
            elseif letter == ')'
                    bracket_level -= 1
            end # elseif
        end # for
        if bracket_level != 0
            return false
        end # if
    else # same level as the endswith statement
        return false
    end # else
    return true
end



"""
    function parsing_newick_string(newick::String)::FNode

Parse a single newick string.
"""
function parsing_newick_string(newick::String)::FNode
    if !is_valid_newick_string(newick)
        throw("$newick is not correctly formatted!")
    end # if
    tree = nwk_parser(string(newick))
    set_binary!(tree)
    number_nodes!(tree)
    tree
end

"""
    ParseNewick(filename::String)::Array{GeneralNode, 1}

This function takes a filename as a String, and returns an array of trees(represented as Node objects).
The file should solely consist of newick tree representations, separated by line.
The function checks for proper newick formatting, and will return an error if the file is
incorrectly formatted.

Returns an Array of Nodes; each Node is the root of the tree represented by a newick string in the file.

* `filename` : name of file containing newick strings to parse.
"""
function ParseNewick(filename::String)::Array{GeneralNode, 1}
    list_of_trees = load_newick(filename)
    list_of_newicks = GeneralNode[]
    for content in list_of_trees
        if content == ""
            continue
        end # if
        tree = parsing_newick_string(content)
        push!(list_of_newicks, tree)
    end # for
    list_of_newicks
end


function nwk_parser(nwk_string::S)::FNode where S <: AbstractString
    if  nwk_string[end] == ';' #no need for semicolon
        nwk_string = chop(nwk_string)
    end #if
    parts = split(nwk_string, ")")
    if length(parts) == 1
        desc, node_name = [], parts[1]
    else
        if !(startswith(parts[1], '('))
            throw(ArgumentError("unmatched braces $(parts[1])"))
        else    
            desc = parse_sibblings(join(parts[1:end-1],")")[2:end])
            node_name = parts[end]
        end
    end
    node = parse_name_length(node_name)
    if length(desc) > 0
        for child in desc
            add_child!(node, child)
        end
    end
    node
end 


function parse_sibblings(nwk_string::S)::Array{FNode} where S <: AbstractString
    bracket_level = 0
    current = Char[]
    children = FNode[]
    # trick to remove special-case of trailing chars
    for c in (nwk_string * ",")
        if c == ',' && bracket_level == 0
            n = nwk_parser(join(current, ""))
            push!(children, n)
            current = Char[]
        else
            if c == '('
                bracket_level += 1
            elseif c == ')'
                bracket_level -= 1
            end
            push!(current, c)
        end
    end
    children
end

function parse_name_length(newick::S)::FNode where S<:AbstractString
    nnam = "no_name"
    ninc = 1.0
    if length(newick) > 0
        if occursin(':',newick)
            name, len = split(strip(newick),':')
            if name == ""
                name = "no_name"
            end # if
            if len == ""
                len = 1.0
            end # if
            if occursin(';',len)
                len = chop(len)
            end # if
            nnam = string(name)
            ninc = parse(Float64, len)
        else
            nnam = string(newick)
        end
    end #if-else
    node = Node(nnam)
    node.inc_length = ninc
    return node
end # function
