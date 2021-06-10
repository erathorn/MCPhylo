
abstract type AbstractNode end

"""
    Node
This data type holds the basic Node structure. The type T is used to specify the type of the data
stored in the node.
* If `nchild` is `0` the Node is a leaf node.
* If `root` is `False` the Node is a child of another node.
* `inc_length` specifies the length of the incomming branch.
* `binary` specifies the path from the root to the Node. `1` and `0` represent left and right turns respectively.
"""
mutable struct GeneralNode{R<:Real, I<:Integer} <: AbstractNode
    name::String
    mother::Union{GeneralNode{R,I}, Missing}
    children::Vector{GeneralNode{R,I}}
    nchild::I
    root::Bool
    inc_length::R
    binary::String
    num::I
    height::R
    IntExtMap::Vector{I}
    blv::Vector{R}
    stats::Dict{String, Float64}
end # struct Node

const FNode = GeneralNode{Float64, Int64}
"""
    function Node()::FNode
This function will initialize an empty node.
"""
function Node()::FNode
        FNode("no_name", missing, FNode[], 0, true, 1.0, "0",1, 1.0,Int64[], Float64[], Dict{String, Float64}())
end


function Node(name::String)::FNode
        FNode(name, missing, FNode[], 0, true, 1.0, "0", 1, 1.0, Int64[], Float64[], Dict{String, Float64}())
end


#################### Base functionality ####################

Base.:(==)(x::T, y::T) where T<:GeneralNode = x.num == y.num
Base.size(x::T) where T<:GeneralNode = size(post_order(x))
Base.length(x::T) where T<:GeneralNode = x.nchild

function Base.summary(io::IO, d::N) where N <: GeneralNode
    summary(io, d.name)
end


function Base.show(io::IO, d::N) where N <: GeneralNode
    print(io, "Tree with root:\n")
    show(io, d.name)
    print(io, "\nLength:\n")
    show(io, "text/plain", tree_length(d))
    print(io, "\nHeight:\n")
    show(io, "text/plain", tree_height(d))
    
end

function showall(io::IO, d::N) where N <: GeneralNode
  show(io, d)
  print(io, "\nNode:\n")
  show(io, "text/plain", d.name)
  print(io, "\n#children:\n")
  show(io, d.nchild)
  print(io, "\nbinary:\n")
  show(io, d.binary)
end
