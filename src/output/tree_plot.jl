#=
    The tree plotting functions are adapted from the tree plotting functionality
    of the EcoJulia Phylo package.

    Further documentation and the original functions can be found here:
        https://github.com/EcoJulia/Phylo.jl
=#
"""
    f(root::T; <keyword arguments>) where T<:GerneralNode

Recipe that handles plotting of MCPhylo Trees. Takes the root node as input.

# Arguments
- 'treetype::Symbol=:dendrogram' :  specifies how the tree is visualized.
                                    Options :fan and :dendrogram
- 'showtips::Bool=:true' : controls if leaf labels are shown or not
- 'tipfont::Union{Tuple{Int64}, Tuple{Int64,Symbol}}=(7,)' :
                specify the font and font colour of the annotations
- 'marker_group=nothing' : currently not supported
- 'line_group=nothing' : currently not supported
"""
@recipe function f(root::T; treetype=:dendrogram, marker_group=nothing,
                   line_group=nothing, showtips=true, tipfont=(7,)
                   ) where T<:GeneralNode

    linecolor --> :black
    grid --> false
    framestyle --> :none
    legend --> false
    colorbar --> true
    size --> (1000, 1000)

    lz = get(plotattributes, :line_z, nothing)
    mz = get(plotattributes, :marker_z, nothing)
    isnothing(lz) || (line_z := _handlez(lz, root))
    isnothing(mz) || (marker_z := _handlez(mz, root))
    mg = _handlez(marker_group, root)
    lg = _handlez(line_group, root)

    d, h, n = _findxy(root)
    adj = 0.03maximum(values(d))
    tipannotations = map(x->(d[x] + adj, h[x], x.name), get_leaves(root))
    x, y = Float64[], Float64[]
    for node ∈ pre_order(root)
        if !node.root
            push!(x, d[node.mother], d[node.mother], d[node], NaN)
            push!(y, h[node.mother], h[node], h[node], NaN)
        end
    end

    marker_x, marker_y = _handlemarkers(plotattributes, mg, root, d, h)

    if treetype == :dendrogram
        Dendrogram(x, y, tipannotations, marker_x, marker_y, showtips, tipfont, mg, lg)
    elseif treetype == :fan
        Fan(x, y, tipannotations, marker_x, marker_y, showtips, tipfont, mg, lg)
    else
        throw(ArgumentError("Unsupported `treetype`; valid values are `:dendrogram` or `:fan`"))
    end
end


struct Dendrogram; x; y; tipannotations; marker_x; marker_y; showtips; tipfont; marker_group; line_group; end
struct Fan; x; y; tipannotations; marker_x; marker_y; showtips; tipfont; marker_group; line_group; end


@recipe function f(dend::Dendrogram)
    ex = extrema(filter(isfinite, dend.x))
    xlims --> (ex[1] - 0.05 * ex[2], ex[2] * 1.15)

    sa = get(plotattributes, :series_annotations, nothing)
    dend.showtips && (annotations := map(x -> (x[1], x[2], (x[3], :left, dend.tipfont...)), dend.tipannotations))
    @series begin
        seriestype := :path
        markersize := 0
        markershape := :none
        series_annotations := nothing
        label --> ""
    #    primary := false

        lc = _extend(get(plotattributes, :linecolor, nothing), dend.x)
        lc !== nothing && (linecolor := lc)
        la = _extend(get(plotattributes, :linealpha, nothing), dend.x)
        la !== nothing && (linealpha := la)
        lz = _extend(get(plotattributes, :line_z, nothing), dend.x)
        lz !== nothing && (line_z := lz)

        dend.x, dend.y
    end
    if !isempty(dend.marker_x) || sa !== nothing
        if isnothing(dend.marker_group)
            @series begin
                seriestype := :scatter
                sa !== nothing && (series_annotations := sa)
                label --> ""
                dend.marker_x, dend.marker_y
            end
        else
            groups = sort(unique(dend.marker_group))
            for group in groups
                idxs = findall(==(group), dend.marker_group)
                @series begin
                    seriestype := :scatter
                    sa !== nothing && (series_annotations := sa[idxs])
                    label --> string(group)
                    dend.marker_x[idxs], dend.marker_y[idxs]
                end
            end
        end
    end
    primary = false
    label = ""
    nothing
end


@recipe function f(fan::Fan)
    adjust(y) = 2pi*y / (length(fan.tipannotations) + 1)

    sa = get(plotattributes, :series_annotations, nothing)
    aspect_ratio := 1
    mx = maximum(filter(isfinite, fan.x))
    if fan.showtips
        xlims --> (1.5 .* (-mx, mx))
        ylims --> (1.5 .* (-mx, mx))
        annotations := map(x -> (_tocirc(x[1], adjust(x[2]))..., (x[3], :left,
            rad2deg(adjust(x[2])), fan.tipfont...)), fan.tipannotations)
    end
    @series begin
        seriestype := :path
        markersize := 0
        markershape := :none
        series_annotations := nothing
        label := ""

        x, y = _circle_transform_segments(fan.x, adjust(fan.y))
        lc = _extend(get(plotattributes, :linecolor, nothing), x)
        lc !== nothing && (linecolor := lc)
        la = _extend(get(plotattributes, :linealpha, nothing), x)
        la !== nothing && (linealpha := la)
        lz = _extend(get(plotattributes, :line_z, nothing), x)
        lz !== nothing && (line_z := lz)
        x, y
    end
    if !isempty(fan.marker_x) || sa !== nothing
        if isnothing(fan.marker_group)
            @series begin
                seriestype := :scatter
                sa !== nothing && (series_annotations := sa)
                label --> ""
                _xcirc.(adjust(fan.marker_y), fan.marker_x), _ycirc.(adjust(fan.marker_y), fan.marker_x)
            end
        else
            groups = sort(unique(fan.marker_group))
            for group in groups
                idxs = findall(==(group), fan.marker_group)
                @series begin
                    seriestype := :scatter
                    sa !== nothing && (series_annotations := sa[idxs])
                    label --> string(group)
                    _xcirc.(adjust(fan.marker_y[idxs]), fan.marker_x[idxs]), _ycirc.(adjust(fan.marker_y[idxs]), fan.marker_x[idxs])
                end
            end
        end
    end
    nothing
end


_handlez(x, root) = x
# need to add this back if i.e. gradiently coloured plots are needed
# _handlez(x::Union{String, Symbol}, tree) = getnodedata.(tree, traversal(tree, preorder), x)
_mylength(x) = 1
_mylength(x::AbstractVector) = length(x)


function _handlemarkers(plotattributes, marker_group, root, d, h)
    marker_x, marker_y = Float64[], Float64[]
    markerfields = filter(x->occursin(r"marker", String(x)), keys(plotattributes))
    isempty(markerfields) && isnothing(marker_group) && return(marker_x, marker_y)
    maxlengthfields = isempty(markerfields) ? 1 : maximum([_mylength(plotattributes[k]) for k in markerfields])
    maxlengthgroup = isnothing(marker_group) ? 1 : length(marker_group)
    maxlength = max(maxlengthfields, maxlengthgroup)
    f = maxlength ∈ (1, count(x-> root.nchild != 0, pre_order(root))) ? filter(x -> x.nchild != 0, pre_order(root)) : pre_order(root)
    append!(marker_x, getindex.(Ref(d), f))
    append!(marker_y, getindex.(Ref(h), f))
    marker_x, marker_y
end


function _extend(tmp, x)
    tmp isa AbstractVector && abs(length(tmp) - count(isnan, x)) < 2 || return nothing
    ret = similar(x, eltype(tmp))
    j = 1 + length(tmp) - count(isnan, x)
    for i in eachindex(x)
        ret[i] = tmp[j]
        isnan(x[i]) && (j += 1)
    end
    return ret
end


function _findxy(root::T)::Tuple{Dict{T, Float64}, Dict{T, Float64}, Vector{String}} where T<:GeneralNode

    # two convenience recursive functions using captured variables
    function findheights!(node::T) where T<:GeneralNode
        if !in(node, keys(height))
            for child in node.children
                findheights!(child)
            end
        end
        if node.nchild != 0
            ch_heights = [height[child] for child in node.children]
            height[node] = (maximum(ch_heights) + minimum(ch_heights)) / 2.
        end
    end

    function finddepths!(node::T, parentdepth::Float64 = 0.0) where T<:GeneralNode
        mydepth = parentdepth
        push!(names, node.name)
        mydepth += node.inc_length
        depth[node] = mydepth
        for child in node.children
            finddepths!(child, mydepth)
        end
    end

    # root_name = root.name / num
    height = Dict(tip => float(i) for (i, tip) in enumerate(x for x in get_leaves(root)))
    nnodes = length(post_order(root))
    sizehint!(height, nnodes)
    findheights!(root)

    depth = Dict{T, Float64}()
    ### Kann man hier node.num stattdessen verwenden, oder Labels hängen später
    ### von dem name Array ab
    names = String[]
    sizehint!(depth, nnodes)
    sizehint!(names, nnodes)
    finddepths!(root)

    depth, height, names
end


function _p_circ(start_θ, end_θ, r=1)
    steps = range(start_θ, stop=end_θ, length = 1+ceil(Int, 60abs(end_θ - start_θ)))
    retx = Array{Float64}(undef, length(steps))
    rety = similar(retx)
    for u in eachindex(steps)
        retx[u] = _xcirc(steps[u], r)
        rety[u] = _ycirc(steps[u], r)
    end
    retx, rety
end


_xcirc(x, r) = r*cos(x)
_ycirc(y, r) = r*sin(y)
_tocirc(x, y) = _xcirc(y, x), _ycirc(y, x)


function _circle_transform_segments(xs, ys)
    retx, rety = Float64[], Float64[]
    function _transform_seg(_x, _y)
        tmpx, tmpy = _p_circ(_y[1], _y[2], _x[1])
        append!(retx, tmpx)
        append!(rety, tmpy)
        push!(retx, _xcirc(_y[3], _x[3]), NaN)
        push!(rety, _ycirc(_y[3], _x[3]), NaN)
    end
    i = 1
    while !(i === nothing) && i < length(xs)
        j = findnext(isnan, xs, i) - 1
        _transform_seg(view(xs,i:j), view(ys, i:j))
        i = j + 2
    end
    retx, rety
end
