"""
    plot(c::AbstractChains, ptype::Vector{Symbol}=[:trace, :density];
       <keyword arguments>)::Array{Plots.Plot}

Function that takes a MCMC chain and creates various different plots (trace &
density by default).

# Arguments
- 'vars::Vector{String}=String[]': specifies the variables of the chain that
                                   are plotted. Trick: Instead of inputting
                                   [varname[1], varname[2], varname[3], ...]
                                   you can simply write [varname]. The function
                                   will understand that you want all variables
                                   with that name.
- 'filename::String=""': when given, the plots will be saved to a file
- 'fmt::Symbol=:pdf': Format of the output file
- 'fuse::Bool=false': Fuse all of the plots into one big plot, instead of
                      displaying each of the different plot types separately
- 'f_layout=nothing': Layout for the fused plot.
- 'fsize::Tuple(Number, Number)=(0,0)': Size of the fused plot.
- 'force::Bool=false': Force plotting of more than 20 variables.
- 'args...': This includes specific arguments for the different plot types
             , like the number of bins for the contourplot or if the barplots
             bars should be stacked or not. Check the specific plot functions
             to use these arguments.

            Plot attributes from the Plots.jl package can also be passed here:
            I.e. try passing background_color=:black for a black background on
            your plots. List of attributes here,
            https://docs.juliaplots.org/latest/generated/attributes_series/
            https://docs.juliaplots.org/latest/generated/attributes_plot/
            https://docs.juliaplots.org/latest/generated/attributes_subplot/
            https://docs.juliaplots.org/latest/generated/attributes_axis/
            though not all are supported:
            https://docs.juliaplots.org/latest/generated/supported/
"""
function plot(c::AbstractChains, ptype::Vector{Symbol}=[:trace, :density];
              vars::Vector{String}=String[], filename::String="",
              fmt::Symbol=:pdf, fuse::Bool=false, f_layout=nothing,
              fsize::Tuple{Number, Number}=(0, 0),
              force::Bool=false, args...
              )::Union{Vector{Plots.Plot}, Tuple{Vector{Plots.Plot}, Plots.Plot}}
  if !isempty(vars)
    indeces = check_vars(c.names, vars)
  else
    indeces = collect(1:length(c.names))
  end # if / else
  isempty(indeces) && throw(ArgumentError("Input Variables not in Chain object. Exiting function."))
  ilength = length(indeces)
  ilength > 20 && !force && throw(ArgumentError("Too many variables for plotting. Set force argument to 'true' to force plotting."))
  if :contour in ptype && ilength == 1
    filter!(e -> e ≠ :contour, ptype)
    if isempty(ptype)
      throw(ArgumentError("Drawing a contourplot requires at least 2 variables."))
    else
      @warn "Drawing a contourplot requires at least 2 variables. Only drawing other plots"
    end # if/else
  end # if
  n = length(ptype)
  p = Array{Plots.Plot}(undef, n)
  layout = :layout in keys(args) ? args[:layout] : ilength == 6 ? (2, 3) :
           ilength in [7, 8] ? (2, 4) : ilength == 9 ? (3, 3) :
           ilength == 10 ? (2, 5) : ilength in [11, 12] ? (3, 4) :
           ilength in [13, 15] ? (3, 5) : ilength in [14, 16] ? (4, 4) :
           ilength in [17, 18] ? (3, 6) : ilength in [19, 20] ? (4, 5) :
           (1, ilength)
  if ilength > 20 && !(:layout in keys(args))
    @warn "For plotting this many variables, it is suggested to pass a layout for better visibility.
         For example: If you have 25 variables to plot, try passing layout=(5,5) for a nice 5x5 grid"
  end # if
  for i in 1:n
    ptype[i] == :bar ? p[i] = bar_int(c, indeces; args...) :
    ptype[i] == :mixeddensity ? p[i] = mixeddensityplot(c, indeces; size=(ilength * 400 / layout[1], 250 * layout[1]),
                                layout=layout, args...) :
    p[i] = Plots.plot(c, indeces; ptype=ptype[i],
                      size=(ilength * 400 / layout[1], 250 * layout[1]),
                      layout=layout, args...)
    if !fuse
      display(p[i])
      if n != 1 && i != n
        println("Press ENTER to draw next plot")
        readline(stdin)
      end # if
    end # if
  end # for
  if fuse
    isnothing(f_layout) && (f_layout = (n, 1))
    fsize == (0, 0) && (fsize = (ilength * 400 / layout[1] * n / f_layout[1], f_layout[1] * 250 * layout[1]))
    allplots = Plots.plot(p..., layout=f_layout, size=fsize)
    display(allplots)
    filename != "" && check_filename(filename, fmt, allplots)
    return (p, allplots)
  end # if
  filename != "" && check_filename(filename, fmt, [p])
  return p
end # plot


"""
  check_vars(sim_names::Vector{AbstractString},
             vars::Vector{String})::Vector{Int64}

--- INTERNAL ---
Helper function that returns a list of indeces that correspond to specific
variables. Only those variables are then plotted in the following steps.
"""
function check_vars(sim_names::Vector{AbstractString}, vars::Vector{String})::Vector{Int64}
    names = []
    for var in vars
        if endswith(var, r"\[[0-9]+\]")
            for sim_name in sim_names
                if sim_name == var
                    push!(names, sim_name)
                end # if
            end # for
        else
            for sim_name in sim_names
                if sim_name == var || occursin(Regex(var * "\\[\\w+\\]"), sim_name)
                  #(occursin(sim_name,r"\[[0-9]*\]") && sim_name[1 : findfirst("[", sim_name)[1]-1] == var))
                    push!(names, sim_name)
                end # if
            end # for
        end # if / else
    end # for
    unique!(names)
    indeces = []
    for name in names
      index = findfirst(isequal(name), sim_names)
      push!(indeces, index)
    end # for
    sort!(indeces)
    return indeces
end # check_vars


"""
    check_filename(filename, fmt, plots)

--- INTERNAL ---
Helper function that checks if a user-given filename is valid, and saves the
created plots to that file.
"""
function check_filename(filename, fmt, plots)
  if !(fmt in [:pdf, :png, :ps, :svg])
      @warn "The given format is not supported. Use :pdf, :png, :ps, or :svg. Defaulting to pdf."
      fmt = :pdf
  end # if
  for (i, plot) in enumerate(plots)
      i == 1 ? savefig(plot[i], "$filename.$fmt") :
               savefig(plot[i], "$filename_$i.$fmt")
  end # for
end # check_filename


#################### Internal Plot Function ####################
@recipe function f(c::AbstractChains, indeces::Vector{Int64}; ptype=:autocor,
                   maxlag=round(Int, 10 * log10(length(c.range))), bins=100,
                   trim=(0.025, 0.975))
  grid --> :dash
  gridalpha --> 0.5
  legend --> false
  legendtitle --> "Chain"
  legendtitlefonthalign := :best
  top_margin := 2mm
  bottom_margin := 2mm
  left_margin := 15mm
  right_margin := 5mm

  arr = []
  ptype == :autocor ? push!(arr, Autocor(c, indeces, maxlag)) :
  ptype == :contour ? push!(arr, Contour(c, indeces, bins)) :
  ptype == :density ? push!(arr, Density(c, indeces, trim)) :
  ptype == :mean    ? push!(arr, Mean(c, indeces)) :
  ptype == :trace   ? push!(arr, Trace(c, indeces)) :
    throw(ArgumentError("unsupported plot type $ptype"))
  return Tuple(arr)
end # recipe

# structs that are used for the recipes
struct Autocor; c; indeces; maxlag; end
struct Contour; c; indeces; bins; end
struct Density; c; indeces; trim; end
struct Mean; c; indeces; end
struct Trace; c; indeces; end


"""
    f(acor::Autocor)

--- INTERNAL ---
Recipe for Autocor plots
"""
@recipe function f(acor::Autocor)
  xguide --> "Lag\n"
  yguide --> "Autocorrelation"
  xlims --> (0, +Inf)
  layout --> (1, length(acor.indeces))

  nrows, nvars, nchains = size(acor.c.value)
  lags = 0:acor.maxlag
  ac = autocor(acor.c, lags=collect(lags))
  x = repeat(collect(lags * step(acor.c)), outer=[nchains])
  for (index, i) in enumerate(acor.indeces)
    subplot := index
    y = vec(ac.value[i,:,:])
    ac_group = repeat(acor.c.chains, inner=[length(lags)])
    for (index, chain) in enumerate(acor.c.chains)
      idxs = findall(==(chain), ac_group)
      @series begin
        label --> string(index)
        title --> acor.c.names[i]
        seriestype := :line
        x[idxs], y[idxs]
      end # series
    end # for
  end # for
  primary := false
end # recipe


"""
    bar_int(c::AbstractChains, indeces::Vector{Int64}; args...)::Plots.Plot

--- INTERNAL ---
Helper function for creating barplots.
"""
function bar_int(c::AbstractChains, indeces::Vector{Int64}; args...)::Plots.Plot
  nrows, nvars, nchains = size(c.value)
  ilength = length(indeces)
  bar_plots = Array{Plots.Plot}(undef, ilength)
  for (index, i) in enumerate(indeces)
    S = unique(c.value[:, i, :])
    n = length(S)
    x = repeat(S, 1, nchains)
    y = zeros(n, nchains)
    for j in 1:nchains
      m = StatsBase.countmap(c.value[:, i, j])
      for k in 1:n
        if S[k] in keys(m)
          y[k, j] = m[S[k]] / nrows
        end # if
      end # for
    end # for
    bar_plots[index] = groupbar(vec(x), vec(y);
                                group=repeat(c.chains, inner=[n]),
                                title=c.names[i], args...)
  end # for
  p = Plots.plot(bar_plots..., layout=(1, ilength), size=(ilength * 400, 250))
  return p
end # function


"""
    f(cont::Contour)

--- INTERNAL ---
Recipe for contour plots
"""
@recipe function f(cont::Contour)
  layout --> (1, sum(collect(1:length(cont.indeces) - 1)))
  legend := false
  colorbar --> :best
  subplot_index = 0
  nrows, nvars, nchains = size(cont.c.value)
  offset = 1e4 * eps()
  n = nrows * nchains
  for (index, i) in enumerate(cont.indeces[1:end-1])
    X = cont.c.value[:, i, :]
    qx = range(minimum(X) - offset, stop=maximum(X) + offset, length=cont.bins + 1)
    mx = map(k -> mean([qx[k], qx[k + 1]]), 1:cont.bins)
    idx = Int[findfirst(k -> qx[k] <= x < qx[k + 1], 1:cont.bins) for x in X]
    for j in cont.indeces[index+1:end]
      subplot_index += 1
      Y = cont.c.value[:, j, :]
      qy = range(minimum(Y) - offset, stop=maximum(Y) + offset, length=cont.bins + 1)
      my = map(k -> mean([qy[k], qy[k + 1]]), 1:cont.bins)
      idy = Int[findfirst(k -> qy[k] <= y < qy[k + 1], 1:cont.bins) for y in Y]
      density = zeros(cont.bins, cont.bins)
      for k in 1:n
        density[idx[k], idy[k]] += 1.0 / n
      end
      @series begin
        subplot := subplot_index
        xguide --> cont.c.names[i]
        yguide --> cont.c.names[j]
        colorbar_title --> "Density"
        title --> cont.c.names[i]
        seriestype := :contour
        mx, my, density
      end # series
    end # for
  end # for
  primary := false
end # recipe


"""
    f(dens::Density)

--- INTERNAL ---
Recipe for density plots
"""
@recipe function f(dens::Density)
  xguide --> "Value\n"
  yguide --> "Density"
  ylims --> (0.0, +Inf)
  layout --> (1, length(dens.indeces))

  nrows, nvars, nchains = size(dens.c.value)
  for (index, i) in enumerate(dens.indeces)
    subplot := index
    val = Array{Vector{Float64}}(undef, nchains)
    dens_group = []
    for j in 1:nchains
      qs = quantile(dens.c.value[:, i, j], [dens.trim[1], dens.trim[2]])
      mask = [qs[1] .<= dens.c.value[:, i, j] .<= qs[2]]
      val[j] = dens.c.value[mask[1], i, j]
      dens_group = vcat(dens_group, repeat([j], inner=sum(mask[1])))
    end # for
    for (index, chain) in enumerate(dens.c.chains)
      idxs = findall(==(chain), dens_group)
      @series begin
        label --> string(index)
        title --> dens.c.names[i]
        seriestype := :density
        [val...;][idxs]
      end # series
    end # for
  end # for
  primary := false
end # recipe


"""
    f(mean::Mean)

--- INTERNAL ---
Recipe for mean plots
"""
@recipe function f(mean::Mean)
  xguide --> "Iteration\n"
  yguide --> "Mean"
  layout --> (1, length(mean.indeces))

  nrows, nvars, nchains = size(mean.c.value)
  val = cummean(mean.c.value)
  x = repeat(collect(mean.c.range), outer=[nchains])
  for (index, i) in enumerate(mean.indeces)
    subplot := index
    y = vec(val[:, i, :])
    mean_group = repeat(mean.c.chains, inner=[length(mean.c.range)])
    for (index, chain) in enumerate(mean.c.chains)
      idxs = findall(==(chain), mean_group)
      @series begin
        label --> string(index)
        title --> mean.c.names[i]
        seriestype := :line
        x[idxs], y[idxs]
      end # series
    end # for
  end # for
  primary := false
end # recipe


"""
    mixeddensityplot(c::AbstractChains,, indeces::Vector{Int64};
                     barbounds::Tuple{Real, Real}=(0, Inf), args...):Plots.Plot

--- INTERNAL ---
Helper function that creates a barplot for each discrete and a density for each
continuous variable.
"""
function mixeddensityplot(c::AbstractChains, indeces::Vector{Int64};
                          barbounds::Tuple{Real, Real}=(0, Inf), args...)
  plots = Array{Plots.Plot}(undef, length(indeces))
  ilength = length(indeces)
  discrete_temp = MCPhylo.indiscretesupport(c, barbounds)
  discrete = Bool[]
  for index in indeces
    try
      push!(discrete, discrete_temp[index])
    catch BoundsError
    end # try / catch
  end
  for i in 1:length(discrete)
    if discrete[i] == true
      plots[i] = bar_int(c, [indeces[i]]; args...)
    else
      plots[i] = Plots.plot(c, [indeces[i]]; ptype=:density, size=(400, 250), args..., layout=(1,1))
    end # if / else
  end # for
  p = Plots.plot(plots...; layout=args[:layout])
  return p
end # function


"""
    f(trace::Trace)

--- INTERNAL ---
Recipe for trace plots
"""
@recipe function f(trace::Trace)
  xguide --> "Iteration\n"
  yguide --> "Value"
  layout --> (1, length(trace.indeces))
  legendtitle --> "Chain"
  widen --> false

  nrows, nvars, nchains = size(trace.c.value)
  x = repeat(collect(trace.c.range), outer=[nchains])
  for (index, i) in enumerate(trace.indeces)
    subplot := index
    y = vec(trace.c.value[:, i, :])
    trace_group = repeat(trace.c.chains, inner=[length(trace.c.range)])
    for (index, chain) in enumerate(trace.c.chains)
      idxs = findall(==(chain), trace_group)
      @series begin
        label --> string(index)
        title --> trace.c.names[i]
        seriestype := :line
        x[idxs], y[idxs]
      end # series
    end # for
  end # for
  primary := false
end # recipe


#=
This is a slightly modified version of the GroupedBar recipe found in the
StatsPlots package:
  https://github.com/JuliaPlots/StatsPlots.jl/blob/master/src/bar.jl
=#
@userplot GroupBar
recipetype(::Val{:groupbar}, args...) = GroupBar(args)
Plots.group_as_matrix(g::GroupBar) = true
grouped_xy(x::AbstractVector, y::AbstractArray) = x, y
grouped_xy(y::AbstractArray) = 1:size(y,1), y
"""
    f(g::GroupBar; spacing=0)

--- INTERNAL ---
Recipe for a grouped bar plot.
"""
@recipe function f(g::GroupBar; spacing=0)
    xguide --> "Value\n"
    yguide --> "Density"
    grid --> :dash
    gridalpha --> 0.5
    legend --> false
    legendtitle --> "Chain"
    legendtitlefonthalign := :best
    margin --> 7mm
    x, y = grouped_xy(g.args...)

    nr, nc = size(y)
    isstack = pop!(plotattributes, :bar_position, :dodge) == :stack
    isylog = pop!(plotattributes, :yscale, :identity) ∈ (:log10, :log)
    ymax = maximum(isstack ? mapslices(sum, y, dims=2) : y)
    ylims --> (0.0, ymax)

    # extract xnums and set default bar width.
    # might need to set xticks as well
    xnums = if eltype(x) <: Number
        xdiff = length(x) > 1 ? mean(diff(x)) : 1
        bar_width --> 0.8 * xdiff
        x
    else
        bar_width --> 0.8
        ux = unique(x)
        xnums = (1:length(ux)) .- 0.5
        xticks --> (xnums, ux)
        xnums
    end
    @assert length(xnums) == nr

    # compute the x centers.  for dodge, make a matrix for each column
    x = if isstack
        x
    else
        bws = plotattributes[:bar_width] / nc
        bar_width := bws * clamp(1 - spacing, 0, 1)
        xmat = zeros(nr,nc)
        for r=1:nr
            bw = Plots._cycle(bws, r)
            farleft = xnums[r] - 0.5 * (bw * nc)
            for c=1:nc
                xmat[r,c] = farleft + 0.5bw + (c-1)*bw
            end
        end
        xmat
    end

    fill_bottom = if isylog
        if isfinite(the_ylims[1])
            min(minimum(y) / 100, the_ylims[1])
        else
            minimum(y) / 100
        end
    else
        0
    end
    # compute fillrange
    y, fr = isstack ? groupedbar_fillrange(y) : (y, get(plotattributes, :fillrange, [fill_bottom]))
    if isylog
        replace!( fr, 0 => fill_bottom )
    end
    fillrange := fr

    seriestype := :bar
    x, y
end


"""
    groupedbar_fillrange(y)

--- INTERNAL ---
Helper function for the groupbar plot
"""
function groupedbar_fillrange(y)
    nr, nc = size(y)
    # bar series fills from y[nr, nc] to fr[nr, nc], y .>= fr
    fr = zeros(nr, nc)
    y = copy(y)
    y[.!isfinite.(y)] .= 0
    for r = 1:nr
        y_neg = 0
        # upper & lower bounds for positive bar
        y_pos = sum([e for e in y[r, :] if e > 0])
        # division subtract towards 0
        for c = 1:nc
            el = y[r, c]
            if el >= 0
                y[r, c] = y_pos
                y_pos -= el
                fr[r, c] = y_pos
            else
                fr[r, c] = y_neg
                y_neg += el
                y[r, c] = y_neg
            end
        end
    end
    y, fr
end


"""
  plot_asdsf(c::AbstractChains; args...)::Plots.Plot

--- INTERNAL ---
Plot the ASDSF values of an Abstract Chains object.
"""
function plot_asdsf(c::AbstractChains; args...)::Plots.Plot
  size(c.stats, 1) == 0 && throw(ArgumentError("No ASDSF values have been computed on this simulation. Cannot plot them."))
  freq = c.sim_params.freq
  xvals = collect(c.sim_params.burnin + freq : freq :c.model.iter)
  p = Plots.plot(ASDSF_trace(c.stats, xvals); args...)
end # plot_asdsf

# struct for asdsf recipe
struct ASDSF_trace; yvals; xvals; end


"""
    f(vals::ASDSF_trace)

--- INTERNAL ---
Recipe for trace plots of ASDSF values
"""
@recipe function f(vals::ASDSF_trace)

  xlims --> (vals.xvals[1], vals.xvals[end])
  ylims --> (0, 1)
  grid --> :dash
  gridalpha --> 0.5
  legend --> false
  legendtitle --> "Tree Dimension"
  legendtitlefonthalign --> :best
  xguide --> "n-th Tree"
  yguide --> "ASDSF Values"
  margin --> 7mm

  for i in 1:size(vals.yvals, 2)
    @series begin
      label --> string(i)
      seriestype := :line
      vals.xvals, vals.yvals[:, i]
    end # series
  end # for
end # recipe
