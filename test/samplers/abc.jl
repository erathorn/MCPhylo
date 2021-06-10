using MCPhylo

@testset "abc exponential" begin
    gs_p = randn()^2
    
    data = Dict(
        :y => rand(Exponential(gs_p), 1000)
    )
    
    mod = Model(
        y=Stochastic(1, (l) -> Exponential(l), false),
        l=Stochastic(() -> LogNormal(), true)
    )
    
    inits = [
        Dict{Symbol,Any}(
          :y => data[:y],
          :l => rand(LogNormal()),
            )
        for i in 1:3
    ]

    f = x -> [mean(x), std(x)]
    scheme = [ABC(:l, 1.0, f, 0.1)]

    setsamplers!(mod, scheme)

    sim = mcmc(mod, data, inits, 10000, burnin=250, thin=5, chains=3)
    
    if maximum(gelmandiag(sim).value[:,2,:]) > 1.1
        sim = mcmc(sim, 5000)
    end
    
    m = mean(sim.value[:,1,:])
    sd = std(sim.value[:,1,:])

    @test m - sd <= gs_p <= m + sd
end


@testset "abc Normal" begin
    gs_m = randn()
    gs_s = randn()^2
    data = Dict(
        :y => rand(Normal(gs_m, gs_s), 1000)
    )
    
    mod = Model(
        y=Stochastic(1, (m, s) -> Normal(m, s), false),
        m=Stochastic(() -> Normal(), true),
        s=Stochastic(() -> Exponential(), true)
    )
    
    inits = [
        Dict{Symbol,Any}(
          :y => data[:y],
          :m => rand(),
          :s => rand(),
        )
        for i in 1:3
    ]
    
    f = x -> [mean(x), std(x)]
    scheme = [ABC(:m, 1.0, f, 0.1),
              ABC(:s, 1.0, f, 0.1)]

    setsamplers!(mod, scheme)

    sim = mcmc(mod, data, inits, 10000, burnin=250, thin=5, chains=3)
    
    if maximum(gelmandiag(sim).value[:,2,:]) > 1.1
        sim = mcmc(sim, 5000)
    end
    
    m = mean(sim.value[:,1,:])
    sd = std(sim.value[:,1,:])

    @test m - sd <= gs_s <= m + sd

    m = mean(sim.value[:,2,:])
    sd = std(sim.value[:,2,:])
    
    @test m - sd <= gs_m <= m + sd
end