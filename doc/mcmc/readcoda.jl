using MCPhylo

## Get the directory in which the example CODA files are saved
dir = dirname(@__FILE__)

## Read the MCMC sampler output from the files
c1 = readcoda(joinpath(dir, "line1.out"), joinpath(dir, "line1.ind"))
c2 = readcoda(joinpath(dir, "line2.out"), joinpath(dir, "line2.ind"))

## Concatenate the resulting chains
c = cat(c1, c2, dims=3)

## Compute summary statistics
describe(c)

#mark that we got to the end of the test file succesfully
@test true
