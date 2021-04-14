function read_cldf(path)
   df = CSV.read(path, DataFrame)
   df = df[:, [:ISO_CODE, :CONCEPT, :COGNATE_CLASS]]
   dropmissing!(df)
   langs = [i for i in unique(df[!,:ISO_CODE]) if !ismissing(i)]
   concepts = unique(df[!,:CONCEPT])
   out = zeros(length(langs), length(concepts))
   for (ind1, concept) in enumerate(concepts)
       df1 = @linq df |> where(:CONCEPT .== concept) |> select(:COGNATE_CLASS, :ISO_CODE)
       classes = String[]

       for (ind2, lang) in enumerate(langs)
          #findfirst(x -> x == df1[(df1.ISO_CODE .== lang),:COGNATE_CLASS][1], classes)
          entry = df1[(df1.ISO_CODE .== lang),:COGNATE_CLASS]#[1]
      
          assignval = -1
          if length(entry) > 1
             cc = entry[1]
             if !(cc in classes)
                push!(classes, cc)
             end
             assignval = findfirst(x -> x == cc, classes)
          end
          out[ind2, ind1] = assignval
       end
    end
    out, langs
end
