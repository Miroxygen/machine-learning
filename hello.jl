using CSV
using DataFrames
using CategoricalArrays

iris = CSV.read("Iris/iris.csv", DataFrame)

#first(iris, 5)

grouped = groupby(iris, :species)

#Only works with csv data
function create_buckets(data, num_of_buckets)
  bucket_size = ceil(Int, nrow(data) / num_of_buckets)
  buckets = []
  for i in 1:bucket_size:nrow(data)
  end_index = min(i + bucket_size - 1, nrow(data))
  push!(buckets, data[i:end_index, :])
  end
  return buckets
end

buckets_by_species = Dict{String, Array{DataFrame, 1}}()

for species in grouped
  species_name = species.species[1]
  buckets_by_species[species_name] = create_buckets(species, 10)
end


iris_bucket = buckets_by_species[:"Iris-setosa"]
print(iris_bucket[1][!,:sepal_length])

