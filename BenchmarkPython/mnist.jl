using MNIST, ReverseDiff
using StatsFuns

# const BATCH_SIZE = 100
# const IMAGE_SIZE = 784
# const CLASS_COUNT = 10

##################
# data wrangling #
##################

# loading MNIST data #
#--------------------#

function load_mnist(data)
    images, label_indices = data
    labels = zeros(CLASS_COUNT, length(label_indices))
    for i in eachindex(label_indices)
        labels[Int(label_indices[i]) + 1, i] = 1.0
    end
    return images, labels
end

# const TRAIN_IMAGES, TRAIN_LABELS = load_mnist(MNIST.traindata())
# const TEST_IMAGES, TEST_LABELS = load_mnist(MNIST.testdata())

# loading batches #
#-----------------#

# immutable Batch{W,B,P,L}
#     weights::W
#     bias::B
#     pixels::P
#     labels::L
# end

function Batch(images, labels, i)
    weights = zeros(CLASS_COUNT, IMAGE_SIZE)
    bias = zeros(CLASS_COUNT)
    range = i:(i + BATCH_SIZE - 1)
    return Batch(weights, bias, images[:, range], labels[:, range])
end

function load_batch!(batch, images, labels, i)
    offset = i - 1
    for batch_col in 1:BATCH_SIZE
        data_col = batch_col + offset
        for k in 1:size(images, 1)
            batch.pixels[k, batch_col] = images[k, data_col]
        end
        for k in 1:size(labels, 1)
            batch.labels[k, batch_col] = labels[k, data_col]
        end
    end
    return batch
end

####################
# model definition #
####################

# Here, we make a model out of simple `softmax` and `cross_entropy` functions. This could be
# improved by implementing something like Tensorflow's `softmax_cross_entropy_with_logits`,
# but my main goal is to show off ReverseDiff rather than implement the best possible model.

# Also note that our input's orientation is transposed compared to example implementations
# presented by row-major frameworks like Tensorflow. Julia is column-major, so I've set up
# the `Batch` code (see above) such that each column of `pixels` is an image and
# `size(pixels, 2) == BATCH_SIZE`.

# objective definitions #
#-----------------------#

# Here we use `@forward` to tell ReverseDiff to differentiate this scalar function in
# forward-mode. This allows us to call `minus_log.(y)` instead of `-(log.(y))`. By defining
# our own "fused" `minus_log` kernel using `@forward`, the operation `minus_log.(y)` becomes
# a single array instruction in the tape (instead of two separate ones,  as is the case with
# `-(log.(y))`), buying us a slight performance gain. Note that we didn't *need* to do this;
# it's simply a good place to show off the `@forward` feature.
ReverseDiff.@forward minus_log(x::Real) = -log(x)

function cross_entropy(y′, y)
    # add a floor and ceiling to the input data
    y = max(y, eps(eltype(y)))
    y = min(y, 1 - eps(eltype(y)))
    if minimum(y) <= 0
        println("error!")
    end
    entropy = mean(-sum(y′ .* log(y) + (1 - y′) .* log(1 - y), 1) ./ CLASS_COUNT)
    #entropy = mean(sum(y′ .* (minus_log.(y)), 1))
    return entropy
end

function col_normalize(A::AbstractArray)
    for (col,s) in enumerate(sum(A,1))
        s == 0 && continue # What does a "normalized" column with a sum of zero look like?
        A[:,col] = A[:,col]/s
    end
    A
end

function softmax(x)
    #x = x - sum(exp_x, 1))
    max_x = max(0, maximum(x))
    rebase_x = x - max_x

    #return exp(rebase_x - StatsFuns.logsumexp(rebase_x))
    return col_normalize(exp(rebase_x - StatsFuns.logsumexp(rebase_x)))
    #return exp(rebase_x) ./ sum(exp(rebase_x), 1)
    #rebase_x - logsumexp(logsumexp(rebase_x) - max_x)
    #x = (exp_x = exp.(x); exp_x ./ sum(exp_x, 1))
end

function model(weights, bias, pixels, labels)
    y = (weights * pixels) .+ bias
    #y = mapslices(StatsFuns.softmax, y, 1)
    y = softmax(y)
    return cross_entropy(labels, y)
end

# gradient definitions #
#----------------------#

# generate the gradient function `∇model!(output, input)` from `model`
∇model! = begin
    # grab a sample batch as our seed data
    batch = Batch(TRAIN_IMAGES, TRAIN_LABELS, 1)

    # `compile_gradient` takes array arguments - it doesn't know anything about `Batch`
    input = (batch.weights, batch.bias, batch.pixels, batch.labels)

    # generate the gradient function (which is then returned from this `begin` block and
    # bound to `∇model!`). See the `ReverseDiff.compile_gradient` docs for details.
    ReverseDiff.compile_gradient(model, input)
end

# Add convenience method to `∇model!` that translates `Batch` args to `Tuple` args. Since
# `∇model!` is a binding to a function type generated by ReverseDiff, we can overload
# the type's calling behavior just like any other Julia type. For details, see
# http://docs.julialang.org/en/release-0.5/manual/methods/#function-like-objects.
function (::typeof(∇model!))(output::Batch, input::Batch)
    output_tuple = (output.weights, output.bias, output.pixels, output.labels)
    input_tuple = (input.weights, input.bias, input.pixels, input.labels)
    return ∇model!(output_tuple, input_tuple)
end

############
# training #
############

function train_batch!(∇batch::Batch, batch::Batch, rate, iters)
    for _ in 1:iters
        ∇model!(∇batch, batch)
        for i in eachindex(batch.weights)
            batch.weights[i] -= rate * ∇batch.weights[i]
        end
        for i in eachindex(batch.bias)
            batch.bias[i] -= rate * ∇batch.bias[i]
        end
    end
end
# rate = 2.81048e-4
function train_all!(∇batch::Batch, batch::Batch, images, labels, rate = 1e-4, iters = 5)
    batch_count = floor(Int, size(images, 2) / BATCH_SIZE)
    for i in 1:batch_count
        load_batch!(batch, images, labels, i)
        train_batch!(∇batch, batch, rate, iters)
    end
    return ∇batch
end

#######################
# running the example #
#######################

#=

# load the code
include(joinpath(Pkg.dir("ReverseDiff"), "examples", "mnist.jl"))

# Construct the initial batch.
batch = Batch(TRAIN_IMAGES, TRAIN_LABELS, 1);

# Pre-allocate a reusable `Batch` for gradient storage.
# Note that this is essentially just a storage buffer;
# the initial values don't matter.
∇batch = Batch(TRAIN_IMAGES, TRAIN_LABELS, 1);

train_all!(∇batch, batch, TRAIN_IMAGES, TRAIN_LABELS)

=#




function accuracy(result::Array, label::Array)
    score = 0
    _, ncol = size(label)
    for i in 1:ncol
        if indmax(result[:, i]) == indmax(label[:, i])
            score += 1
        end
    end
    score / ncol
end


function test_model(batch::Batch)
    # load all test images
    #tbatch = Batch(TEST_IMAGES, TEST_LABELS, 1);
    # tlabels = batch.weights * tbatch.pixels
    tlabels = batch.weights * TEST_IMAGES .+ batch.bias
    # compute accuracy and return
    return @printf("%7.4f", accuracy(tlabels, TEST_LABELS))
end