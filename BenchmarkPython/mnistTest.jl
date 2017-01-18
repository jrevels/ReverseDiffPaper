include("mnist.jl")

# Construct the initial batch.
batch = Batch(TRAIN_IMAGES, TRAIN_LABELS, 1);

# Pre-allocate a reusable `Batch` for gradient storage.
# Note that this is essentially just a storage buffer;
# the initial values don't matter.
∇batch = Batch(TRAIN_IMAGES, TRAIN_LABELS, 1);
train_all!(∇batch, batch, TRAIN_IMAGES, TRAIN_LABELS, 4.2e-5, 1000)

@printf("The prediction accuracy is: %7.4f", test_model(batch))

function test_compare(batch::Batch, batch0::Batch)
    for i in 1:4
        println(getfield(batch, i) == getfield(batch0, i))
    end
end

function check_valid(batch::Batch)
    return (sum(batch.weights), sum(batch.bias), sum(batch.pixels), sum(batch.labels))
end

function check_border(y::AbstractArray)
    println((minus_log.(minimum(y)), minus_log.(maximum(y))))
end

function normal_col(A::AbstractArray)
    for (col,s) in enumerate(sum(A,1))
        s == 0 && continue # What does a "normalized" column with a sum of zero look like?
        A[:,col] = A[:,col]/s
    end
end

rate = 5e-4
∇model!(∇batch, batch)

for i in eachindex(batch.weights)
    batch.weights[i] -= rate * ∇batch.weights[i]
end
for i in eachindex(batch.bias)
    batch.bias[i] -= rate * ∇batch.bias[i]
end

check_valid(batch)

y = softmax((batch.weights * batch.pixels) .+ batch.bias)
check_border(y)

y = max(y, eps(eltype(y)))
y = min(y, 1 - eps(eltype(y)))
check_border(y)

function col_logsumexp(A::AbstractArray)
    _, ncol = size(A)
    colsum = zeros(ncol)
    for i in 1:ncol
        colsum[i] = StatsFuns.logsumexp(A[:, i])
    end
    return colsum
end
