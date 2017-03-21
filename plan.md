# The Journal We're Targeting

We're going to target [SISC](https://www.siam.org/journals/sisc/redefined.php).

From our category's description:

"Papers in this category should concern the novel design and development of computational
methods and high-quality software, parallel algorithms, high-performance computing issues,
new architectures, data analysis, or visualization. The primary focus should be on
computational methods that have potentially large impact for an important class of
scientific or engineering problems."

# The Paper Will...

The paper will focus on the ways that dynamic compilation (i.e. JIT compilation), multiple
dispatch, SIMD vectorization, and strong metaprogramming facilities can be leveraged to
implement various techniques for native automatic differentiation.

The paper will present Julia as a popular language which has all of the above features, and
has already demonstrated an aptitude for the application domains in which automatic
differentiation is valuable. The reader should get the sense that Julia is the correct and
obvious choice for the implementation of this research.

The paper will present ForwardDiff + ReverseDiff as product(s) of this research, along with
benchmarks and downstream applications.

The paper will argue that native-language AD is paramount for composing general-purpose
language features with DSL-based frameworks (e.g. TensorFlow). Case-in-point: JuMP's
ability to easily register user-defined Julia functions. Selling this point correctly
would go a long way towards shutting down opponents who would dismiss the paper on the
point of "meh, this isn't TensorFlow."

# We Should Avoid...

We should avoid making the paper seem like a baseless advertisement for Julia.

We should avoid writing the paper as if it was a ForwardDiff/ReverseDiff manual.

We should avoid presenting techniques without also demonstrating the technique's use by way
of practical example.

We should avoid wasting time trying to "sell" AD in general. We can, of course, sell it a
little bit in the introduction (particularly bridging the conceptual gap between AD and
backpropagation), but we should assume our audience is mostly aware that AD exists and is
ubiquitously useful.

# Paper Structure

- Introduction
- Methodology
    - Language Features
        - multiple dispatch/operator-overloading design
        - JIT compilation
        - metaprogramming
    - Forward Mode
        - Dual Numbers
            - effectiveness of JIT compilation (show @code_native)
            - memory/cache efficiency
                - stack-allocated partial derivatives
                - chunk size
                - SIMD
            - genericism (nested duals, complex numbers, intervals)
                - calculate second-order subdifferential of a complex function?
                - Interval-Newton method
            - user-defined derivative API
    - Reverse Mode
        - Taping Utilities (Instructions/TrackedReals/TrackedArrays etc.)
            - non-allocating execution ("converts allocating code to pre-allocated code")
            - sparsity exploitation
            - ReverseDiff's tape is useable as a pure-Julia IR
                - dependency analysis is achievable by checking object_ids
                - scheduled parallel execution
                - interval constraint programming
                - converting Julia code to TensorFlow graphs
            - avoiding `getindex` taping
            - destructive assignment
            - fixed-parameter support
            - "nested" tapes
        - genericism
            - support for "special" array types
            - sparse array support
            - GPU-backed array support
        - Functor-based Directives (@forward, @skip, etc.)
            - handling branching in forward mode
        - constant elision via Julia's conversion system
        - user-defined derivative API
    - Perturbation Confusion
- Benchmarks
    - Softmax
    - Bayesian Linear Regression
    - Convolutional Neural Net
    - Multi-Layer Perceptron
    - Recurrent Neural Net
    - KL Divergence (Celeste)
    - Compare against:
        - AutoGrad.jl/Knet.jl
        - Theano
        - TensorFlow
        - ADOL-C
        - Torch
        - DiffSharp
- Featured Applications
    - Usage statistics?
    - JuMP
        - Coolest downstream examples?
    - Celeste
        - Constraint Transformations
        - KL Divergence
        - CG optimization (Hessian-vector products)
    - MOOSE
        - SIMD vectorization
- Conclusion

# Work Timeline

For ReverseDiff work that still needs to be done for the paper, see issues in the
ReverseDiff repository labeled "priority".

- Month 1
    - Work:
        - Celeste Work
        - ReverseDiff SoftMax Benchmark
        - ReverseDiff Bayesian Linear Regression Benchmark
        - All TensorFlow Benchmarks
        - Review/Correct Jarrett's Julia Benchmarks

- Month 2
    - Paper Sections:
        - Intro
        - Methodology: Language Features
    - Work:
        - [Support undifferentiated parameters for pre-recorded API](https://github.com/JuliaDiff/ReverseDiff.jl/issues/36)
        - [Support Base.LinearSlow array types](https://github.com/JuliaDiff/ReverseDiff.jl/issues/29)
        - Convolutional Neural Net Benchmark
        - Multi-Layer Perceptron Benchmark
        - Celeste Work
        - All Theano Benchmarks
        - All AutoGrad.jl/Knet.jl Benchmarks
        - Review/Correct Jarrett's ReverseDiff Benchmarks

- Month 3
    - Paper Sections:
        - Methodology: Forward Mode
        - Applications
    - Work:
        - [support for GPU-backed arrays](https://github.com/JuliaDiff/ReverseDiff.jl/issues/44)
        - [Exploiting sparsity in higher-order differentiation computations](https://github.com/JuliaDiff/ReverseDiff.jl/issues/41)
        - Recurrent Neural Net Benchmark
        - Celeste Work
        - All Torch Benchmarks
        - All DiffSharp Benchmarks
        - Review/Correct Jarrett's ReverseDiff Benchmarks

- Month 4
    - Paper Sections:
        - Methodology: Reverse Mode
        - Methodology: Perturbation Confusion
    - Work:
        - [Perturbation Confusion](https://github.com/JuliaDiff/ReverseDiff.jl/issues/45)
        - [Make it easy for users to inject derivative definitions](https://github.com/JuliaDiff/ReverseDiff.jl/issues/15)
        - All ADOL-C Benchmarks
        - Review/Correct Jarrett's ReverseDiff Benchmarks

- Month 5
    - Paper Sections:
        - Revisions
        - Benchmarks
    - Work:
        - "Definitive" benchmark runs on common hardware + performance data analysis
        - Paper Writing

# Materials/Interesting Links/Random Stuff

https://news.ycombinator.com/item?id=13428098
