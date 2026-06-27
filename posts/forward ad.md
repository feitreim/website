# Forward Mode Automatic Differentiation
### date: 29 Nov, 2025

## Introduction
Automatic Differentiation(AD) is a core algorithm in modern machine learning, implementations like Torch's [autograd](https://pytorch.org/blog/overview-of-pytorch-autograd-engine/) and JAX's [jax.grad](https://docs.jax.dev/en/latest/automatic-differentiation.html) dominate the space. autograd and jax.grad both primarily compute derivatives using back propagation (backprop), a case of backwards mode AD.  Forward mode automatic differentiation is effectively the opposite of backwards mode, we calculate the derivatives 
$$
\nabla\mathcal{L}=
\begin{bmatrix} 
\frac{\partial\mathcal{L}}{\partial w^L}\\ 
\vdots \\
\frac{\partial\mathcal{L}}{\partial w^1}\\ 
\end{bmatrix}
$$
from beginning to end, as we perform the forward pass, instead of from the end to the beginning, traversing back up the graph.

## Dual Numbers
Very simply, you can think of the dual numbers as being similar to the complex numbers in this sense: the imaginary numbers take the form of $a+bi, i=\sqrt{-1}$ and the dual numbers are defined $a+b\epsilon$ where $\epsilon^2 = 0, \epsilon \neq 0$.

Now, both $i=\sqrt{-1} \implies i^2=-1$ and $\epsilon^2 = 0, \epsilon \neq 0$ are both somewhat odd sounding statements at first, they have really interesting properties that make them nice to study. Complex numbers are an algebraically closed field, meaning all polynomials (excl. order 0) with complex coefficients have complex roots, a property the Reals do not have. 

We will get the special property of the Duals soon, but first we can just look at performing basic operations with them, $\epsilon^2 = 0$ is a really nice property it turns out. Addition
$$
(a+b\epsilon)+(c+d\epsilon) =(a+c)+(b+d)\epsilon
$$is easy but multiplication (think FOIL) will get more interesting.

$$
\begin{align*}
(a+b\epsilon)(c+d\epsilon) &= ac+ad\epsilon+bc\epsilon+bd\epsilon^2\\
&=ac+(ad+bc)\epsilon
\end{align*}
$$
or division (first step is to multiply the numerator and denominator by the conjugate of the denominator).
$$
\begin{align*}
\frac{a+b\epsilon}{c+d\epsilon} &= 
\frac{(a+b\epsilon)(c-d\epsilon)}{(c+d\epsilon)(c-d\epsilon)}\\
&=\frac{ac+bc\epsilon-ad\epsilon-ad\epsilon^2}{c^2+cd\epsilon-cd\epsilon-d^2\epsilon^2}\\
&=\frac{ac+bc\epsilon-ad\epsilon}{c^2}\\
&=\frac{a}{c}+\frac{b}{c}\epsilon-\frac{ad}{c^2}\epsilon\\
&=\frac{a}{c}+\left(\frac{b}{c}-\frac{ad}{c^2}\right)\epsilon
\end{align*}
$$
Okay, the dual numbers still provide a pretty simple system of arithmetic, but what does this get us? What is the special property? 

## It's the Derivative!
Okay lets look at at example, consider $f(x) = 5x$, if we evaluate this at $x=2$ we see $f(2)=5(2)=10$, pretty simple, we can also find the derivative $f'(x) = 5$, so $f'(2)=5$. Now lets extend that to the Duals, where instead of $x=2$, we will use $x=2+1\epsilon$
$$
\begin{align*}
f_\mathbb{D}(2+1\epsilon)&=(5+0\epsilon)(2+1\epsilon)\\
&=(5*2)+(5*1+0*2)\epsilon \\
&=10+5\epsilon
\end{align*}
$$
We can see that the real component is just the result of $f(x=2)$, while the dual component is $f'(x=2)$. When we apply a dual number to function that is ordinarily only real valued, it is easy to imagine simply lifting any real number $c$ to $c+0\epsilon$, this will ensure that we ended up with the derivative that we desire. 

We can now show that this property holds for any differentiable function $f$. Remember first the definition of the Taylor series of $f$ around some point $c$
$$
f(x) = \sum_{n=0}^\infty\frac{f^{(n)}(c)}{n!}(x-c)^n
$$
(reminder: $f^{(n)}$ is the nth derivative of $f$). Now we can think about evaluating $f(a+b\epsilon)$, but before we do that we must choose our starting point $c$, I propose that we let $c=a$, the real part of our dual number, this gives us:
$$
\begin{align*}
f(a+b\epsilon) &= \sum_{n=0}^\infty\frac{f^{(n)}(a)}{n!}(a+b\epsilon-a)^n \\
&=\sum_{n=0}^\infty\frac{f^{(n)}(a)}{n!}(b\epsilon)^n \\
&=\frac{f(a)}{0!}(b\epsilon)^0+\frac{f^{(1)}(a)}{1!}(b\epsilon)^1+\frac{f^{(2)}(a)}{2!}(b\epsilon)^2+\cdots \\
&=f(a)+f^{(1)}(a)(b\epsilon)+0+\cdots\\
&=f(a)+bf^{(1)}(a)\epsilon
\end{align*}
$$
Where every term for $n>1$ just goes to 0 by $\epsilon^2=0$. 

## Implementation
Implementing the duals in your language of choice is quite easy, just create a new type with a real part and a dual part, define all your operations, and you're good to go! (I choose rust).
```rust
struct Dual {
    real: f32,
    dual: f32,
}

...

impl Mul for &Dual {
    type Output = Dual;
    fn mul(self, rhs: Self) -> Self::Output {
        Dual {
            real: self.real * rhs.real,
            dual: rhs.real * self.dual + self.real * rhs.dual,
        }
    }
}

...
```
Thats just an example of one of the arithmetic functions you will need to define of course, but it illustrates the simplicity. Here is an example evaluating the function $f(x) = 5x$
```rust
   #[test]
    fn test_mul_grad() {
        let a = Node::with_grad(5.0);
        let b = Node::with_no_grad(5.0);
        let r = mul(a, b);
        assert_eq!(r.eval().real, 25_f32);
        assert_eq!(r.eval().dual, 5_f32);
        println!("{} = {}", r.trace(), r.eval().to_string());
    }
```
And its output
```
(5 + 1ε * 5 + 0ε) = 25 + 5ε
```
 You might notice theres a lot of extra fluff here, `Node`, `eval`, `trace`. The reason for this that my actual goal is to create a system for lazily creating the computation graph, then evaluating and/or tracing it later, the dual numbers are just a fun rabbit hole I fell down while I fully develop this system.

## Benefits and Limitations
There is a reason why we don't use forward mode AD for training neural networks, as you may have noticed, the derivatives flow from the inputs to the outputs, and also there are issues with preserving the differentiation properties when performing arithmetic between two dual numbers that both have non-zero dual parts, because of this, in a machine learning system you would need to run multiple forwards passes, one for each object you are optimizing, unfortunately, this falls quite shore relative to backwards AD.
Thats not to say there are no benefits or use cases for forward AD in machine learning, the primary benefit is memory usage, forward AD is much more memory efficient, at the cost of many times more computation under usual circumstances. However, one specific scenario I imagine could potentially benefit from forward AD is fine tuning a model with a LoRA, especially in a reinforcement learning setting. If you only had a single vector or matrix to optimize, you could use the memory saving from not storing intermediate results to more simultaneous rollouts, or rollouts that are able to last much longer due, etc. I would like to explore this more in the future.

## Future Work
I have several goals for this codebase in the future, the first is increasing my fluency with rust and building more complex systems in it than I have before. My second goal is practice creating libraries with good developer experience specifically for machine learning tasks. My last goal is to implement some system for optimizing/lowering the graph into a more performant form, aka a ML compiler, I use torch.compile everyday, and I have a working understanding of how it and similar systems work, but I desire a much deeper level understanding, in order to both optimize my existing work that utilizes these systems, but also to be able to contribute to improving ML compiler projects and the libraries that they are a part of.
