# Moai Examples


## Hello, World

J
```
echo 'Hello, world!'
```

Moai
```
puts "Hello, world!"
```


## Average

This is a quippy J function that is frequently used to demonstrate the capabilities of the language to inquisitive neophytes.

J
```
avg =: +/ % #   NB. The average is the sum (+/) over (%) the number of elements (#)
``` 

Moai
```
def avg(xs).
  foldl xs from 0 with (+) / len(xs)
```


## Fibonacci

J
```
fib =: (-&2 +&$: -&1) ^: (1&<)
```
This can be understood as follows.

This function computes whether its argument is greater than 1. If it is not, that argument is returned. (This is what the `^: (1&<)` section is doing.)

Then, it executes the functions `-&2` and `-&1` on the argument, which are both analogous to Haskell `subtract 2` and `subtract 1`, respectively.

The results of these are recurred upon, as `$:` is the recursion primitive. Finally, they are summed together.

Taken from the [J wiki article](https://code.jsoftware.com/wiki/Essays/Fibonacci_Sequence) on the matter. :)


Moai
```
def fib(n).
  case n < 1.
    true. n
    false. fib(n - 2) + fib(n - 1)
```


Moai will not have any "if" statements.


## Matrix Multiplication

Matrix multiplication is carried out via the `.` operator in J, which is infuriatingly whitespace sensitive.

`M +/ . * N` will yield the multiplication of M and N.

```
M =: i.2 2
N =: 2 3 $ 1 + |.i.6
echo M +/ . * N   NB. The spaces are not optional
```

Here is the Moai version.
```
def M.
    1 + iota(4) as [2, 2]


def N.
    reverse(1 + iota(6)) as [2, 3]


def main.
    print(M @ N).
```


Will output
```
[ 12, 9, 6
; 30, 23, 16
]
```


The @ symbol is chosen by precedent of Python, the greatest programming language.


## Golden Ratio

J
```
phi =: (1&+@(1&%)^:_) 1
```

This iterates the function `1&+@(1&%)` until its value stops changing (as defined by a certain tolerance).


Moai
```
def phi.
    iterate x from 1 until x =~= phi_step(x). # Approx equals
        phi_step(x)


def phi_step(x).
    1 + (1/x)
```


## Scalar Pervasion

As in J, scalars are automatically lifted by certain primitives to apply over their arguments.

J
```
1 + 1 2 3
```

yields

```
2 3 4
```

Moai
```
1 + [1, 2, 3]
```

yields

```
[2, 3, 4]
```
