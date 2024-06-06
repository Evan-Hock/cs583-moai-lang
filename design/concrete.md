# Moai Concrete Syntax

The language does not not have a context-free grammar. It utilizes off-side rule for many features.

## Note On Syntax Notation

This document utilizes a version of EBNF that allows for productions to take parameters. These can be
thought of as macros for productions.


## Function/Constant Definitions

```
<def> ::= "def" <name> ["(" <list(",", <pattern>)> ")"] "." <expr>
<list($delim, $p)> ::= [$p [$delim <list($delim, $p)>]]
```


The toplevel consists of a list of these definitions. The argument list may be either present or absent. It
may also be present and have zero parameters contained within. The semantics for an argument list with zero
parameters and an absent argument list are the same.

After registering the definitions, the program will begin from the entrypoint, which is a function called "main".
`main` may have zero or more arguments. In the event that it has more than zero arguments,
its first argument will be initialized with an array containing the following values.

1. The program name, as invoked from the command-line
2. The command line arguments, which consist of the rest of the array.


The other arguments can be present, but they will never be used.

*Stretch goal.*

It may be more desirable to introduce variadic functions into the language, and call main with the
list of command line arguments instead of just the first argument with an array of arguments.


## Expressions

The following constructions are valid expressions.

1. Function applications

```
<expr> "(" <nonempty-list(",", <expr>)> ")"
```

where

```
<nonempty-list($delim, $p)> ::= $p [$delim <list($delim, $p)>]
```


2. Binary operator applications

```
<expr> <op> <expr>
```

All binary operators are left-associative.

All binary operators have the same precedence, due to array language precedent.

There are no unary operators.


3. "Case" expressions

```
"case" <expr> "." {<alternative>}+
```

where

```
<alternative> ::= <pattern> "." <expr>
```


4. "For" expressions

```
"for" <pattern> "in" <expr> "." <expr>
```


5. "Let" expressions

```
"let" <pattern> "=" <expr> "\n" <expr>
```

These are let bindings. Only one binding may be introduced at a time, i.e. these let bindings do not introduce a new indentation context.


6. Lambda expressions

```
"\" <nonempty-list(<whitespace>, <pattern>)> "." <expr>
```


7. Fold expressions

```
<fold-type> <expr> ["from" <expr>] "with" <expr>
```

where

```
<fold-type> ::= "foldl" | "foldr"
```

If the middle expression is absent, the fold is assumed to be a fold over a nonempty collection. This has the
same semantics as Haskell's `foldl1` and `foldr1` functions.

The argument order for fold expressions is as follows: the first argument is the collection, the second argument
is the initial accumulator, and the third argument is the folding function.

8. "iterate" expressions

```
"iterate" <pattern> "from" <expr> "until" <expr> "." <expr>
```

The semantics of this are very similar to Haskell's `until` function.

These expressions consist of three portions.

The pattern introduces bindings that may be utilized within the rightmost two expressions.

The first expression is a value with which to initialize the pattern. It is matched against the pattern and bound to it.

The expression after "until" is used to indicate whether the iteration should halt. The bindings introduced by the pattern are bound here. If the expression evaluates to a truthy value based on the
current value, the iteration is to halt.

The final expression is used to indicate what the value for the next iteration should be.

The final return value of an "iterate" expression is the final value of the iteration.


9. Sections

```
"(" <op> <expr> ")" | "(" <expr> <op> ")"
```

Sections are available, and function precisely as they do in Haskell.


10. Literals

The following literals are supported.

* Integer literals
* Floating-point literals
* Array literals, with the conventional `[e1, e2, ..., eN]` syntax.
* Boolean literals, with the literals `true` and `false`.
* String literals, with double-quotes.


11. Types

The language shall be dynamically typed. A more advanced type system is not planned at this time.


12. Builtins

The language shall come with a small assortment of builtins.

### puts

Prints a string to the console.


### identity

Function that neither modifies its argument nor affects anything.


### neg

Unary negation.


### not

Boolean negation.


### len

Retrieve the number of individual 0-cells in an array.


### shape

Yields the shape of an array, which is an array consisting of the sizes of each of its dimensions.


### ndim

Yields the rank of an array, which is the number of dimensions that it has.


### iota

Yields a one-dimensional array of integers from 0 up to (but not including) its argument.


### reverse

Reverses the major cells in an array.


### abs

Computes the magnitude of a number.