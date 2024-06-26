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

After processing the definitions, the language will drop into the REPL, with all of the loaded definitions available.


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

All binary operators are right-associative.

All binary operators have the same precedence, due to array language precedent.


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
"for" <name> "in" <expr> "." <expr>
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
<fold-type> <expr> "from" <expr> "with" <expr>
```

where

```
<fold-type> ::= "foldl" | "foldr"
```

The argument order for fold expressions is as follows: the first argument is the collection, the second argument
is the initial accumulator, and the third argument is the folding function.

8. "iterate" expressions

```
"iterate" <name> "from" <expr> "until" <expr> "." <expr>
```

The semantics of this are very similar to Haskell's `until` function.

These expressions consist of three portions.

The name introduces a binding that may be utilized within the rightmost two expressions.

The first expression is a value with which to initialize the binding.

The expression after "until" is used to indicate whether the iteration should halt. The binding in the first clause is bound here. If the expression evaluates to a truthy value
based on the current value, the iteration is to halt.

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
* Floating point literals
* Array literals, with the conventional `[e1, e2, ..., eN]` syntax.
* Matrix literals, with the slightly less conventional `[e11, e12, ..., e1N; e21, e22, ... e2N; ...; eM1, eM2, ..., eMN]


11. Types

The language shall be dynamically typed. A more advanced type system is not planned at this time.


12. Builtins

The language shall come with a small assortment of builtins.


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