module Moai.AST
    ( AST
    , Name
    , Definition
    , Expr(..)
    , Params
    , Alternatives
    , NonEmptyParams
    , BinOperator(..)
    , UnOperator(..)
    , Pattern(..)
    , SimplePattern(..)
    , Term(..)
    , Literal(..)
    ) where


import Data.List.NonEmpty (NonEmpty)

type AST = Expr

type Name = String

type Definition
    = (Name, Params, Expr)

data Expr
    = UnOp UnOperator Expr -- Application of a builtin function. These are special-cased by the compiler, at least for now.
    | BinOp BinOperator (Maybe Expr) (Maybe Expr) -- Application of a binary function, or an operator section.
    | App Expr (NonEmpty Expr) -- Application of a function to a NONEMPTY list of arguments.
    | Case Expr Alternatives -- Pattern-matching case expressions.
    | For Expr Name Expr -- For loops (map expressions).
    | Let Pattern Expr Expr -- Local bindings.
    | Lambda NonEmptyParams Expr
    | Foldl Expr Expr Expr -- Left-associative fold over a list of elements, with an optional starting argument applied to the far left.
    | Foldr Expr Expr Expr -- Right-associative fold over a list of elements, with an optional starting argument applied to the far right.
    | Iterate Name Expr Expr Expr -- Iteration expressions, analogous to Haskell's "until" function, or APL's "power" operator when passed a function as its right argument.
    | Term Term

type Params
    = [Pattern]

type NonEmptyParams
    = NonEmpty Pattern

data BinOperator
    = Add -- +
    | Sub -- -
    | Mul -- *
    | Div -- /
    | Reshape -- as
    | From -- at
    | Eq -- ==
    | Neq -- /=
    | Gt -- >
    | Gte -- >=
    | Lt -- <
    | Lte -- <=

data UnOperator
    = Identity -- identity()
    | Neg -- neg()
    | Not --  not()
    | Len -- len()
    | Shape -- shape()
    | Ndim -- ndim()
    | Iota -- iota()
    | Reverse -- reverse()
    | Abs -- abs()

type Alternatives
    = NonEmpty (Pattern, Expr)

data Pattern
    = Base SimplePattern
    | Array [SimplePattern] -- [1, 2, 3] or [x, y, z]
    | Matrix Int Int [SimplePattern] -- [1, 2, 3; x, y, z] -> Matrix 2 3 [Num 1, Num 2, Num 3, Var "x", Var "y", Var "z"]
    | Rest [SimplePattern] (Maybe Name) -- [x, ..xs] or [x, y, ..]

data SimplePattern
    = Var Name
    | Num Double

data Term
    = Id Name
    | Lit Literal

data Literal
    = ScalarLit Double -- e.g. 1
    | ArrayLit [Double] -- e.g. [1, 2, 3]
    | MatrixLit [[Double]] -- e.g. [1, 2; 3, 4]. If the matrix is non-rectangular, it will be filled with 0s.