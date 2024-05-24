module Moai.AST
    ( AST
    , Definition(..)
    , Body(..)
    , SeqExpr(..)
    , Expr(..)
    , PrimaryExpr(..)
    , Value(..)
    , BinOperator(..)
    , Possibility(..)
    , Pattern(..)
    ) where

infixr 9 :>>

type AST = [Definition]

type Name = String

data Definition
    = Def Name Params Expr

-- Kind of big
data Expr
    = BinOp BinOperator (Maybe Expr) (Maybe Expr)
    | App Expr (NonEmpty Expr)
    | Case Expr [Possibility]
    | For Expr Name Expr
    | Let Pattern Expr Expr
    | Lambda Params Expr
    | Foldl Expr (Maybe Expr) Expr
    | Foldr Expr (Maybe Expr) Expr
    | Product Expr Expr
    | Expr :>> Expr

type Params = [Name]

data BinOperator
    = Plus -- +
    | Minus -- -
    | Times -- *
    | Divide -- /
    | Modulus -- mod
    | Reshape
    | From

data Possibility
    = Possibility Pattern Expr

data Pattern
    = Var Name
    | Cons Name [Pattern]
