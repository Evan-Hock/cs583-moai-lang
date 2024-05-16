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

type AST = [Definition]

type Name = String

data Definition
    = FunDef Name Params Expr
    | Const Name Expr

-- Kind of big
data Expr
    = BinOp BinOperator Expr Expr
    | App Expr Expr
    | Expr Case Expr [Possibility]
    | For Expr Expr
    | Let Pattern Expr Expr
    | Lam Params Expr
    | Foldl Expr Expr Expr
    | Foldr Expr Expr Expr
    | Product Expr Expr

type Params = [Name]

data BinOperator
    = Plus -- +
    | Minus -- -
    | Times -- *
    | Divide -- /
    | Modulus -- mod
    | Index

data Possibility
    = Possibility Pattern Expr

data Pattern
    = Var Name
    | Cons Name [Pattern]
