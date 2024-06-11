{
module Moai.Tokenizer
    ( tokenize
    , TokenCategory(..)
    , Token(..)
    ) where


import Data.ByteString (StrictByteString)
import qualified Data.ByteString.Char8 as Char8
import Data.Char
}


%wrapper "monadUserState-bytestring"


$digit = 0-9
$alpha = [a-zA-Z_]
$ws = $white # \n
$significant = [^$white\#]
$pm = [\+\-]
@identifier = $alpha [$alpha $digit]*
@float = $pm? $digit+ "." $digit+
@integer = $pm? $digit+
@boolean = true | false
@ignore = $ws*(\#[^\n]*)?\n


tokens :-
    ^@ignore / $significant { \ _ _ -> dedentAll }
    ^@ignore ;
    \n / $significant { \ _ _ -> fmap (unannotated Newline :) <$> dedentAll }
    \n { token $ \ _ _ -> newline }
    ^$ws+ / $significant { handleIndentation }
    $ws+ ;

    -- Keywords and punctuation
    as { basic As }
    at { basic At }
    case { basic Case }
    foldl { basic Foldl }
    foldr { basic Foldr }
    for { basic For }
    from { basic From }
    iterate { basic Iterate }
    in { basic In }
    until { basic Until }
    with { basic With }

    identity { basic Identity }
    neg { basic Neg }
    not { basic Not }
    len { basic Len }
    shape { basic Shape }
    ndim { basic Ndim }
    iota { basic Iota }
    reverse { basic Reverse }
    abs { basic Abs }

    @float { float }
    @integer { integer }
    @boolean { boolean }

    @identifier { identifier }

    "." { basic Period }
    "," { basic Comma }
    ";" { basic Semicolon }
    "=" { basic Bind }
    "(" { basic LParen }
    ")" { basic RParen }
    "[" { basic LBrace }
    "]" { basic RBrace }
    "+" { basic Cross }
    "-" { basic Hyphen }
    "*" { basic Star }
    "/" { basic Slash }
    "%" { basic Percent }
    "==" { basic Eq }
    "/=" { basic Neq }
    "<" { basic Lt }
    "<=" { basic Lte }
    ">" { basic Gt }
    ">=" { basic Gte }
    \\ { basic Lambda }


{
data TokenCategory
    = As
    | At
    | Bind
    | Case
    | Comma
    | Cross
    | Number Double
    | Eq
    | Neq
    | Foldl
    | Foldr
    | For
    | From
    | Gt
    | Gte
    | Lambda
    | LBrace
    | LParen
    | Lt
    | Lte
    | RBrace
    | RParen
    | Iterate
    | In
    | Percent
    | Period
    | Slash
    | Star
    | Until
    | With
    deriving
        ( Eq
        , Show
        )


data EofMarked a = EofMarked Bool a
    deriving
        ( Eq
        , Show
        )


instance Functor EofMarked where
    fmap f (EofMarked b x) = EofMarked b (f x)


data Token = Token (Maybe Annotation) TokenCategory


data Annotation = Annotation
    { line :: Int
    , col :: Int
    , str :: StrictByteString
    }


data AlexUserState = AlexUserState
    { indentDepth :: Int
    , indentStack :: Int
    }


type Result = EofMarked [Token]


indent :: Int64 -> Alex Result
indent len = do
    ust <- alexGetUserState
    alexSetUserState AlexUserState
        { indentStack = len - ust.indentDepth : ust.indentStack
        , indentDepth = len
        }


dedentUntil :: Int64 -> Alex Int
dedentUntil len = go 0
  where
    go n = do
        ust <- alexGetUserState
        if ust.indentDepth <= len then
            return n
        else case ust.indentstack of
            [] -> return n
            top : rest -> do
                alexSetUserState AlexUserState
                    { indentStack = rest
                    , totalIndentation = ust.indentDepth - top
                    }

                go (n + 1)


dedent :: Int64 -> Alex Result
dedent len = do
    dedents <- dedentUntil len
    s <- alexGetUserState
    when (s.indentDepth /= len) $
        alexError "Unexpected indentation level"

    proceed $ replicate dedents Dedent


handleIndentation :: AlexAction Result
handleIndentation (AlexPn _ line col), _, str, _) len = do
    ust <- alexGetUserState
    case compare len ust.indentDepth of
        EQ -> ignore
        GT -> indent
        LT -> dedent


float :: AlexAction Result
float = pureT $ \ str ->
    case Char8.unpack str of
        '+':rest -> Float (read rest)
        '-':rest -> Float (negate (read rest))
        xs -> Float (read xs)


integer :: AlexAction Result
integer = pureT $ Integer . read . Char8.unpack


boolean :: AlexAction Result
boolean = pureT $ Boolean . read . capitalize . Char8.unpack


capitalize :: String -> String
capitalize [] = []
capitalize (h:t) = toUpper h : t


basic :: TokenCategory -> AlexAction Result
basic = pureT . const


pureT :: (StrictByteString -> TokenCategory) -> AlexAction Result
pureT f = token $ \ (AlexPn _ line col, _, str, _) len ->
    let
        str' = ByteString.toStrict (ByteString.take len str)
        annotation = Just $ Annotation { str = str', .. }
        category = f str'
    in
        proceed [Token annotation category]


unannotated :: TokenCategory -> Token
unannotated = Token Nothing


eof :: a -> Alex (EofMarked a)
eof = return . EofMarked True


proceed :: a -> Alex (EofMarked a)
proceed = return . EofMarked False


ignore :: a -> Alex Result
ignore = proceed []


alexEOF :: Alex Result
alexEOF = do
    finalDedents <- dedentAll
    eof finalDedents


tokenize :: ByteString -> Either String [Token]
tokenize input = runAlex input go
  where
    go = do
        EofMarked halt toks <- alexMonadScan
        if halt then
            return toks
        else case toks of
            [] -> go
            _ -> fmap (toks ++) go
}
