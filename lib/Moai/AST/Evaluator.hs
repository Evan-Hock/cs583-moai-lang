module Moai.AST.Evaluator
    ( eval
    , Shape
    , MoaiData(..)
    , MoaiArray(..)
    , MoaiError(..)
    , MoaiException
    , MoaiEvalMonad
    , MoaiEnvironment
    ) where


import Control.Applicative
import Control.Monad
import Control.Monad.Trans.Class
import Control.Monad.Trans.Reader
import Data.Array
import Data.Foldable
import Data.HashMap.Strict (HashMap)
import qualified Data.HashMap.Strict as HashMap
import Data.List as List
import Data.List.NonEmpty (NonEmpty(..))
import qualified Data.List.NonEmpty as NonEmpty
import Data.Ord
import Moai.AST


-- This representation is not great, but in practice
-- the shape should never be longer than like 3 or 4,
-- so I think it is acceptable
type Shape = [Int]


data MoaiData
    = Verb Params Expr -- Functions. Thunks have empty parameter lists.
    | Noun MoaiArray -- Non-function objects, aka arrays.


data MoaiArray
    = MoaiArray
        { moaiarray_shape :: Shape
        , moaiarray_data :: Array Int Double
        }


data MoaiError
    = TypeError -- Type mismatch.
    | ShapeError -- Incompatible shapes
    | RankError -- Function applied with improper rank
    | IndexError -- Out-of-bounds index
    | ArgumentError -- Too many arguments to function
    | PatternFailure -- Failure to bind a pattern
    | ReferenceError -- Failed to find a name


type MoaiException = (MoaiError, String)


type MoaiEnvironment = HashMap Name MoaiData


type MoaiEvalMonad = ReaderT MoaiEnvironment (Except MoaiException) MoaiData


data FoldDirection
    = Left
    | Right


registerDefinitions :: [Definition] -> MoaiEnvironment
registerDefinitions = foldl' registerDefinition HashMap.empty


registerDefinition :: MoaiEnvironment -> Definition -> MoaiEnvironment
registerDefinition env (name, params, body) = HashMap.insert name (Verb params body) env


eval :: MoaiEnvironment -> AST -> Except MoaiException MoaiData
eval env expr = runReaderT (eval' expr) env


eval' :: Expr -> MoaiEvalMonad
eval' (UnOp op expr) = evalUnOp op expr
eval' (BinOp op mlarg mrarg) = evalBinOp op mlarg mrarg
eval' (App f args) = evalApp f args
eval' (Case expr alts) = evalCase expr (NonEmpty.toList alts)
eval' (For expr name body) = evalFor expr name body 
eval' (Let pat expr body) = evalLet pat expr body
eval' (Lambda params expr) = evalLambda params expr
eval' (Iterate name init halter body) = evalIterate name init halter body
eval' (Foldl expr fromExpr fexpr) = evalFold Left expr fromExpr fexpr
eval' (Foldr expr fromExpr fexpr) = evalFold Right expr fromExpr fexpr
eval' (Term term) = evalTerm term


evalTerm :: Term -> MoaiEvalMonad
evalTerm (Id name) = lookupName name
evalTerm (Lit lit) = evalLit lit


lookupName :: Name -> MoaiEvalMonad
lookupName name = do
    mval <- asks (HashMap.lookup name)
    case mval of
        Nothing -> lift $ throwE (ReferenceError, "Unknown name '" ++ name ++ "'")
        Just val ->
            case val of
                -- For right now these will just function as macros
                Verb [] expr -> eval' expr
                _ -> return val


evalLit :: Literal -> MoaiEvalMonad
evalLit (ScalarLit n) = evalScalar n
evalLit (ArrayLit values) = evalList values
evalLit (MatrixLit values) = evalMatrix values


evalScalar :: Double -> MoaiEvalMonad
evalScalar = return . Noun . moaiScalar


evalList :: [Double] -> MoaiEvalMonad
evalList list = return $ Noun MoaiArray
    { moaiarray_shape=[len]
    , moaiarray_data=listArray (0, len - 1) list
    }
  where
    len = length list


evalMatrix :: [[Double]] -> MoaiEvalMonad
evalMatrix matrix = return $ Noun MoaiArray
    { moaiarray_shape=shape
    , moaiarray_data=listArray (0, product shape - 1) . concatMap (fill ncols) $ matrix
    }
  where
    ncols = maximum (map length matrix)
    shape = [length matrix, ncols]


fill :: Int -> [Double] -> [Double]
fill n xs = xs ++ replicate (n - length xs) 0


evalFold :: FoldDirection -> Expr -> Expr -> Expr
evalFold dir expr fromExpr fexpr = do
    res <- eval' expr
    case res of
        Noun coll -> do
            f <- eval' fexpr
            case f of
                Verb params@[_, _] body -> do
                    zero <- eval' fromExpr
                    let v = folder params body
                    case dir of
                        Left -> foldM v zero (moaiarray_data coll)
                        Right -> foldrM (flip v) zero (moaiarray_data coll)

                _ -> lift $ throwE (TypeError, "Folds can only be done with binary functions")
        
        _ -> lift $ throwE (TypeError, "Folds must be done over a collection")


folder :: Params -> Expr -> MoaiData -> Double -> MoaiEvalMonad
folder params body acc x =
    case bindParams params (acc :| [Noun (moaiScalar x)]) of
        Nothing -> lift $ throwE (PatternFailure, "Failed to bind some patterns in a function application in a fold expression")
        Just bindings -> bindAndEval bindings body


evalIterate :: Name -> Expr -> Expr -> Expr
evalIterate binding init halter body = do
    initRes <- eval' init
    go initRes
  where
    go currVal = do
        halt <- bindAndEval [(binding, currVal)] halter
        case halt of
            Noun MoaiArray{moaiarray_data=haltVal} | any (== 0) (elems haltVal) ->
                bindAndEval [(binding, currVal)] body >>= go
        
            _ -> return currVal


evalLambda :: NonEmptyParams -> Expr -> MoaiEvalMonad
evalLambda params body = return $ Verb (NonEmpty.toList params) body


evalLet :: Pattern -> Expr -> Expr -> MoaiEvalMonad
evalLet pat expr body = do
    res <- eval' expr
    case tryBind pat res of
        Nothing -> lift $ throwE (PatternFailure, "Pattern match failure in let binding")
        Just bindings -> bindAndEval bindings body


evalFor :: Expr -> Name -> Expr -> MoaiEvalMonad
evalFor expr name body = do
    res <- eval' expr
    case res of
        Noun array -> do
            resData <- for (moaiarray_data array) $ \ elem ->
                bindAndEval [(name, Noun . moaiScalar $ elem)] body

            return $ Noun array{moaiarray_data=resData}

        _ -> lift $ throwE (TypeError, "Cannot iterate over a function")


evalCase :: Expr -> [(Pattern, Expr)] -> MoaiEvalMonad
evalCase testExpr alts = do
    testValue <- eval' expr
    go testValue alts
  where
    go testValue [] = lift $ throwE (PatternFailure, "Pattern match failure in case expression")
    go testValue ((pat, expr):rest) =
        case tryBind pat testValue of
            Just bindings -> bindAndEval bindings expr
            Nothing -> go rest


tryBind :: Pattern -> MoaiData -> Maybe [(Name, MoaiData)]
tryBind (Base simple) = tryBindSimple simple
tryBind (Array pats) = tryBindArray pats
tryBind (Matrix nrows ncols pats) = tryBindMatrix nrows ncols pats
tryBind (Rest pats mrest) = tryBindRest pats mrest


tryBindSimple :: SimplePattern -> MoaiData -> Maybe [(Name, MoaiData)]
tryBindSimple (Num n) = tryBindNum n
tryBindSimple (Var v) = Just . List.singleton . bindSingleVar v


nounBinder :: (MoaiArray -> Maybe [(Name, MoaiData)]) -> MoaiData -> Maybe [(Name, MoaiData)]
nounBinder binder testValue = do
    Noun array <- testValue
    binder array


tryBindNum :: Double -> MoaiData -> Maybe [(Name, MoaiData)]
tryBindNum n = nounBinder $ \ array -> do
    guard $ null . moaiarray_shape $ array
    guard $ moaiarray_data array ! 0 == n
    return []


bindSingleVar :: Name -> MoaiData -> (Name, MoaiData)
bindSingleVar = (,)


tryBindArray :: [SimplePattern] -> MoaiData -> Maybe [(Name, MoaiData)]
tryBindArray pats = nounBinder $ \ array -> do
    [len] <- moaiarray_shape array
    guard $ len == length pats
    bindPatList pats array


tryBindMatrix :: Int -> Int -> [SimplePattern] -> MoaiData -> Maybe [(Name, MoaiData)]
tryBindMatrix nrows ncols pats = nounBinder $ \ array -> do
    shape@[_, _] <- moaiarray_shape array
    guard $ shape == [nrows, ncols]
    bindPatList pats array


tryBindRest :: [SimplePattern] -> Maybe Name -> MoaiData -> Maybe [(Name, MoaiData)]
tryBindRest pats mrest = nounBinder $ \ array -> do
    let npats = length pats
    [len] <- moaiarray_shape array
    guard $ len <= npats
    bindings <- bindPatList pats array
    return $
        case mrest of
            Nothing -> bindings
            Just v -> 
                let
                    n = len - npats
                    rest = Noun MoaiArray
                        { moaiarray_shape=[n]
                        , moaiarray_data=listArray (0, n - 1) . drop npats . elems . moaiarray_data $ array
                        }
                in
                    bindings ++ [(v, rest)]


bindPatList :: [SimplePattern] -> MoaiArray -> Maybe [(Name, MoaiData)]
bindPatList pats array =
    concat <$> for (zip pats . elems . moaiarray_data $ array) ( \ (pat, elem) ->
        tryBindSimple pat . Noun . moaiScalar $ elem )


bindAndEval :: [(Name, MoaiData)] -> Expr -> MoaiEvalMonad
bindAndEval bindings expr =
    local (bindEnvironment bindings) $
        eval' expr


bindEnvironment :: [(Name, MoaiData)] -> MoaiEnvironment -> MoaiEnvironment
bindEnvironment newBindings bindings = foldl' (uncurry HashMap.insert) bindings newBindings


evalApp :: Expr -> NonEmpty Expr -> MoaiEvalMonad
evalApp f args = do
    f' <- eval' f
    case f' of
        Verb params body -> app params args body
        _ -> lift $ throwE (TypeError, "Attempt to apply to non-function")


app :: Params -> NonEmpty Expr -> Expr -> MoaiEvalMonad
app params args body =
    case NonEmpty.length args `compare` length params of
        LT -> lift $ throwE (ArgumentError, "Too few arguments applied to function")
        GT -> lift $ throwE (ArgumentError, "Too many arguments applied to function")
        EQ -> do
            args' <- traverse eval' args
            case bindParams params args' of
                Nothing -> lift $ throwE (PatternFailure, "Failed to bind some patterns in a function application")
                Just bindings -> bindAndEval bindings body


bindParams :: Params -> NonEmpty MoaiData -> Maybe [(Name, MoaiData)]
bindParams params args =
    concat <$> for (zip params . NonEmpty.toList $ args) $
        uncurry tryBind


-- This is unimaginably jank, but it's not too far off from how J presumably does this
evalBinOp :: BinOperator -> Expr -> Expr -> MoaiEvalMonad
evalBinOp op Nothing Nothing = return $ dyad (BinOp op (arg "x") (arg "y"))
evalBinOp op Nothing right@(Just _) = return $ monad (BinOp op (arg "y") right)
evalBinOp op left@(Just _) Nothing = return $ monad (BinOp op left (arg "y"))
evalBinOp op (Just left) (Just right) = evalBinOp' op left right


arg :: Name -> Maybe Expr
arg = Just . Term . Id


param :: Name -> Pattern
param = Base . Var


dyad :: Expr -> MoaiData
dyad = Verb (param "x" :| [param "y"])


monad :: Expr -> MoaiData
monad = Verb (NonEmpty.singleton (param "y"))


evalBinOp' :: BinOperator -> Expr -> Expr -> MoaiEvalMonad
evalBinOp' Add = moaiAdd
evalBinOp' Sub = moaiSub
evalBinOp' Mul = moaiMul
evalBinOp' Div = moaiDiv
evalBinOp' Reshape = moaiReshape
evalBinOp' From = moaiFrom
evalBinOp' Eq = moaiEq
evalBinOp' Neq = moaiNeq
evalBinOp' Gt = moaiGt
evalBinOp' Gte = moaiGte
evalBinOp' Lte = moaiLte


-- Extracts the boilerplate of typechecking out of binary array functions
ef2 :: String -> (MoaiArray -> MoaiArray -> Except MoaiException MoaiArray) -> Expr -> Expr -> MoaiEvalMonad
ef2 name body expr1 expr2 = do
    case (eval' expr1, eval' expr2) of
        (Noun x, Noun y) -> lift $ Noun <$> body x y
        _ -> arrayExpected name


-- Applies a binary function between two arrays
moaiArrayZipWith :: String -> (Double -> Double -> Double) -> MoaiArray -> MoaiArray -> Except MoaiException MoaiArray
moaiArrayZipWith name op x y =
    case comparing (length . moaiarray_shape) x y of
        GT -> moaiArrayZipWith' name (flip op) y x
        _ -> moaiArrayZipWith' name op x y


moaiArrayZipWith' :: String -> (Double -> Double -> Double) -> MoaiArray -> MoaiArray -> Except MoaiException MoaiArray
moaiArrayZipWith' name op smaller@MoaiArray{moaiarray_shape=smallshape} bigger@MoaiArray{moaiarray_shape=bigshape}
    | smallshape `isPrefixOf` bigshape =
        return $
            let
                itemSize = product (drop (length smallshape) bigshape)
            in
                bigger{moaiarray_data=listArray (bounds bigger) (concat . zipWith (map . op) (elems smaller) . chunksOf itemSize . elems $ bigger)}
    | otherwise = throwE (ShapeError, "Incompatible shapes in binary function '" ++ name ++ "'")


-- Extracts some of the boilerplate out of the binary functions
rank0BinOp :: String -> (Double -> Double -> Double) -> Expr -> Expr -> MoaiEvalMonad
rank0BinOp name = ef2 name . moaiArrayZipWith name

 
moaiAdd :: Expr -> Expr -> MoaiEvalMonad
moaiAdd = rank0BinOp "+" (+)


moaiSub :: Expr -> Expr -> MoaiEvalMonad
moaiSub = rank0BinOp "-" (-)


moaiMul :: Expr -> Expr -> MoaiEvalMonad
moaiMul = rank0BinOp "*" (*)


moaiDiv :: Expr -> Expr -> MoaiEvalMonad
moaiDiv = rank0BinOp "/" (/)


-- Makes a perfectly good Boolean function naughty
doublise :: Eq a => (a -> a -> Bool) -> a -> a -> Double
doublise = fmap (fmap (fromIntegral . fromEnum))


moaiEq :: Expr -> Expr -> MoaiEvalMonad
moaiEq = rank0BinOp "==" (doublise (==))


moaiNeq :: Expr -> Expr -> MoaiEvalMonad
moaiNeq = rank0BinOp "/=" (doublise (/=))


moaiGt :: Expr -> Expr -> MoaiEvalMonad
moaiGt = rank0BinOp ">" (doublise (>))


moaiGte :: Expr -> Expr -> MoaiEvalMonad
moaiGte = rank0BinOp ">=" (doublise (>=))


moaiLt :: Expr -> Expr -> MoaiEvalMonad
moaiLt = rank0BinOp "<" (doublise (<))


moaiLte :: Expr -> Expr -> MoaiEvalMonad
moaiLte = rakn0BinOp "<=" (doublise (<=))


moaiReshape :: Expr -> Expr -> MoaiEvalMonad
moaiReshape = ef2 "as" $ \ dataSource shape ->
    case moaiarray_shape shape of
        _:_:_ -> throwE (RankError, "Cannot reshape with non-vector")
        _ ->
            return $
                let
                    newShape = floor <$> elems (moaiarray_data shape)
                    xs = elems (moaiarray_data dataSource)
                in
                    MoaiArray
                        { moaiarray_shape=newShape
                        , moaiarray_data=
                            listArray (0, product newShape - 1) $
                                if null xs then
                                    repeat 0
                                else
                                    cycle xs
                        }


moaiFrom :: Expr -> Expr -> MoaiEvalMonad
moaiFrom = ef2 "at" $ \ array indices ->
    case moaiarray_shape array of
        [] -> throwE (RankError, "Cannot index into a scalar")
        shape ->
            case moaiarray_shape indices of
                _:_:_ -> throwE (RankError, "Cannot index with a value with rank higher than a vector")

                _ -> indexMany (floor <$> elems (moaiarray_data indices)) shape (moaiarray_data array)


-- Pulls a slice out of an array
slice :: (Int, Int) -> Array Int Double -> Except MoaiException (Array Int Double)
slice (lo, hi) xs
    | lo < loBound || hi > hiBound = throwE (IndexError, "Index out of bounds")
    | otherwise = return $ listArray (0, len - 1) . take len . drop lo . elems $ xs
  where
    (loBound, hiBound) = bounds xs
    len = hi - lo + 1


-- Implements the indexing for the "from" function. Essentially calculates the
-- slice to extract from the array, and then pulls that slice out
indexMany :: [Int] -> Shape -> Array Int Double -> Except MoaiException MoaiArray
indexMany indices shape array = do
    (newShape, sliceIndices) <- foldM indexStep (shape, bounds array) indices
    MoaiArray newShape <$> slice sliceIndices array


-- Calculates the next slice based on the size of the current axis and the current slice
indexStep :: ([Int], (Int, Int)) -> Int -> Except MoaiException (Int, Int)
indexStep ([], _) _ = throwE (RankError, "Too many indices")
indexStep (leading:trailing, (lo, hi)) i
    | i < 0 || i >= leading = throwE (IndexError, "Index out of bounds")
    | otherwise = return (trailing, (lo + stride*i, lo + stride*(i + 1) - 1))
  where
    stride = (hi - lo + 1) `div` leading


evalUnOp :: UnOperator -> Expr -> MoaiEvalMonad
evalUnOp Identity = eval' -- Ignore
evalUnOp Neg = moaiNeg
evalUnOp Not = moaiNot
evalUnOp Len = moaiLen
evalUnOp Shape = moaiShape
evalUnOp Ndim = moaiNdim
evalUnOp Iota = moaiIota
evalUnOp Reverse = moaiReverse
evalUnOp Abs = moaiAbs


-- Used to implement the boilerplate for the builtin unary functions that cannot throw.
-- "name" is the name of the function, and is utilized for error reporting.
-- "body" is the body of the function. It has the specified type, as all of the functions return an array.
f1 :: String -> (MoaiArray -> MoaiArray) -> Expr -> MoaiEvalMonad
f1 name body = ef1 name (pure . body)
    

-- Used to implement the boilerplate for the builtin unary functions that may be able to throw.
-- "name" is the name of the function, and is utilized for error reporting.
-- "body" is the body of the function. It has the specified type, as all of the functions return an array.
-- "expr" is the ultimate paramter of the function, as an expression.
ef1 :: String -> (MoaiArray -> Except MoaiException MoaiArray) -> Expr -> MoaiEvalMonad
ef1 name body expr = do
    case eval' expr of
        Noun array -> lift $ Noun <$> body array
        _ -> arrayExpected name


arrayExpected :: String -> MoaiEvalMonad
arrayExpected name = lift $ throwE (TypeError, "Function provided where array expected in application of '" ++ name ++ "'")


moaiArrayMap :: (Double -> Double) -> MoaiArray -> MoaiArray
moaiArrayMap f ma@MoaiArray{moaiarray_data=xs} = ma{moaiarray_data=f <$> xs}


moaiNeg :: Expr -> MoaiEvalMonad
moaiNeg = f1 "neg" $ moaiArrayMap negate


doubleNot :: Double -> Double
doubleNot 0 = 1
doubleNot _ = 0


moaiNot :: Expr -> MoaiEvalMonad
moaiNot = f1 "not" $ moaiArrayMap doubleNot


moaiScalar :: Double -> MoaiArray
moaiScalar scalar =
    MoaiArray [] (listArray (0, 0) [scalar])


-- Yields the length of an array
moaiLen :: Expr -> MoaiEvalMonad
moaiLen = f1 "len" $ moaiScalar . fromIntegral . product . moaiarray_shape


moaiShape :: Expr -> MoaiEvalMonad
moaiShape = f1 "shape" $ \ MoaiArray{moaiarray_shape=shape} ->
    let
        len = length shape
    in
        MoaiArray{moaiarray_shape=[len], moaiarray_data=listArray (0, len - 1) (fromIntegral <$> shape)}


moaiNdim :: Expr -> MoaiEvalMonad
moaiNdim = f1 "ndim" $ moaiScalar . fromIntegral . length . moaiarray_shape


moaiIota :: Expr -> MoaiEvalMonad
moaiIota = ef1 "iota" $ \ m ->
    case moaiarray_shape m of
        _:_:_ -> throwE (RankError, "Iota can only take scalar or vector arguments")
        _ ->
            return $
                let
                    newShape = floor <$> elems (moaiarray_data m)
                in
                    MoaiArray{moaiarray_shape=newShape, moaiarray_data=listArray (0, product newShape - 1) [0..]}


moaiReverse :: Expr -> MoaiEvalMonad
moaiReverse = f1 "reverse" $ \ m@MoaiArray{moaiarray_shape=shape, moaiarray_data=xs} ->
    if null shape then
        m
    else
    let
        itemLength = product (tail shape)
    in
        m{moaiarray_data=listArray (bounds xs) (concat . reverse . chunksOf itemLength . elems $ xs)}


moaiAbs :: Expr -> MoaiEvalMonad
moaiAbs = f1 "abs" $ moaiArrayMap abs


-- Splits a list into sublists of size i. Cuts off the last chunk if there are not enough elements in the list.
chunksOf :: Int -> [a] -> [[a]]
chunksOf i
    | i <= 0 = error $ "chunksOf: chunk size must be positive; got " ++ show i
    | otherwise = go
  where
    go xs
        | null xs = []
        | otherwise = ys : go zs
      where
        (ys, zs) = splitAt i xs


-- Decomposes a list into (init, last) pair
unsnoc :: [a] -> Maybe ([a], a)
unsnoc [] = Nothing
unsnoc [x] = Just ([], x)
unsnoc (x:xs) = Just (x : ys, z)
  where
    -- This is an unsound pattern, but we are guaranteed to make it
    Just (ys, z) = unsnoc xs