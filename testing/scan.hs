module Main where


import Control.Exception
import qualified Data.ByteString.Lazy as LazyByteString
import System.Environment
import System.Exit

import Moai.Tokenizer


usage :: IO a
usage = do
    progName <- getProgName
    die $ "Usage: " ++ progName ++ " <source-file>"


main = do
    filepath <- handle ( \ PatternMatchFail -> usage ) $ do
        filepath:_ <- getArgs
        return filepath

    source <- LazyByteString.readFile filepath
    either die (mapM_ print) (tokenize source)