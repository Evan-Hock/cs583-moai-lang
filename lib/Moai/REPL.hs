module Moai.REPL
    ( start )
    where


import qualified Data.ByteString.Lazy as LazyByteString


start :: FilePath -> IO ()
start filepath = do
    source <- LazyByteString.readFile filepath
    putStrLn source