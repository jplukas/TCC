import json
from typing import Any, List, Literal, Final, TypeAlias, get_args
from itertools import product
import chromadb

granType:TypeAlias = str

_CHROMA_CLIENT:Final = chromadb.PersistentClient("isidb")

with open('opts.json', 'r') as optsfile:
    _OPTS:Final[dict[str, Any]] = json.load(optsfile)

def getCollBaseName(emb_name: str) -> str:
    return emb_name.replace('-', '_').replace(':', '_').replace('/', '_')

def getCollName(emb_name: str, edition: str = "en") -> str:
    return getCollBaseName(emb_name) + "__" + edition

def getOpt(optName: str):
    return _OPTS[optName]

def getArg(argname: str | List[str] | None = None):
    if argname is None: return _OPTS["args"]
    if isinstance(argname, str): return _OPTS["args"][argname]
    return {k: _OPTS["args"][k] for k in argname}

def getDocs(collName: str, granularity: granType, include):
    collection = _CHROMA_CLIENT.get_collection(collName)
    wheredoc:Final = {"topics" : {"sentence": "all"}, "sentences": {"sentence": {"$ne" : "all"}}, "mixed": None}
    return collection.get(include=include, where=wheredoc[granularity])

def docCount(coll_name: str, granularity: granType) -> int:
    collection = _CHROMA_CLIENT.get_collection(coll_name)
    wheredoc:Final = {"topics" : {"sentence": "all"}, "sentences": {"sentence": {"$ne" : "all"}}, "mixed": None}
    return len(collection.get(include=[], where=wheredoc[granularity])["ids"])

def getBaseName(emb_name: str, granularity: granType):
    return f"{granularity}_{emb_name}"

def dictIter(conf: dict):
    for p in product(*conf.values()):
        yield(dict(zip(conf.keys(), p)))

def iterConfigs(argnames: List[str] | None = None):
    return dictIter(getArg(argnames))

def getModelFilePath(conf:dict[str, Any]):
    granularity = conf["granularity"]
    coll_name = getCollName(conf["embedder"])
    n_topics = conf["n_topics"]
    return f"models/{granularity}_{coll_name}_{n_topics}"

# __all__ = ["getCollBaseName", "getOpt", "getArg", "getDocs", "docCount", "granType", "dictIter", "iterConfigs", "getCollName"]
        