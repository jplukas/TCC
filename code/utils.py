import json
from typing import Any, List, Literal, Final, TypeAlias, get_args
from itertools import product
import chromadb

granType:TypeAlias = str

_CHROMA_CLIENT:Final = chromadb.PersistentClient("isidb")

dataframe_path = "./topic_modeling_results_df.pkl"
selected_confs_json_path = "./selected_confs.json"

class OPTSObj:
    def __init__(self, optsfile):
        self.data = json.load(optsfile)

    def __getitem__(self, item):
        if type(item) is tuple:
            if item[1] == "coll_name":
                return getColnames()
            return self.data[item[0]][item[1]]
        
        return self.data[item]

with open('opts.json', 'r') as optsfile:
    _OPTS = OPTSObj(optsfile)

def getCollBaseName(emb_name: str) -> str:
    return emb_name.replace('-', '_').replace(':', '_').replace('/', '_')

def getCollName(emb_name: str, edition: str = "en") -> str:
    return getCollBaseName(emb_name) + "__" + edition

def getOpt(optName: str):
    return _OPTS[optName]

def getArg(argname: str | List[str] | None = None):
    if argname is None:
        collections = getColnames()
        return {**_OPTS["args"], "collections": collections if len(collections) > 0 else None}
    if isinstance(argname, str): return _OPTS["args", argname]
    return {k: _OPTS["args", k] for k in argname}

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
    return dictIter({k: dictIter(v) if isinstance(v, dict) else v for k, v in getArg(argnames).items()})

def getModelFilePath(conf:dict[str, Any]):
    baseFilePath = getBaseFilePath(conf)
    n_topics = conf["n_topics"]
    clustering = conf["clustering"]
    min_cluster_size = clustering["min_cluster_size"]
    min_samples = clustering["min_samples"]
    return f"./models/{baseFilePath}_{min_cluster_size}_{min_samples}_{n_topics}"

def getReductionFilePath(conf:dict[str, Any]):
    baseFilePath = getBaseFilePath(conf)
    return f"./reduced_embds/{baseFilePath}.npy"

def getHTMLFilePath(conf:dict[str, Any]):
    modelFilePath = getModelFilePath(conf)
    filename = modelFilePath.removeprefix("./models/")
    return f"./visualizations/html/{filename}.html"


def getBaseFilePath(conf:dict[str, Any]):
    granularity = conf["granularity"]
    coll_name = getCollName(conf["embedder"]) if "coll_name" not in conf else conf["coll_name"]
    reduction = conf["reduction"]
    n_neighbors = reduction["n_neighbors"]
    dimensionality = reduction["n_components"]
    densmap = "dense" if reduction["densmap"] else "normal"
    return f"{granularity}_{coll_name}_{n_neighbors}_{dimensionality}_{densmap}"

def getCollections(embedders:List[str]=None):
    if embedders is None: embedders = getArg("embedder")
    return {embedder: getCollection(embedder) for embedder in embedders}

def getColnames():
    return [col.name for col in getCollections().keys()]

def getCollection(embedder):
    colname = getCollName(embedder)
    return _CHROMA_CLIENT.get_collection(name=colname)

def createIfNotExist():
    pass

def getClient():
    return _CHROMA_CLIENT

# __all__ = ["getCollBaseName", "getOpt", "getArg", "getDocs", "docCount", "granType", "dictIter", "iterConfigs", "getCollName"]
        