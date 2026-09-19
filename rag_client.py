""" This file will discover local Chroma DB, Open the selected collection, 
retrieve the semantically relevant chnucks and turn those chucks into prompt ready text
"""
# importing dependencies
import os
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

def _openai_embedding_function()->OpenAIEmbeddingFunction:
    """Create the embedding function used by both indexing and query-time retrieval."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")
    
    os.environ["CHROMA_OPENAI_API_KEY"] = api_key

    return OpenAIEmbeddingFunction(
        api_key_env_var="CHROMA_OPENAI_API_KEY",
        model_name = "text-embedding-3-small"
    )

def discover_chroma_backends()->Dict[str, Dict[str, str]]:
    """ Discover the local chroma db collection in the Project Directory"""
    back_end: Dict[str, Dict[str, str]] = {}
    current_dir = Path(".")
    
    candidates_dir = sorted(path for path in current_dir.iterdir() if path.is_dir() and path.name.lower().startswith("chroma"))

    for directory in candidates_dir:
        try:
            client = chromadb.PersistentClient(
                path = str(directory),
                settings = Settings(anonymized_telemetry=False)
            )

            # get all collections
            collections = client.list_collections()
            
            for collection_info in collections:
                collection_name = (collection_info if isinstance(collection_info, str) else collection_info.name)

                print(f"Colelction Name: {collection_name}")
                key = f"{directory.name}::{collection_name}"
                try:
                    collection = client.get_collection(collection_name)
                    document_count = collection.count()
                    print(f"Total Document Chunks Found for collection {collection_name} = {document_count} ")
                except Exception as e:
                    document_count = "N/A"

                back_end[key] = {
                    "directory" : str(directory),
                    "collection_name" : collection_name,
                    "display_name" : f"{collection_name} ({directory.name}, {document_count} chunks)",
                    "document_count" : str(document_count)
                }

        except Exception as e:
            key = f"{directory.name}:: 'error'"
            back_end[key] ={
                "directory" : str(directory),
                "collection_name" : "",
                "display_name" : f"{directory.name} : 'error' = str(e)",
                "document_count" : "N/A"
            }
    return back_end

def initialize_rag_system(chroma_dir:str, collection_name:str):
    """Initialize a persistant chromaDB collection for Semantic Retieval"""
    try:
        cleint = chromadb.PersistentClient(
            path = chroma_dir,
            settings= Settings(anonymized_telemetry=False)
        )
        return cleint.get_collection(name = collection_name, embedding_function=_openai_embedding_function()), True, None
    
    except Exception as e:
        return None, False, str(e)
    
def retrieve_documents(
    collection,
    query:str,
    n_results: int = 3,
    mission_filter : Optional [str] = None)->Optional[Dict]:

    """Retrieve semantically relevant documents with optional mission filtering."""

    if not query or not query.strip():
        return None
    
    where_filter = None

    if mission_filter and mission_filter.strip().lower() not in {"all", "any", "none"}:
        where_filter = {"mission" : mission_filter.lower().strip().replace(" ", "_")}

    query_kwargs = {
        "query_texts" : [query.strip()],
        "n_results" : max(1, n_results),
        "include" : ["documents", "distances", "metadatas"]
    }

    if where_filter is not None:
        query_kwargs['where'] = where_filter
    
    return collection.query(**query_kwargs)

def format_context(documents:List[str], metadatas: List[Dict])->str:
    """Format retrieved chunks into a readable, source-labelled context block."""
    if not documents:
        return ""

    context_parts = ["NASA DOCUMENT CONTEXT"]
    safe_metadatas = metadatas or [{} for _ in documents]

    for index, (document, metadata) in enumerate(zip(documents, safe_metadatas), start = 1):
        metadata = metadata or {}
        mission = str(metadata.get("mission","unknown")).replace("_", " ").title()
        source = str(metadata.get("source", "unknown source"))
        category = str(metadata.get("document_category", "general document")).replace("_"," ").title()

        context_parts.append(f"\n[Index {index}] | Mission: {mission} | Source: {source} | Caregory: {category}")

        clean_document = (document or "").strip()
        if len(clean_document) > 4000: # chunking
            clean_document = clean_document[:4000].rstrip() + "...[truncated]"
        context_parts.append(clean_document)

    return "\n".join(context_parts)










