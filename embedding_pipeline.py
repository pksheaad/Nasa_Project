"""
CHROMADB EMBEDDING PIPELINE for NASA speace mission. text file only.

This text file read the parsed text data from various NASA space mission folders and create
a permanent CHROMADB  collection with openai embeddings for RAG application.
Optimize a process only text file to avoid duplication with json versions

Supported data source:
    - APOLLO 11 extracted data text only
    - APOLLO 13 extracted data text only
    - APOLLO 11 Textract extracted data text only
    - Challenger transcribed audioe adata text only
"""

# import dependencies
import os
import json
import logging
from pathlib import Path
import openai
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import hashlib
import time
from time import time
from datetime import datetime
import argparse
from typing import List, Dict, Tuple, Optional, Any

from openai.types import embedding
from openai.types.shared import metadata


# Create a logger to record the log information
logging.basicConfig(level=logging.INFO,
format = "%(asctime)s %(levelname)s %(message)s",
handlers= [
    logging.FileHandler("chroma_embedding_text_only.log"),
    logging.StreamHandler()
])

logger = logging.getLogger(__name__)
#

class ChromaEmbeddingPipelineTextOnly:
    """
    Pipeline to create CHROMADB with collection using openai emdebbings
    """
    def __init__(self,
                openai_api_key:str,
                chromadb_persist_directory:str = "chroma_db",
                collection_name: str = "nasa_space_missions_text",
                embedding_model: str = "text-embedding-3-small",
                chunk_size: int = 1000,
                chunk_overlap : int = 200
                 ) -> None:

        if not openai_api_key:
            raise ValueError("MISSING OPENAI API KEY")

        if chunk_size <= 0:
            raise ValueError("chunk size cant not be less than zero")

        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk overlap canot be less than zero and greater than chunk size")

        
        self.openai_api_key = openai_api_key
        self.chromadb_persist_directory = chromadb_persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        

        # create a client of openai
        self.client = OpenAI(api_key = openai_api_key)
        os.environ["CHROMA_OPENAI_API_KEY"] = openai_api_key

        # create embedding function instance
        self.embedding_function = OpenAIEmbeddingFunction(
            api_key = "CHROMA_OPENAI_API_KEY",
            model_name = embedding_model
        )

        # create chromadb instance
        self.chroma_client = chromadb.PersistentClient(
            path = chromadb_persist_directory,
            settings = Settings(anonymized_telemetry=False)
        )

        # create collection
        self.collection = self.chroma_client.get_or_create_collection(
                                name = collection_name, 
                                embedding_function = self.embedding_function,
                                metadata = {
                                    "discription" : "NASA space mission text chunks",
                                    "embedding_model" : embedding_model 
                                })
        # looger info
        logger.info(
            "Initialize collection %s in %s",
            collection_name, chromadb_persist_directory
        )

    def chunk(self, text:str, metadata: Dict[str,Any])->List[Tuple[str,Dict[str,Any]]]:
        """
        Split the text into overlapping character chunks. preferring natural boundaries
        """
        normalize = " ".join(text.split())
        # check whether the normalize is empty
        if not normalize:
            return []

        # Check whether chunking is even necessary
        if len(normalize) <= self.chunk_size:
            single_metadata = dict(metadata)
            single_metadata.update({
                "chunk_index" : 0,
                "chunk_start" : 0,
                "chunk_end" : len(normalize),
                "chunk_length" : len(normalize),
                "total_chunks" : 1
            })

            return [(normalize, single_metadata)]
        # Create an empty list for multiple chunks
        chunks: List[Tuple[str,Dict[str,Any]]] = []

        start = 0
        text_length = len(normalize)

        while start < text_length:
            target_end = min(start + self.chunk_size, text_length)
            end = target_end

            if target_end < text_length:
                search_floor = max(start + self.chunk_size//2, start+1)
                boundary_region = normalize[search_floor : target_end]
                candidates = [
                    boundary_region.rfind(". "),
                    boundary_region.rfind("\n"),
                    boundary_region.rfind("! "),
                    boundary_region.rfind( "? "),
                    boundary_region.rfind(" ")
                ]
                best = max(candidates)
                if best > 0:
                    end = search_floor + best + 1

            chunk = normalize[start:end].strip()
            if chunk:
                chunk_metadata = dict(metadata)
                chunk_metadata.update({
                        "chunk_index" : len(chunks),
                        "chunk_start" : start,
                        "chunk_end" : end,
                        "chunk_length" : len(chunk)
                    })
                chunks.append((chunk, chunk_metadata))

            if end >= text_length:
                break
            next_start = end - self.chunk_overlap     
        
            start = next_start if next_start > start else end

        total_chunk = len(chunks)
        for _, metadata in chunks:
                metadata['total_chunks'] = total_chunk

        return chunks

    def check_document_exists(self, doc_ids:str)->bool:
        """
        Check whether the docuemnt is exists into collection
        """
        try:
            result = self.collection.get(ids=[doc_ids])
            return bool(result.get("ids"))
        except Exception as e:
            logger.warning("Exception occure ducring fetching the document")
            logger.error("Error %s = ", str(e))
            return False

    def get_embedding(self, text:str)->List[float]:
        """
        Generate embedding for text
        """
        if not text:
            raise ValueError("Text not found for embeddings")

        try:
            resposne = self.client.embeddings.create(
                input = text.strip(),
                model = self.embedding_model
            )

            embeddings = resposne.data[0].embedding
            return embeddings

        except Exception as e:
            logger.info(f"Error occured during creation the embedding for text {text}")
            logger.error(f"Reason {str(e)}")
            raise

    def update_document(self, doc_id:str, text:str, metadatas:Dict[str, Any])->bool:
        """
        Update the existing document in collection
        Arguments:
            - doc_id(str) : Document id to update
            - text : New text need to be update
            - metadatas: new metadata
        """
        # create embeddings for new text:
        embeddings = self.get_embedding(text = text)

        # update the existing document
        if not doc_id:
            raise ValueError(f"Docuemnt id not found")
            try:
                self.collection.update(ids = [doc_id],
                                            embeddings = [embeddings],
                                            metadatas = [metadatas],
                                            documents=[text])
                logger.debug(f"{doc_id} updated successfully")
                return True
            except Exception as e:
                logger.error(f"Exception {str(e)} occured during updating the document {doc_id}")
                return False
    
    def delete_documents_by_source(self, source_pattern:str)->int:
        """
        Delete the number of docuemnts based on source pattern
        Arguments:
            source_patter(str) source patter which is available in metadata of documents on that basis delete the document
        Return
            int: Number of documents
        """
        # get all the documents
        documents = self.collection.get()

        doc_to_delete = []
        for i, metadata in enumerate(documents['metadatas']):
            if source_pattern in metadata.get("source", ""):
                doc_to_delete.append(documents['ids'][i])
        
        
        try:
            if doc_to_delete:
                self.collection.delete(ids = doc_to_delete)
                logger.info(f"Docuemnt{len(doc_to_delete)} has been deleted for source pattern: {source_pattern}")
                return len(doc_to_delete)

            else:
                logger.warning(f"No docuemnts found to delete for source pattern: {source_pattern}")
                return 0    
        
        except Exception as e:
            logger.error(f"Error {str(e)} occured during deleting the source pattern: {source_pattern}")
            return 0

    def extract_mission_from_path(self, file_path: Path)->str:
        """
        Extract the mission name from file path
        """   
        path_str = str(file_path).lower()
        if "apollo11" in path_str:
            return "apollo_11"
        elif "apollo13" in path_str:
            return "apollo_13"
        elif "challenger" in path_str:
            return "challenger"
        else:
            return "unknown"

    def get_file_documents(self, file_path:Path)->List[str]:
        """
        Get all the document ids for a specified file

        Arguments:
            -file_path(Path) file path
        Return:
            - List of all docuemnt ids
        """
        source = file_path.stem
        mission = self.extract_mission_from_path(file_path = file_path)

        try:
            all_docs = self.collection.get()
            file_doc_ids = []

            for i, metadata in enumerate(all_docs['metadatas']):
                if metadata.get("source") == source and  metadata.get("mission") == mission:
                    file_doc_ids.append(all_docs["ids"][i])
            
            return file_doc_ids

        except Exception as e:
            logger.error(f"Error {str(e)} occured during getting documents id for {file_path}")
            return []

    def generate_document_id(self, file_path:Path, metadata : Dict[str, Any])->str:
        """
        Generate a stable ID: mission_source_chunk_0001 plus a short path hash.
        """
        source = str(metadata.get("source",file_path.stem))
        mission = str(metadata.get("mission", self.extract_mission_from_path(file_path = file_path)))
        chunk_index = int(metadata.get("chunk_index", 0))

        def clean(value:str)->str:
            return "".join(ch if ch.isalnum() else "_" for ch in value.lower().strip("_"))
        
        path_hash = hashlib.sha1(str(file_path).encode("utf-8")).hexdigest()[:8]

        return f"{clean(mission)}_{clean(source)}_{path_hash}_chunk_{chunk_index:04d}"

    def extract_data_type_from_path(self, file_path:Path)->str:
        """ Extract data type from file path"""
        path_str = str(file_path).lower()

        if "transcript" in path_str:
            return "transcript"
        elif "textract" in path_str:
            return "textract extracted"
        elif "flight_plan" in path_str:
            return "flight_plan extracted"
        elif "audio" in path_str:
            return "audio extracted"
        else:
            return "document"

    def extract_document_category_from_filename(self, file_name:str)->str:
        """Extract document catagory from file_path"""
        file_name_lower = file_name.lower()

        # Apollo transcript types
        if "pao" in file_name_lower:
            return "public_affairs_officer"
        elif "cm" in file_name_lower:
            return "commnad_module"
        elif "tec" in file_name_lower:
            return "technical"
        elif "fligh_plan" in file_name_lower:
            return "filight plan"

        # challanger audio type
        elif "mission_audio" in file_name_lower:
            return "mission_audio"

        # Nasa archive transcript type:
        elif "19900066485" in file_name_lower:
            return "technical_report"
        elif "19710015566" in file_name_lower:
            return "mission_report"
        elif "ntrs" in file_name_lower:
            return "nasa_acrhive"

        # general type category
        elif "full_text" in file_name_lower:
            return "complete_document"
        else:
            return "general_document"

    def process_text_file(self, file_path:Path)->List[Tuple[str, Dict[str, Any]]]:
        """
        Process a plain text file with enhaced metadata extraction
        Arguments:
            -file_path(Path): file path
        Return:
            List of tuple with metadata information
        """
        try:
            with open (file = file_path, mode= "r", encoding="utf-8") as f:
                content = f.read()

            if not content:
                return []

            metadata = {
                "source" : file_path.stem,
                "file_path" : str(file_path),
                "file_type" : "text",
                "content_type" : "full text",
                "mission" : self.extract_mission_from_path(file_path = file_path),
                "data_type" : self.extract_data_type_from_path(file_path = file_path),
                "document_category" : self.extract_document_category_from_filename(file_name = file_path.name),
                "file_size" : len(content),
                "processed_timestamp" : datetime.now().isoformat()
            } 
            return self.chunk(text = content, metadata = metadata)

        
        except Exception as e:
            logger.error(f"Error: {str(e)} Occured during process the text file")
            return []

    def scan_text_files_only(self, base_path:str)->List[Path]:
        """
        Scan the base directory for text file only.
        Arguments:
            base_path: base path directory
        Return:
            List of filetred text file to process
        """
        base_path = Path(base_path)

        data_dirs= [
            "apollo11",
            "apollo13",
            "challenger"
        ]

        file_to_process = []

        for dir_name in data_dirs:
            dir_path = base_path/dir_name
            
            if dir_path.exists():
                logger.info(f"Scanning the files in {dir_path} ")

                # Procesing only txt file
                text_files = list(dir_path.glob("**/*.txt"))
                file_to_process.extend(text_files)
                logger.info(f"Total Number of files in directory {dir_path} is {len(file_to_process)}")

        # filetr out the unwanted file:
        filter_files = []

        for file_name  in file_to_process:
            if file_name.name.lower().startswith(".") or "summary" in file_name.name.lower() or file_name.suffix.lower()!=".txt":
                continue
            filter_files.append(file_name)

        logger.info(f"Total filtered file: {len(filter_files)}")

        # mission count
        mission_counts = {}

        for file_path in filter_files:
            mission_name = self.extract_mission_from_path(file_path = file_path)
            mission_counts[mission_name] = mission_counts.get(mission_name, 0) + 1

        logger.info("Mission Files")
        for mission, count in mission_counts.items():
            logger.info(f"Mission Name: {mission} Count: {count}")

        return filter_files

    def add_documents_to_collection(
        self, 
        documents:List[Tuple[str, Dict[str, Any]]],
        file_path:Path,
        batch_size:int = 50,
        update_mode:str = "skip" )->Dict[str, int]:
        """
        Add chunked documents to ChromaDB with skip/update/replace semantics.
        """
        if not documents:
            return {"skipped" : 0, "updated" : 0, "added" : 0}

        if update_mode not in ["skip", "replace", "update"]:
            raise ValueError(f"{update_mode} must be in ['skip', 'replace', 'update']")

        stats = {"added" : 0, "updated" : 0, "skipped" : 0}

        if documents and update_mode == "replace":
            existing_doc_id = self.get_file_documents(file_path = file_path)
            # deleting the file
            if existing_doc_id:
                self.collection.delete(ids = existing_doc_id)
                logger.info(f"Number of file deleted {len(existing_doc_id)} from {file_path.name}")

        for batch_start in range(0, len(documents), batch_size):
            batch = documents[batch_start:batch_start + batch_size]
            add_ids, add_metatadata, add_embeddings, add_docs = [],[],[],[]
            
            for text, metadata in batch:
                doc_ids = self.generate_document_id(file_path = file_path, metadata = metadata)
                exists = self.check_document_exists(doc_ids = doc_ids)

                if exists and update_mode =="skip":
                    stats["skipped"] +=1
                    continue

                if exists and update_mode == "update":
                    if self.update_document(doc_id=doc_ids, text = text, metadatas= metadata):
                        stats['updated'] +=1
                    else:
                        logger.error("Failed to update existing document: %s", doc_ids)
                    continue
                embedding = self.get_embedding(text=text)    

                add_ids.append(doc_ids)
                add_metatadata.append(metadata)
                add_embeddings.append(embedding)
                add_docs.append(text)

            if add_ids:
                self.collection.add(
                        ids = add_ids,
                        embeddings = add_embeddings,
                        metadatas = add_metatadata,
                        documents = add_docs
                    )
                stats["added"] += len(add_ids)
                logger.info("Added batch of %d chunks from %s", len(add_ids), file_path.name)

        return stats

    def process_all_text_data(self, base_path:str, update_mode:str="skip", batch_size:int = 50)->Dict[str, Any]:
        """
        Process all supported text files and aggregate ingestion statistics.
        """
        stats = {
            'files_processed': 0,
            'documents_added': 0,
            'documents_updated': 0,
            'documents_skipped': 0,
            'errors': 0,
            'total_chunks': 0,
            'missions': {}
        }

        files = self.scan_text_files_only(base_path = base_path )
        for file_path in files:
            mission = self.extract_mission_from_path(file_path = file_path)
            mission_stats = stats["missions"].setdefault(
                mission, 
                {"files" : 0, "added" : 0,"updated" : 0, "skipped" : 0, "chunks" : 0}
            )
            try:
                chunks = self.process_text_file(file_path = file_path)
                file_stats = self.add_documents_to_collection(documents = chunks,file_path = file_path, batch_size=batch_size, update_mode = update_mode)

                stats["files_processed"] += 1
                stats["documents_added"] += file_stats['added']
                stats["documents_updated"] += file_stats['updated']
                stats["documents_skipped"] += file_stats['skipped']
                stats["total_chunks"] += len(chunks)

                mission_stats['files'] += 1
                mission_stats['added'] += file_stats['added']
                mission_stats["updated"] += file_stats['updated']
                mission_stats["skipped"] += file_stats['skipped']
                mission_stats["chunks"] += len(chunks)

            
            except Exception as e:
                stats["errors"] += 1
                logger.exception(f"Exception {str(e)} occured during process the file from {base_path}")

        return stats

    def get_collection_info(self)->Dict[str, Any]:
        try:
            return{
                "Collection Name" : self.collection.name,
                "Document Count" : self.collection.count(),
                "metadata" : self.collection.metadata,
                "persistant_directory" : self.chromadb_persist_directory,
                "embedding_model" : self.embedding_model
            }
        except Exception as e:
            logger.exception(f"Exception: {str(e)} occured during fetching collection info")   
            return {
                "error" : str(e)
            }

    def query_collection(self, query_text:str, n_results = 5)->Dict[str, Any]:
        if not query_text:
            raise ValueError("Please provide the query")

        try:
            results = self.collection.query(
                query_texts = [query_text.strip()],
                n_results = max(1, n_results),
                include = ["documents", "metadatas","distances"]
            )

            return results

        except Exception as e:
            logger.error(f"Exception {str(e)} occured during fetching the query: {query_text}")
            raise

    def get_collection_stats(self)->Dict[str, Any]:
        """Get detailed statistics about the collection"""
        try:
            all_docs = self.collection.get()

            metadatas = all_docs['metadatas']
            if not metadatas:
                return {
                    "error" : "No documents in the collection"
                }
            
            stats = {
                'total_documents': len(all_docs['metadatas']),
                'missions': {},
                'data_types': {},
                'document_categories': {},
                'file_types': {}
            }

            for metadata in metadatas:
                mission = metadata.get("mission", "unknown")
                data_type = metadata.get("data_type", "unknown")
                document_category = metadata.get("document_category","unknown")
                file_type = metadata.get("file_type", "unknown")

                stats["missions"][mission] = stats["missions"].get(mission,0) + 1
                stats["data_types"][data_type] = stats["data_types"].get(data_type,0) + 1
                stats["document_categories"][document_category] = stats["document_categories"].get(document_category,0) + 1
                stats["file_types"][file_type] = stats["file_types"].get(file_type,0) + 1
            
            return stats

        except Exception as e:
            logger.exception("Exception occured getting the colection Stats")
            return {
                "error" : str(e)
            }

# Create Main Function
def main():
    """MAIN FUNCTION"""
    parser = argparse.ArgumentParser(description = "ChromaDB Embedding Pipeline for NASA Data")
    parser.add_argument("--data_path", default = ".", help="Path to data directory")
    parser.add_argument("--api_key", required = True, help = "OPEN AI API KEY")
    parser.add_argument("--chroma_dir", default= "./chroma_db_openai", help = "ChromaDB persist directory")
    parser.add_argument("--collection_name", default = "nasa_space_missions_text", help = "Collection name")
    parser.add_argument("--embedding_model", default="text-embedding-3-small", help = "OPENAI EMBEDDING MODEL")
    parser.add_argument("--chunk_size", type = int, default = 500, help = "size of chunks")
    parser.add_argument("--chunk_overlap", type = int, default = 100, help = "chunk overlap size")
    parser.add_argument("--batch_size", type = int, default = 50, help = "Batch size for processing")
    parser.add_argument("--update_mode", choices=['skip', 'replace', 'update'], default = 'skip', 
                        help = "How to handle existing documents: skip, update, or replace" )
    parser.add_argument("--test_query", help = "Test query after processing")
    parser.add_argument("--stats_only", action="store_true", help = "Only show collection statistics") 
    parser.add_argument("--delete_source", help = "Delete all documents from a specific source pattern") 
    args = parser.parse_args()

    # Initializing the pipeline
    logger.info("Initializing ChromaDB Embedding Pipeline...")          
    pipeline = ChromaEmbeddingPipelineTextOnly(
        openai_api_key = args.api_key,
        chromadb_persist_directory = args.chroma_dir,
        collection_name = args.collection_name,
        embedding_model = args.embedding_model,
        chunk_size = args.chunk_size,
        chunk_overlap = args.chunk_overlap
        
    )

    if args.delete_source:
        deleted_count = pipeline.delete_documents_by_source(source_pattern = args.delete_source)
        logger.info(f"Number of document deleted: {deleted_count} for source: {args.delete_source}")
        #return
        

    ## If stats only, show collection statistics and exit
    if args.stats_only:
        logger.info("Collection_Statistics")
        stats = pipeline.get_collection_stats()
        for key, value in stats.items():
            logger.info(f"{key} : {value}")
        return

    # # Process all data
    logger.info(f"Starting text data processing with update model: {args.update_mode}")
    start_time = time()
    stats = pipeline.process_all_text_data(base_path = args.data_path, update_mode = args.update_mode, batch_size = args.batch_size)

    end_time = time()
    processing_time = round(end_time - start_time, 4)
    logger.info("="*60)
    logger.info("PROCESSING COMPLETE")
    logger.info("="*60)
    logger.info(f"File Processed: {stats['files_processed']}")
    logger.info(f"Docuemnt Added: {stats['documents_added']}")
    logger.info(f"Docuemnt Updated: {stats['documents_updated']}")
    logger.info(f"Docuemnt Skipped: {stats['documents_skipped']}")
    logger.info(f"Total Chunks: {stats['total_chunks']}")
    logger.info(f"Error: {stats['errors']}")
    logger.info(f"Prcessing Time: {processing_time} seconds")

    logger.info("MISSION INFO")
    for mission, mission_stats in stats['missions'].items():
        logger.info(f"{mission}: {mission_stats['files']} files {mission_stats['chunks']} chunks")
        logger.info(f"Added: {mission_stats['added']}, Updated: {mission_stats['updated']}, Skipped: {mission_stats['skipped']} ")

    # Test query if provided
    if args.test_query:
        logger.info(f"\n Testing query for: {args.test_query}")
        results = pipeline.query_collection(query_text = args.test_query)
        if results and 'documents' in results:
            logger.info(f"Total Number of Documents is results: {len(results['documents'][0])}")

            for i, doc in enumerate(results["documents"][0][:3],1): # show top three
                logger.info(f"Result{i} : {doc[:200]}")
    
    logger.info("Pipeline completed successfully")


if __name__ =="__main__":
    main()




        


 
                





        


 


