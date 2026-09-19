"""Batch evaluation for the NASA RAG project.

Loads evaluation_dataset.txt, runs retrieval + generation for every test case,
computes per-question metrics, and writes aggregate mean/distribution summaries.
"""

import argparse
import json
import os
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List
from dotenv import load_dotenv
from openai.types import embedding
import pandas as pd


# open ai
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


from ragas import EvaluationDataset,evaluate,SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    SemanticSimilarity
)
from sqlalchemy.sql.functions import aggregate_strings

# LLM/RAG/RAGA classes
import rag_client
import llm_client

# ==========================================================
# STEP 1 - Load evaluation_dataset.txt
# ==========================================================

def load_evaluation_dataset(dataset_path:str)->List[Dict[str,str]]:
    """Load JSON-lines evaluation records and validate required fields."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

    required = {"id" , "mission", "category", "question", "expected_answer"}

    records: List[Dict[str, str]] = [] # finally return

    with path.open("r", encoding="utf-8") as handler:
        for line_number, raw_line in enumerate(handler, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid Json file at line number {line_number} : {e}")

            missing = required.difference(item)
            if missing:
                raise ValueError(f"Dataset line {line_number} has missing field: {sorted(missing)}")
            if any (not(str(item[field])).strip() for field in required):
                raise ValueError(f"Dataset line {line_number} contains empty values")

            records.append(item)

    if len(records) < 5:
        raise ValueError("Number of Evalation dataset can not be less than 5")
        
    return records

# ==========================================================
# STEP 2 - Run RAG pipeline and create SingleTurnSample
# ==========================================================
def build_ragas_dataset(test_records:List[Dict[str, str]], collection, api_key:str, model:str, n_results:int = 3):
    
    samples = []
    metadata_rows = []

    for index, item in enumerate(test_records, 1):
        id = item['id'].strip()
        question = item['question'].strip()
        category = item['category'].strip()
        mission = item['mission'].strip()
        reference = (item['expected_answer'].strip())

        print(f"\n [Index {index} / {len(test_records)}] ")
        print(f"id: {id}")
        print(f"Question: {question}")
        print(f"Reference: {reference}")
        print(f"Category: {category}")
        print(f"Mission: {mission}")

        # ----------------------------------------
        # Retrieve documents from ChromaDB
        # ----------------------------------------

        doc_results = (rag_client.retrieve_documents(
            collection = collection,
            query = question,
            n_results = n_results,
            mission_filter = mission )
            )

        documents = []
        metadatas = []
        
        if doc_results:
            documents_group = doc_results.get('documents') or []
            metadatas_group = doc_results.get('metadatas') or []

            if documents_group:
                documents = documents_group[0] or []
        
            if metadatas_group:
                metadatas = metadatas_group[0] or []

        if not documents:
            print("No Context found")

        # ----------------------------------------
        # Format retrieved context
        # ----------------------------------------

        context = rag_client.format_context(documents = documents, metadatas = metadatas)

        # ----------------------------------------
        # Generate answer from LLM
        # ----------------------------------------
        generated_answer = llm_client.generate_response(
            openai_key = api_key,
            context = context,
            user_message = question,
            conversation_history = [],
            model = model
        ) 

        # ----------------------------------------
        # Create RAGAS SingleTurnSample
        # ----------------------------------------
        sample = SingleTurnSample(
            user_input = question,
            retrieved_contexts = documents,
            response = generated_answer,
            reference = reference
        )
        samples.append(sample)

        # Keep our project-specific information
        metadata_rows.append({
            "id" : id,
            "question" : question,
            "expected_answer" : reference,
            "category" : category,
            "mission" : mission,
            "generated_answer" : generated_answer
        }    
        )
    # --------------------------------------------
    # Create RAGAS EvaluationDataset
    # --------------------------------------------
    eval_data = EvaluationDataset(
        samples = samples
    )

    return (eval_data, metadata_rows)

# ==========================================================
# STEP 3 - Create RAGAS metrics
# ==========================================================

def create_metrics(api_key: str):
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(api_key = api_key, model = "gpt-3.5-turbo", temperature = 0)
    )

    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(api_key = api_key, model = "text-embedding-3-small"))

    # creating metrics
    metrics = [
        Faithfulness(llm = evaluator_llm),
        ResponseRelevancy(llm = evaluator_llm, embeddings = evaluator_embeddings),
        SemanticSimilarity(embeddings = evaluator_embeddings)
    ]

    return metrics

# ==========================================================
# STEP 4 - Batch evaluation using ragas.evaluate()
# ==========================================================
def evaluate_batch(api_key:str, eval_data : EvaluationDataset, metadata_rows:List[Dict[str, str]]):
    metrics = create_metrics(api_key = api_key)
    print("\n Running raga batch evaluation")

    # Evaluate the entire dataset
    results = evaluate(
        dataset = eval_data,
        metrics = metrics
    ) 

    #---------------------------------------
    # Convert Raga Result to Dataframe
    #---------------------------------------

    results_df = (results.to_pandas().reset_index(drop = True))
    metadata_df = pd.DataFrame(metadata_rows).reset_index(drop = True)

    if len(results_df) != len(metadata_df):
        print("RAGA Resulu count does not match with EValuation dataset count")

    # ----------------------------------------
    # Determine metric columns
    # ----------------------------------------
    non_metric_columns = {

        "user_input",
        "retrieved_contexts",
        "response",
        "reference",
        "reference_contexts",
        "rubrics"
    }

    metric_columns = [
        column for column in results_df.columns if column not in non_metric_columns
    ]

    # ----------------------------------------
    # Combine project metadata + scores
    # ----------------------------------------
    per_question_df = pd.concat([metadata_df, results_df[metric_columns]], axis = 1)

    # ----------------------------------------
    # Find numeric metric columns
    # ----------------------------------------

    numeric_metrics = [column  for column in metric_columns if pd.api.types.is_numeric_dtype(per_question_df[column]) ]

    # ----------------------------------------
    # Calculate aggregate statistics
    # ----------------------------------------
    aggregate_rows = []
    for metric_name in numeric_metrics:
        values = pd.to_numeric(per_question_df[metric_name], errors = "coerce").dropna()
        aggregate_rows.append({
            "metric_name" : metric_name,
            "min" : round(float(values.min()), 4),
            "max" : round(float(values.max()),4),
            "mean" : round(float(values.mean()), 4),
            "std_dev": round(float(values.std(ddof = 0)),4),
            "count" : values.count()
        })

    aggregate_df = pd.DataFrame(aggregate_rows)

    return (per_question_df, aggregate_df)

# ==========================================================
# STEP 5 - Save evaluation report
# ==========================================================

def save_results(per_question_df, aggregate_df, output_file):
    report = {
        "per_question_results" : per_question_df.to_dict(orient = "records"),
        "aggregate_metrices" : aggregate_df.to_dict(orient = "records")
    }

    with open(output_file, "w", encoding = "utf-8") as f:
        json.dump(report, f, indent = 2, ensure_ascii = False )

    print(f"Evaluation report stored in {output_file}")

#######################################################################
# Main
#######################################################################
def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description = ("RAG system using RAGAS", "Batch Evaluation of NASA"))
    parser.add_argument("--api_key", required = True, help = "OPEN AI API KEY")
    parser.add_argument("--data_set", default = "evaluation_dataset.txt" , help = "Evaluation dataset")
    parser.add_argument("--chroma_dir", default ="./chroma_db_openai", help = "Chroma DB directory")
    parser.add_argument("--collection_name", default = "nasa_space_missions_text", help = "Collection Name")
    parser.add_argument("--model", default = "gpt-3.5-turbo" , help = "OPENAI GPT Model")
    parser.add_argument("--n_results", type = int, default = 3, help = "Number of results set")
    parser.add_argument("--output", default = "batch_evaluation_results.json", help = "Evaluation Report JSON file")

    args = parser.parse_args()

    # --------------------------------------------------
    # Make command-line API key available to all modules
    # --------------------------------------------------
    os.environ["OPENAI_API_KEY"] = args.api_key
    os.environ["CHROMA_OPENAI_API_KEY"] = args.api_key

    # ----------------------------------------
    # Load test dataset
    # ----------------------------------------
    test_record = load_evaluation_dataset(dataset_path = args.data_set)
    print("="*70)
    print(f"Question Loaded: {len(test_record)}")
    print(f"Categories: {[item['category'] for item in test_record]}")

    # ----------------------------------------
    # Initialize ChromaDB
    # ----------------------------------------
    collections, success, error = rag_client.initialize_rag_system(
        chroma_dir = args.chroma_dir,
        collection_name = args.collection_name
    )

    if success:
        print("RAG INITIALIZE")
    else:
        raise ValueError("Unable to initialize the RAG System")

    # ----------------------------------------
    # Build RAGAS EvaluationDataset
    # ----------------------------------------
    eval_data, metadata_rows = build_ragas_dataset(
        test_records = test_record,
        collection = collections,
        api_key = args.api_key,
        model = args.model,
        n_results = args.n_results
    )

    print("RAGAS Evaluation dataset crested successfully")
    print(f"Total Samples: {len(eval_data)}")

    # ----------------------------------------
    # Batch RAGAS evaluation
    # ----------------------------------------
    per_question_df, aggregate_df = evaluate_batch(
                                api_key = args.api_key, 
                                eval_data = eval_data, 
                                metadata_rows = metadata_rows)

    # Display per question results
    print("\n"+ "="*70)
    print("PER QUESTION METRIC SUMMARY\n")
    print(per_question_df.to_string(index = False))

    # Display Aggregate Summary
    print("="*70)
    print("AGGREGATE METRICS SUMMARY")
    print(aggregate_df.to_string(index = False))

    # Save the results
    save_results(per_question_df = per_question_df, aggregate_df = aggregate_df, output_file = args.output)

    print("*"*70)
    print("BATCH EVALUATION COMPLETE")





if __name__ == "__main__":
    main()

        
        




            



