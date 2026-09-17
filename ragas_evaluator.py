import os
from typing import List, Dict
from dill import temp
from dotenv import load_dotenv

from ragas import llms
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


try:
    from ragas import SingleTurnSample
    from ragas.metrics import ResponseRelevancy, Faithfulness
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

load_dotenv()

def evaluate_response_quality(question:str, answer:str,context:List[str])->Dict[str, float]:
    """
    Evaluate one RAG answer with RAGAS faithfulness and response relevancy.

    These two metrics do not require a human reference answer, making them suitable
    for real-time evaluation in the Streamlit chat application.
    """
    if not RAGAS_AVAILABLE:
        return {"error" : "RAGAS is not avaialable"}

    if not question or not answer:
        return {
            "error" : "Question and answer is require for Evaluation"
        }
    
    if not context:
        return {
            "error" : "Atleast one context is required for Evaluation"
        }

    if not os.getenv(key = "OPENAI_API_KEY"):
        return {
            "error" : "OPENA AI API key not found"
        }

    try:

        # LangchainLLMWrapper wrapping
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model = "gpt-3.5-turbo" , temperature = 0))

        #LangchainEmbeddingsWrapper wrapping
        evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model = "text-embedding-3-small"))

        # Create sample of SingleTurnSample
        sample = SingleTurnSample(
            user_input = question,
            retrieved_contexts = context,
            response = answer
        )

        # evaluator
        metrics = {
            "faithfulness" : Faithfulness(llm = evaluator_llm),
            "response_relevancy" : ResponseRelevancy(llm = evaluator_llm, embeddings = evaluator_embeddings)
        }

        # create a score dictionary
        scores:Dict["str", float] = {}

        for name, metric in metrics.items():
            score = metric.single_turn_score(sample)
            scores[name] = round(float(score), 4)

        scores["overall_score"] = round(sum(scores.values())/len(scores), 4)
        return scores

    except Exception as e:
        return {"error" : f"RAGAS Evaluation failed due to {str(e)}"}


    