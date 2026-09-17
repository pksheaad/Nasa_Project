""" Responsibility: Transform a user question plus retrive evidence into a safe,
readable, context grouded answer. This module should not perform retieval itself,
seperation of concern makes the system easier to test and replace
"""
# Import dependecies
from typing import List, Dict
from openai import OpenAI
import os
from dotenv import load_dotenv

def generate_response(
    openai_key : str, # Open AI API key
    user_message : str, # User QUestion
    context : str, # Evidence return by RAG System
    conversation_history : List[Dict], # prior user / assitance historical information
    model : str = "gpt-3.5-turbo"
)->str:

    """
    Generate the NASA grounded response using RAG system
    Arguments:
        -openai_key:str OpenAI api key
        -user_message:str Current user question
        -context:str Context recieved from ARG system
        -conversation_history: List[Dict] Historical chat information as list of Dictionaries. Role and Content
        -model : str model used to resposne defualt is gpt-3.5-turbo
    """

    # check api
    if not openai_key:
        raise ValueError("API_key not found")
    
    # check for user_message
    if not user_message or not user_message.strip():
        raise ValueError("User Question can not be empty")

    # create System prompt
    system_prompt =(
        "You are a careful NASA space-mission expert and research assistant. "
        "Answer questions primarily from the retrieved NASA context provided to you. "
        "Be accurate, concise, and educational. When the supplied context contains the "
        "answer, ground your response in it and mention the relevant mission/source when useful. "
        "If the context is missing, incomplete, or does not support a claim, clearly say that the "
        "available project documents do not provide enough evidence rather than inventing facts. "
        "Distinguish retrieved evidence from general background knowledge. Do not fabricate dates, "
        "crew names, quotations, telemetry, causes, or mission events."
    ) 

    context_message = (
        "Retrived NASA Context \n"
        f"{context if context and  context.strip() else '[No Relevant Information Recieved]'}"
        )

    # Message construction 
    messages : List[Dict[str, str]] = [
        {
         "role" : "system",
         "content" : system_prompt
        },
        {
            "role" : "system",
            "content" : context_message
        }
        ]

    # history construction
    for message in (conversation_history or [])[-10:]:
        role = message.get("role")
        content = message.get("content")

        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            messages.append({"role" : role, "content" : content})

        # add the final user question
    messages.append({"role" : "user", "content" : user_message.strip()})

    # create open_ai client
    client = OpenAI(api_key = openai_key)
    complition = client.chat.completions.create(
        model = model, 
        messages = messages,
        temperature = 0.20,
        max_tokens = 800
    )
    resposne = complition.choices[0].message.content.strip()

    return resposne


            
