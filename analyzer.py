# analyzer.py

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
import os

# ==== Configs ====

VECTORSTORE_PATH = "vectorstores"
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_LLM_REPO = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# ==== Helpers ====

def format_example(example: dict) -> str:
    """Format a few-shot example for the prompt."""
    return f"Q: {example['query']}\nA: {example['answer']}"

def get_vectorstore_for_system(system_type: str):
    """Load FAISS vectorstore for a specific system (PV, Chiller, etc.)."""
    system_path = os.path.join(VECTORSTORE_PATH, system_type.lower())
    embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
    return FAISS.load_local(system_path, embeddings, allow_dangerous_deserialization=True)

def get_llm_model():
    """Load the language model from HuggingFace."""
    return HuggingFaceEndpoint(repo_id=HF_LLM_REPO, temperature=0.6)

# ==== Few-Shot Chain Builder ====

def build_fewshot_chain(user_query: str, system_type: str) -> LLMChain:
    """Construct a few-shot chain using examples from the vector store."""
    vectorstore = get_vectorstore_for_system(system_type)
    similar_examples = vectorstore.similarity_search(user_query, k=5)
    
    # Extract question-answer pairs
    examples = []
    for doc in similar_examples:
        qna = doc.page_content.split("\n")
        query_line = next((line for line in qna if line.startswith("Q:")), None)
        answer_line = next((line for line in qna if line.startswith("A:")), None)
        if query_line and answer_line:
            examples.append({
                "query": query_line.replace("Q: ", ""),
                "answer": answer_line.replace("A: ", "")
            })

    # Build few-shot prompt
    example_prompt = PromptTemplate.from_template("Q: {query}\nA: {answer}")
    main_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="You are an expert diagnostic assistant for smart facilities.\nUse past cases to respond to the query.\n",
        suffix="Q: {query}\nA:",
        input_variables=["query"],
        example_separator="\n\n",
        format_example_fn=format_example
    )

    return LLMChain(llm=get_llm_model(), prompt=main_prompt)

# ==== CSV Fault Analysis ====

def analyze_csv(file_path: str, system_type: str) -> str:
    """Perform rule-based anomaly detection from the uploaded CSV file."""
    df = pd.read_csv(file_path)
    
    if system_type.lower() == "pv":
        issues = []
        for idx, row in df.iterrows():
            gen_ratio = row['energy_generated_kWh'] / (row['expected_generation_kWh'] + 1e-5)
            if gen_ratio < 0.75:
                issues.append(f"[{row['datetime']}] Low generation: {gen_ratio:.2f} (Expected {row['expected_generation_kWh']} vs Actual {row['energy_generated_kWh']})")
            if row['inverter_efficiency_percent'] < 90:
                issues.append(f"[{row['datetime']}] Low inverter efficiency: {row['inverter_efficiency_percent']}%")
            if row['module_temperature_C'] > 50:
                issues.append(f"[{row['datetime']}] High panel temperature: {row['module_temperature_C']}°C")

        return "\n".join(issues) if issues else "✅ No anomalies detected in PV data."

    # Placeholder for other systems
    return f"⚠️ Analysis not yet implemented for {system_type}."

