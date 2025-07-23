import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
from langchain.agents import initialize_agent, Tool
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
import pandas as pd
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv(override=True)
api_key = os.getenv("GOOGLE_API_KEY")

# === LLM Setup ===
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0.1
)

# === CSV Fault Analysis Tool ===
def analyze_csv(file_path: str):
    df = pd.read_csv(file_path)
    suggestions = []
    for index, row in df.iterrows():
        if row['energy_generated_kWh'] < 0.85 * row['expected_generation_kWh']:
            suggestions.append(f"⚠️ **{row['datetime']}** - Underperformance in system `{row['system_id']}`.\n- Consider checking **panel cleaning** or **shading**.")
        if row['inverter_efficiency_percent'] < 95:
            suggestions.append(f"⚠️ **{row['datetime']}** - Low inverter efficiency in `{row['system_id']}`.\n- Suggest inspecting **inverter wiring**.")
        if row['module_temperature_C'] > 45:
            suggestions.append(f"⚠️ **{row['datetime']}** - High module temperature in `{row['system_id']}`.\n- Possible **cooling issue** or **over-irradiance**.")
    return "\n\n".join(suggestions)

csv_analysis_tool = Tool(
    name="CSV Fault Detector",
    func=lambda _: analyze_csv("pv_clean_data.csv"),
    description="Detects faults and anomalies from PV system data CSV file"
)

# === Agent Setup ===
tools = [csv_analysis_tool, PythonREPLTool()]
agent = initialize_agent(
    tools, llm, agent="zero-shot-react-description", verbose=False
)

# === Vector DB (Few-Shot Prompt Embeddings) ===
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
example_docs = [
    Document(page_content="System XYZ has module temperature of 48°C and inverter efficiency of 93%.",
             metadata={"output": "⚠️ High module temperature. Consider cooling inspection.\n⚠️ Inverter underperformance. Check connections."}),
    Document(page_content="System ABC generated 4.2 kWh, but expected 5.5 kWh. Inverter efficiency is 97%.",
             metadata={"output": "⚠️ Underperformance detected. Possible soiling or partial shading. Inverter looks fine."}),
]

vector_db = Chroma.from_documents(example_docs, embedding=embedding_model, collection_name="fewshot_examples")

def get_fewshot_examples(query):
    results = vector_db.similarity_search(query, k=2)
    return [{"input": r.page_content, "output": r.metadata["output"]} for r in results]

# === Prompt Construction ===
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])

# === Streamlit UI ===
def main():
    st.set_page_config(page_title="PV Fault Analyzer", layout="wide")
    st.title("☀️ Autonomous PV Fault Analyzer (LangChain + Gemini)")

    uploaded_file = st.file_uploader("📤 Upload PV Sensor CSV File", type="csv")
    query_input = st.text_area("💬 Describe scenario or paste a system status log:")

    if uploaded_file:
        with open("pv_clean_data.csv", "wb") as f:
            f.write(uploaded_file.read())
        st.success("✅ CSV uploaded successfully.")

        if st.button("🧠 Run Agent Analysis"):
            with st.spinner("Analyzing CSV for faults and suggestions..."):
                result = agent.run("Analyze the PV system CSV and suggest optimizations and actions.")
                time.sleep(0.5)
            st.subheader("🔍 Agent Suggestions")
            st.markdown(result)

    if query_input:
        examples = get_fewshot_examples(query_input)
        fewshot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples
        )
        prompt_chain = ChatPromptTemplate.from_messages([
            ("system", "You're a solar PV system expert assistant. Analyze sensor data and suggest actions."),
            fewshot_prompt,
            ("human", "{query}")
        ])

        chain = prompt_chain | llm

        with st.spinner("Generating assistant insights..."):
            response = chain.invoke({"query": query_input})
            time.sleep(0.5)

        st.subheader("🤖 Few-Shot Assistant Response")
        st.markdown(response.content)

if __name__ == "__main__":
    main()
