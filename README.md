## 🔍 Autonomous PV Fault Analyzer (LangChain + Gemini)

An intelligent, end-to-end solar power fault analysis tool powered by **LangChain**, **Google Gemini**, and **Streamlit**. This assistant ingests real PV sensor data from `.csv` files, detects performance issues, and recommends actionable optimizations using LLM-powered analysis.

---

## 📁 Project Structure

```bash
📂 Autonomous-Maintenance-System_1/
├── app_1.py                # Unified app: Streamlit UI + Fault Analysis Logic
├── pv_sample.csv           # Example PV data (realistic)
├── requirements.txt        # All dependencies
├── .env                    # API key for Gemini (not committed)
└── README.md               # This file
```

---

## 💡 Key Features

✅ **CSV-based fault analysis**
✅ **LLM-powered reasoning (Gemini)**
✅ **Few-shot prompt tuning using Chroma DB**
✅ **Unified codebase (Streamlit + logic in one file)**
✅ **Real-time assistant insights**

---

## 🛠️ Setup Instructions

### 📦 Create a Virtual Environment

```bash
# Step 1: Create Environment
python -m venv myenv
```

### 🔃 Activate the Environment

* **Linux/macOS**:

```bash
source myenv/bin/activate
```

* **Windows**:

```bash
./myenv/Scripts/activate
```

### 📥 Install Dependencies

```bash
# (Recommended) Upgrade pip first
python.exe -m pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

---

## 🚀 Run the Application

```bash
streamlit run app_1.py
```

This will launch the **web UI** at `http://localhost:8501`.

---

## 🧪 How It Works

### 🧠 `app_1.py` – Unified Logic + UI

* Upload your PV system's `.csv` file via the Streamlit interface
* Choose the system type (e.g., PV, Chiller, etc.)
* The app performs **AI-driven fault detection**:

  * Detects low energy output, low inverter efficiency, overheating
  * Uses **LangChain + Gemini** for root cause analysis
  * Suggests fixes and optimizations using real-world reasoning
* A query text box allows manual log review or status input

  * Gemini will respond using few-shot examples retrieved from Chroma DB

---

## 🧾 Example CSV Input

```csv
datetime,system_id,expected_generation_kWh,energy_generated_kWh,inverter_efficiency_percent,module_temperature_C
2025-07-22 07:27:41,PV_SYS_100,112.88,93.27,98.55,29.19
2025-07-22 06:27:41,PV_SYS_101,175.73,143.24,92.13,38.62
...
```

---

## 📦 requirements.txt

```
langchain
langchain-core
langchain-community
langchain-experimental
langchain-google-genai
google-generativeai
chromadb
python-dotenv
pandas
streamlit
```

---

## 🤖 Innovative Extension Idea

> **🛰️ Auto-Feedback Loops for Remote PV Maintenance**
>
> Extend the app to:
>
> * Trigger alerts to on-site technicians
> * Suggest parts to inspect/order (based on efficiency or temperature drop)
> * Integrate **auto-email or WhatsApp notifications** for anomalies
> * Use **LangGraph** for chaining multiple AI actions (e.g., "Analyze + Notify + Suggest Spare Part")

---

## 📌 Note

* Place your Gemini API key in a `.env` file:

```
GOOGLE_API_KEY=your_api_key_here
```
## Screen Shots
<div style="display:flex;align-items:center;justify-content:center; flex-direction:column; width:100vw; gap:20px;">
<img src="./Assets/Screenshot 2025-07-23 214522.png"/>
<img src="./Assets/Screenshot.png"/>
<img src="./Assets/Screenshot 2025-07-23 214635_1.png"/>
</div>