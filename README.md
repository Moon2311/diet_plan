#  Diabetes Diet Plan Generator with LangChain & LangGraph

A comprehensive AI-powered system that generates personalized diabetes diet plans using **LangChain**, **LangGraph**, and **Ollama (Phi-3)**. The system uses state-driven workflows and RAG (Retrieval-Augmented Generation) to create medically-informed, detailed meal plans tailored to individual patient profiles.

---

##  Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Output Example](#output-example)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

##  Features

###  Core Capabilities
- **Personalized Diet Plans**: Custom meal plans based on 8 health parameters
- **Risk Assessment**: AI-powered diabetes risk evaluation
- **RAG Integration**: Retrieves similar patient cases for informed recommendations
- **Comprehensive Output**: Breakfast, lunch, dinner, snacks, and lifestyle advice
- **State Management**: LangGraph orchestrates multi-step workflow
- **Medical Accuracy**: Considers glucose levels, BMI, age, insulin, and more

###  Diet Plan Components
-  **Breakfast Plan** - Morning meals with timing and portions
-  **Lunch Plan** - Main meal with balanced macros
-  **Dinner Plan** - Light evening meals
-  **Snack Recommendations** - Mid-morning, evening, and bedtime options
-  **Foods to Avoid** - Specific items to eliminate
-  **Lifestyle Recommendations** - Exercise, sleep, hydration, monitoring
-  **Nutritional Benefits** - Why each food helps manage diabetes
-  **Portion Control** - Exact measurements for every meal

---

##  Architecture

### Technology Stack
```
┌─────────────────────────────────────────┐
│          LangGraph Workflow             │
│  (State Management & Orchestration)     │
└─────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌──────────────┐        ┌──────────────┐
│  LangChain   │        │     RAG      │
│   Chains     │◄──────►│ FAISS Vector │
│  (Prompts)   │        │    Store     │
└──────────────┘        └──────────────┘
        │                       │
        ▼                       ▼
┌──────────────┐        ┌──────────────┐
│ Ollama Phi-3 │        │  HuggingFace │
│     LLM      │        │  Embeddings  │
└──────────────┘        └──────────────┘
```

### Workflow Pipeline
```
1. Process Input → 2. Risk Assessment → 3. RAG Retrieval →
4. Breakfast Plan → 5. Lunch Plan → 6. Dinner Plan →
7. Snacks Plan → 8. Lifestyle Tips → 9. Foods to Avoid →
10. Compile Final Plan
```

---

##  Installation

### Prerequisites
- Python 3.9 or higher
- [Ollama](https://ollama.ai/) installed locally
- 8GB+ RAM recommended

### Step 1: Install Ollama and Phi-3
```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# For Windows, download from: https://ollama.ai/download

# Pull Phi-3 model
ollama pull phi3
```

### Step 2: Install Python Dependencies
```bash
# Clone the repository (or download the files)
git clone <your-repo-url>
cd diabetes-diet-planner

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install langchain==0.1.0
pip install langchain-community==0.0.13
pip install langchain-huggingface==0.0.1
pip install langgraph==0.0.26
pip install faiss-cpu==1.7.4
pip install sentence-transformers==2.3.1
pip install pandas==2.1.4
pip install pydantic==2.5.3
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

---

##  Quick Start

### 1. Start Ollama Server
```bash
# In a separate terminal
ollama run phi3
```

### 2. Prepare Your Data (Optional)
If you have historical patient data in CSV format:
```python
from diet_plan_generator import convert_csv_to_text

# Convert CSV to text format for RAG
convert_csv_to_text("diabetes.csv", "patient_data.txt")
```

### 3. Generate Diet Plan
```python
from diet_plan_generator import DiabetesInput, generate_personalized_diet_plan

# Create patient profile
patient = DiabetesInput(
    Pregnancies=2,
    Glucose=148,          # mg/dL
    BloodPressure=72,     # mm Hg
    SkinThickness=35,     # mm
    Insulin=85,           # µU/mL
    BMI=33.6,
    DiabetesPedigreeFunction=0.627,
    Age=50
)

# Generate comprehensive diet plan
diet_plan = generate_personalized_diet_plan(patient)

# Print the plan
print(diet_plan)
```

---

##  Usage

### Basic Usage
```python
from diet_plan_generator import DiabetesInput, generate_personalized_diet_plan

# Define patient data
patient = DiabetesInput(
    Pregnancies=1,
    Glucose=120,
    BloodPressure=70,
    SkinThickness=25,
    Insulin=100,
    BMI=28.5,
    DiabetesPedigreeFunction=0.4,
    Age=35
)

# Generate plan
diet_plan = generate_personalized_diet_plan(patient)

# Save to file
with open("my_diet_plan.txt", "w", encoding="utf-8") as f:
    f.write(diet_plan)
```

### Advanced Usage - With Historical Data
```python
# Step 1: Prepare historical data
convert_csv_to_text("diabetes.csv", "patient_data.txt")

# Step 2: Generate plan (RAG will automatically use the data)
patient = DiabetesInput(
    Pregnancies=3,
    Glucose=165,
    BloodPressure=80,
    SkinThickness=30,
    Insulin=120,
    BMI=35.2,
    DiabetesPedigreeFunction=0.8,
    Age=55
)

diet_plan = generate_personalized_diet_plan(patient)
```

---

##  Project Structure

```
diabetes-diet-planner/
│
├── diet_plan_generator.py      # Main application file
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── diabetes.csv                 # (Optional) Historical patient data
├── patient_data.txt             # Generated from CSV for RAG
├── diet_plan_output.txt         # Generated diet plan output
│
└── venv/                        # Virtual environment (not in repo)
```

---

##  How It Works

### 1. **Patient Input Processing**
```python
# Pydantic model validates and structures input
patient = DiabetesInput(
    Pregnancies=2,
    Glucose=148,
    # ... other parameters
)
```

### 2. **LangGraph State Management**
```python
class DietPlanState(TypedDict):
    patient_input: DiabetesInput
    patient_text: str
    risk_level: str
    breakfast_plan: str
    # ... other state variables
```

### 3. **LangChain Chains**
Each meal plan uses a specialized chain:
```python
breakfast_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert nutritionist..."),
    ("user", "Create a detailed BREAKFAST plan...")
])

chain = breakfast_prompt | llm | StrOutputParser()
breakfast_plan = chain.invoke({"patient_info": patient_text})
```

### 4. **RAG Retrieval**
```python
# Find similar patient cases
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)
similar_docs = vector_store.similarity_search(patient_text, k=5)
```

### 5. **Workflow Execution**
```python
workflow = StateGraph(DietPlanState)
workflow.add_node("process_input", process_patient_input)
workflow.add_node("assess_risk", assess_risk_with_llm)
# ... more nodes
app = workflow.compile()
final_state = app.invoke(initial_state)
```

---

##  Configuration

### Modify LLM Parameters
Edit the `get_llm()` function:
```python
def get_llm():
    return Ollama(
        model="phi3",
        temperature=0.5,        # Lower = more focused (0.0-1.0)
        num_predict=2048,       # Max tokens per response
        top_p=0.95,             # Nucleus sampling
        repeat_penalty=1.1      # Reduce repetition
    )
```

### Change Embedding Model
Edit `load_embeddings()`:
```python
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"  # More accurate
        # model_name="sentence-transformers/all-MiniLM-L6-v2" # Faster
    )
```

### Adjust RAG Parameters
In `retrieve_similar_cases()`:
```python
similar_docs = vector_store.similarity_search(
    state["patient_text"],
    k=5  # Change number of similar cases to retrieve
)
```

---

##  Output Example

```
================================================================================
 COMPREHENSIVE DIABETES DIET PLAN
================================================================================

 PATIENT SUMMARY:
Patient Profile:
- Age: 50 years
- Pregnancies: 2
- Glucose Level: 148 mg/dL
- Blood Pressure: 72 mm Hg
- BMI: 33.6
- Diabetes Pedigree Function: 0.627

  RISK ASSESSMENT:
Risk Level: High Risk
Key Risk Factors:
1. Elevated glucose level (148 mg/dL - above normal)
2. High BMI (33.6 - indicates obesity)
3. Family history (DPF 0.627 - elevated)
...

================================================================================
 BREAKFAST PLAN
================================================================================
TIMING: 7:00 AM - 8:00 AM (within 1 hour of waking)

MAIN DISHES:
Option 1: Steel-cut oatmeal (3/4 cup cooked) with:
  - 1 tablespoon ground flaxseeds
  - 1/2 cup fresh berries (blueberries or strawberries)
  - 10 raw almonds, chopped
  
Option 2: Vegetable omelet (2 eggs + 1 egg white) with:
  - 1 cup mixed vegetables (spinach, tomatoes, bell peppers)
  - 1 slice whole grain toast
  - 1 teaspoon olive oil for cooking
...

================================================================================
 LUNCH PLAN
================================================================================
...
```

---

##  Troubleshooting

### Common Issues

#### 1. **Ollama Connection Error**
```
ERROR: Ollama not running
```
**Solution:**
```bash
# Start Ollama
ollama serve

# In another terminal
ollama run phi3
```

#### 2. **Import Errors**
```
ModuleNotFoundError: No module named 'langchain'
```
**Solution:**
```bash
pip install --upgrade langchain langchain-community langgraph
```

#### 3. **FAISS Installation Issues (Windows)**
```bash
# If faiss-cpu fails, try:
conda install -c conda-forge faiss-cpu
# OR
pip install faiss-cpu --no-cache-dir
```

#### 4. **Memory Issues**
If Phi-3 runs slowly:
```python
# Use a smaller model
ollama pull phi3:mini

# Update get_llm()
def get_llm():
    return Ollama(model="phi3:mini", ...)
```

#### 5. **No Historical Data Warning**
```
  Step 3: No historical data found
```
This is normal if you haven't created `patient_data.txt`. The system will still work without RAG.

---

##  CSV Data Format

If using historical data, your CSV should have these columns:
```csv
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
...
```

---

##  Best Practices

### For Accurate Results:
1. **Provide accurate patient data** - All measurements should be recent
2. **Use historical data** - More RAG context = better recommendations
3. **Review with healthcare provider** - AI-generated plans should be verified
4. **Monitor glucose levels** - Track response to diet changes
5. **Adjust as needed** - Every patient responds differently

### For Better Performance:
1. **Keep Ollama running** - Avoid restart delays
2. **Use SSD storage** - Faster model loading
3. **Allocate sufficient RAM** - 8GB minimum recommended
4. **Batch process** - Generate multiple plans efficiently

---

##  Extending the System

### Add New Meal Types
```python
def generate_brunch_plan(state: DietPlanState) -> DietPlanState:
    llm = get_llm()
    brunch_prompt = ChatPromptTemplate.from_messages([...])
    chain = brunch_prompt | llm | StrOutputParser()
    state["brunch_plan"] = chain.invoke({...})
    return state

# Add to workflow
workflow.add_node("brunch", generate_brunch_plan)
workflow.add_edge("breakfast", "brunch")
```

### Use Different LLMs
```python
# OpenAI
from langchain_openai import ChatOpenAI
def get_llm():
    return ChatOpenAI(model="gpt-4", temperature=0.5)

# Anthropic Claude
from langchain_anthropic import ChatAnthropic
def get_llm():
    return ChatAnthropic(model="claude-3-sonnet-20240229")
```

### Add Web Search
```python
from langchain.tools import DuckDuckGoSearchRun

def research_latest_guidelines(state: DietPlanState) -> DietPlanState:
    search = DuckDuckGoSearchRun()
    results = search.run("latest diabetes diet guidelines 2024")
    state["latest_research"] = results
    return state
```

---

##  Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

##  Disclaimer

**This application is for educational and informational purposes only.**

- NOT a substitute for professional medical advice
- Always consult healthcare providers before dietary changes
- Generated plans should be reviewed by qualified nutritionists
- Individual results may vary significantly
- Monitor blood glucose levels regularly

---

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

---

##  Acknowledgments

- **LangChain** - For the amazing LLM framework
- **LangGraph** - For state management capabilities
- **Ollama** - For local LLM deployment
- **HuggingFace** - For embeddings models
- **Meta** - For Phi-3 language model

---

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Email: talhaabdulsattar018@example.com
- Documentation: [Link to docs]

---

##  Roadmap

- [ ] Web interface with Gradio/Streamlit
- [ ] Multi-language support
- [ ] Mobile app integration
- [ ] Integration with health monitoring devices
- [ ] Meal image generation
- [ ] Recipe database with cooking instructions
- [ ] Progress tracking and analytics
- [ ] Multi-patient management system

---

**Built with  using LangChain, LangGraph, and Ollama**

*Last Updated: December 2024*