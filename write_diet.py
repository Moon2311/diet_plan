from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
# Chains
from langchain_classic.chains import LLMChain
from pydantic import BaseModel, Field
import pandas as pd
import os

# ----------------------------- 
# PYDANTIC MODELS
# ----------------------------- 
class DiabetesInput(BaseModel):
    Pregnancies: int = Field(..., ge=0, description="Number of pregnancies")
    Glucose: float = Field(..., gt=0, description="Glucose level (mg/dL)")
    BloodPressure: float = Field(..., gt=0, description="Blood pressure (mm Hg)")
    SkinThickness: float = Field(..., ge=0, description="Skin thickness (mm)")
    Insulin: float = Field(..., ge=0, description="Insulin level (µU/mL)")
    BMI: float = Field(..., gt=0, description="Body Mass Index")
    DiabetesPedigreeFunction: float = Field(..., ge=0, description="Diabetes pedigree function")
    Age: int = Field(..., ge=0, description="Age of the patient")

# ----------------------------- 
# STATE DEFINITION
# ----------------------------- 
class DietPlanState(TypedDict):
    """State for the diet plan generation workflow"""
    patient_input: DiabetesInput
    patient_text: str
    similar_cases: List[str]
    retrieved_context: str
    risk_level: str
    risk_analysis: str
    breakfast_plan: str
    lunch_plan: str
    dinner_plan: str
    snacks_plan: str
    complete_diet_plan: str
    lifestyle_recommendations: str
    foods_to_avoid: str
    final_output: str
    error: str | None

# ----------------------------- 
# LANGCHAIN SETUP
# ----------------------------- 
def get_llm():
    """Initialize Ollama LLM with LangChain"""
    return Ollama(
        model="phi3",
        temperature=0.5,
        num_predict=2048,
        top_p=0.95,
        repeat_penalty=1.1
    )

def load_embeddings():
    """Load HuggingFace embeddings model"""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ----------------------------- 
# DATA PROCESSING
# ----------------------------- 
def convert_csv_to_text(file_path: str, output_file: str) -> None:
    """Convert CSV data to text descriptions"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    df = pd.read_csv(file_path)
    
    def row_to_text(row):
        outcome_text = "positive for diabetes" if row["Outcome"] == 1 else "negative for diabetes"
        return (
            f"Patient with {row['Pregnancies']} pregnancies, glucose level {row['Glucose']}, "
            f"blood pressure {row['BloodPressure']}, skin thickness {row['SkinThickness']}, "
            f"insulin {row['Insulin']}, BMI {row['BMI']}, "
            f"diabetes pedigree function {row['DiabetesPedigreeFunction']}, "
            f"age {row['Age']}, outcome {outcome_text}."
        )
    
    df["text"] = df.apply(row_to_text, axis=1)
    
    with open(output_file, "w") as f:
        for line in df["text"]:
            f.write(line + "\n")
    
    print(f"✅ Saved text data to {output_file}")

def create_vector_store(text_file: str, embeddings) -> FAISS:
    """Create FAISS vector store from text file"""
    with open(text_file, "r") as f:
        text = f.read()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.create_documents([text])
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

# ----------------------------- 
# LANGGRAPH NODES WITH LANGCHAIN
# ----------------------------- 
def process_patient_input(state: DietPlanState) -> DietPlanState:
    """Node 1: Process and format patient input"""
    patient = state["patient_input"]
    
    patient_text = (
        f"Patient Profile:\n"
        f"- Age: {patient.Age} years\n"
        f"- Pregnancies: {patient.Pregnancies}\n"
        f"- Glucose Level: {patient.Glucose} mg/dL\n"
        f"- Blood Pressure: {patient.BloodPressure} mm Hg\n"
        f"- Skin Thickness: {patient.SkinThickness} mm\n"
        f"- Insulin: {patient.Insulin} µU/mL\n"
        f"- BMI: {patient.BMI}\n"
        f"- Diabetes Pedigree Function: {patient.DiabetesPedigreeFunction}"
    )
    
    state["patient_text"] = patient_text
    print("✅ Step 1: Patient input processed")
    return state

def assess_risk_with_llm(state: DietPlanState) -> DietPlanState:
    """Node 2: Assess diabetes risk level using LangChain"""
    
    llm = get_llm()
    
    # Create prompt template
    risk_prompt = PromptTemplate(
        input_variables=["patient_info"],
        template="""You are a medical expert specializing in diabetes risk assessment.

{patient_info}

Based on the above patient data, provide:
1. Risk Level: (Low/Moderate/High)
2. Key Risk Factors: List the main concerns
3. Risk Explanation: Brief explanation of why this risk level

Be concise and medical. Format your response clearly."""
    )
    
    # Create chain
    chain = risk_prompt | llm | StrOutputParser()
    
    # Generate risk analysis
    risk_analysis = chain.invoke({"patient_info": state["patient_text"]})
    
    # Extract risk level
    if "High" in risk_analysis or "high" in risk_analysis:
        risk_level = "High Risk"
    elif "Moderate" in risk_analysis or "moderate" in risk_analysis:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Low Risk"
    
    state["risk_level"] = risk_level
    state["risk_analysis"] = risk_analysis
    print(f"✅ Step 2: Risk assessed - {risk_level}")
    return state

def retrieve_similar_cases(state: DietPlanState) -> DietPlanState:
    """Node 3: Retrieve similar patient cases using RAG"""
    try:
        embeddings = load_embeddings()
        
        if os.path.exists("patient_data.txt"):
            vector_store = create_vector_store("patient_data.txt", embeddings)
            
            # Retrieve top 5 similar cases
            similar_docs = vector_store.similarity_search(
                state["patient_text"],
                k=5
            )
            
            similar_cases = [doc.page_content for doc in similar_docs]
            state["similar_cases"] = similar_cases
            state["retrieved_context"] = "\n".join(similar_cases)
            print(f"✅ Step 3: Retrieved {len(similar_cases)} similar cases")
        else:
            state["similar_cases"] = []
            state["retrieved_context"] = "No historical data available."
            print("⚠️  Step 3: No historical data found")
    
    except Exception as e:
        state["error"] = f"Retrieval error: {str(e)}"
        state["similar_cases"] = []
        state["retrieved_context"] = ""
        print(f"❌ Step 3: Error - {str(e)}")
    
    return state

def generate_breakfast_plan(state: DietPlanState) -> DietPlanState:
    """Node 4: Generate breakfast plan using LangChain"""
    
    llm = get_llm()
    
    breakfast_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert nutritionist specializing in diabetic meal planning."),
        ("user", """Create a detailed BREAKFAST plan for this diabetic patient:

{patient_info}

Risk Level: {risk_level}

Provide:
1. TIMING: Best time to eat breakfast
2. MAIN DISHES: 2-3 options with exact portions
3. BEVERAGES: Suitable drinks
4. NUTRITIONAL BENEFITS: Why these foods help manage diabetes
5. PREPARATION TIPS: Simple cooking instructions

Be specific with portions (e.g., "1 cup", "2 slices", "100g").
Focus on low glycemic index foods.""")
    ])
    
    chain = breakfast_prompt | llm | StrOutputParser()
    
    breakfast_plan = chain.invoke({
        "patient_info": state["patient_text"],
        "risk_level": state["risk_level"]
    })
    
    state["breakfast_plan"] = breakfast_plan
    print("✅ Step 4: Breakfast plan generated")
    return state

def generate_lunch_plan(state: DietPlanState) -> DietPlanState:
    """Node 5: Generate lunch plan using LangChain"""
    
    llm = get_llm()
    
    lunch_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert nutritionist specializing in diabetic meal planning."),
        ("user", """Create a detailed LUNCH plan for this diabetic patient:

{patient_info}

Risk Level: {risk_level}

Provide:
1. TIMING: Best time for lunch
2. MAIN COURSE: Protein, vegetables, and whole grains with portions
3. SALAD: Specific ingredients and dressing
4. SIDE DISHES: Healthy accompaniments
5. NUTRITIONAL BALANCE: Explain the macro balance

Focus on fiber-rich, protein-rich foods. Keep carbs controlled.""")
    ])
    
    chain = lunch_prompt | llm | StrOutputParser()
    
    lunch_plan = chain.invoke({
        "patient_info": state["patient_text"],
        "risk_level": state["risk_level"]
    })
    
    state["lunch_plan"] = lunch_plan
    print("✅ Step 5: Lunch plan generated")
    return state

def generate_dinner_plan(state: DietPlanState) -> DietPlanState:
    """Node 6: Generate dinner plan using LangChain"""
    
    llm = get_llm()
    
    dinner_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert nutritionist specializing in diabetic meal planning."),
        ("user", """Create a detailed DINNER plan for this diabetic patient:

{patient_info}

Risk Level: {risk_level}

Provide:
1. TIMING: Ideal dinner time (should be 2-3 hours before bed)
2. MAIN DISH: Light but satisfying protein with portions
3. VEGETABLES: Variety of non-starchy vegetables
4. CARBOHYDRATES: Controlled portions of complex carbs
5. SOUP (optional): Light, healthy soup option

Dinner should be lighter than lunch. Avoid heavy, hard-to-digest foods.""")
    ])
    
    chain = dinner_prompt | llm | StrOutputParser()
    
    dinner_plan = chain.invoke({
        "patient_info": state["patient_text"],
        "risk_level": state["risk_level"]
    })
    
    state["dinner_plan"] = dinner_plan
    print("✅ Step 6: Dinner plan generated")
    return state

def generate_snacks_plan(state: DietPlanState) -> DietPlanState:
    """Node 7: Generate snacks plan using LangChain"""
    
    llm = get_llm()
    
    snacks_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert nutritionist specializing in diabetic meal planning."),
        ("user", """Create HEALTHY SNACKS recommendations for this diabetic patient:

{patient_info}

Risk Level: {risk_level}

Provide:
1. MID-MORNING SNACK (10-11 AM): 2-3 options with portions
2. EVENING SNACK (4-5 PM): 2-3 options with portions
3. BEDTIME SNACK (if needed): Light options to prevent night hypoglycemia
4. PORTION CONTROL: Exact amounts for each snack
5. BENEFITS: Why these snacks help stabilize blood sugar

Focus on nuts, seeds, low-sugar fruits, and protein-rich options.""")
    ])
    
    chain = snacks_prompt | llm | StrOutputParser()
    
    snacks_plan = chain.invoke({
        "patient_info": state["patient_text"],
        "risk_level": state["risk_level"]
    })
    
    state["snacks_plan"] = snacks_plan
    print("✅ Step 7: Snacks plan generated")
    return state

def generate_lifestyle_recommendations(state: DietPlanState) -> DietPlanState:
    """Node 8: Generate lifestyle and exercise recommendations"""
    
    llm = get_llm()
    patient = state["patient_input"]
    
    lifestyle_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a diabetes management specialist."),
        ("user", """Provide comprehensive LIFESTYLE RECOMMENDATIONS for this patient:

{patient_info}

Risk Level: {risk_level}
BMI: {bmi}

Provide:
1. EXERCISE PLAN: Specific activities, duration, frequency
2. HYDRATION: Daily water intake target
3. SLEEP: Recommended sleep hours and schedule
4. STRESS MANAGEMENT: Practical techniques
5. BLOOD SUGAR MONITORING: When and how often to check
6. MEDICATION TIMING: Best times (if applicable)
7. WEIGHT MANAGEMENT: Realistic goals if BMI is high

Be practical and achievable.""")
    ])
    
    chain = lifestyle_prompt | llm | StrOutputParser()
    
    lifestyle_recommendations = chain.invoke({
        "patient_info": state["patient_text"],
        "risk_level": state["risk_level"],
        "bmi": patient.BMI
    })
    
    state["lifestyle_recommendations"] = lifestyle_recommendations
    print("✅ Step 8: Lifestyle recommendations generated")
    return state

def generate_foods_to_avoid(state: DietPlanState) -> DietPlanState:
    """Node 9: Generate foods to avoid list"""
    
    llm = get_llm()
    
    avoid_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a diabetes nutritionist."),
        ("user", """List FOODS TO STRICTLY AVOID for this diabetic patient:

{patient_info}

Risk Level: {risk_level}

Provide:
1. HIGH GLYCEMIC FOODS: Specific items to avoid
2. HIDDEN SUGARS: Products with hidden sugars
3. UNHEALTHY FATS: Trans fats and excessive saturated fats
4. PROCESSED FOODS: Specific processed items to avoid
5. BEVERAGES TO AVOID: Sugary and alcoholic drinks
6. WHY AVOID: Brief explanation for each category

Be specific with examples.""")
    ])
    
    chain = avoid_prompt | llm | StrOutputParser()
    
    foods_to_avoid = chain.invoke({
        "patient_info": state["patient_text"],
        "risk_level": state["risk_level"]
    })
    
    state["foods_to_avoid"] = foods_to_avoid
    print("✅ Step 9: Foods to avoid list generated")
    return state

def compile_final_diet_plan(state: DietPlanState) -> DietPlanState:
    """Node 10: Compile all components into final comprehensive diet plan"""
    
    final_output = f"""
{'='*80}
🏥 COMPREHENSIVE DIABETES DIET PLAN
{'='*80}

📋 PATIENT SUMMARY:
{state['patient_text']}

⚠️  RISK ASSESSMENT:
Risk Level: {state['risk_level']}
{state['risk_analysis']}

{'='*80}
🌅 BREAKFAST PLAN
{'='*80}
{state['breakfast_plan']}

{'='*80}
🌞 LUNCH PLAN
{'='*80}
{state['lunch_plan']}

{'='*80}
🌙 DINNER PLAN
{'='*80}
{state['dinner_plan']}

{'='*80}
🍎 HEALTHY SNACKS
{'='*80}
{state['snacks_plan']}

{'='*80}
🚫 FOODS TO AVOID
{'='*80}
{state['foods_to_avoid']}

{'='*80}
💪 LIFESTYLE RECOMMENDATIONS
{'='*80}
{state['lifestyle_recommendations']}

{'='*80}
📊 SIMILAR CASES ANALYZED: {len(state['similar_cases'])}
{'='*80}

⚕️  IMPORTANT NOTES:
1. This diet plan is a general guideline. Consult your doctor before making changes.
2. Monitor blood glucose levels regularly (before and 2 hours after meals).
3. Adjust portions based on your activity level and glucose readings.
4. Stay consistent with meal timing to maintain stable blood sugar.
5. Keep a food diary to track what works best for you.

💊 REMEMBER: Medication should be taken as prescribed by your physician.
🏃 EXERCISE: Aim for at least 150 minutes of moderate activity per week.
💧 HYDRATION: Drink 8-10 glasses of water daily.

{'='*80}
"""
    
    state["final_output"] = final_output
    state["complete_diet_plan"] = final_output
    print("✅ Step 10: Final comprehensive plan compiled")
    return state

# ----------------------------- 
# WORKFLOW CREATION
# ----------------------------- 
def create_diet_plan_workflow():
    """Create the LangGraph workflow with LangChain integration"""
    
    workflow = StateGraph(DietPlanState)
    
    # Add all nodes
    workflow.add_node("process_input", process_patient_input)
    workflow.add_node("assess_risk", assess_risk_with_llm)
    workflow.add_node("retrieve_cases", retrieve_similar_cases)
    workflow.add_node("breakfast", generate_breakfast_plan)
    workflow.add_node("lunch", generate_lunch_plan)
    workflow.add_node("dinner", generate_dinner_plan)
    workflow.add_node("snacks", generate_snacks_plan)
    workflow.add_node("lifestyle", generate_lifestyle_recommendations)
    workflow.add_node("avoid_foods", generate_foods_to_avoid)
    workflow.add_node("compile", compile_final_diet_plan)
    
    # Define workflow edges
    workflow.set_entry_point("process_input")
    workflow.add_edge("process_input", "assess_risk")
    workflow.add_edge("assess_risk", "retrieve_cases")
    workflow.add_edge("retrieve_cases", "breakfast")
    workflow.add_edge("breakfast", "lunch")
    workflow.add_edge("lunch", "dinner")
    workflow.add_edge("dinner", "snacks")
    workflow.add_edge("snacks", "lifestyle")
    workflow.add_edge("lifestyle", "avoid_foods")
    workflow.add_edge("avoid_foods", "compile")
    workflow.add_edge("compile", END)
    
    return workflow.compile()

# ----------------------------- 
# MAIN EXECUTION
# ----------------------------- 
def generate_personalized_diet_plan(patient_data: DiabetesInput) -> str:
    """
    Main function to generate personalized diet plan using LangChain + LangGraph
    
    Args:
        patient_data: DiabetesInput object with patient information
    
    Returns:
        str: Complete formatted diet plan
    """
    
    print("\n" + "="*80)
    print("🏥 DIABETES DIET PLAN GENERATOR")
    print("="*80 + "\n")
    print("🔄 Starting workflow...\n")
    
    # Initialize state
    initial_state: DietPlanState = {
        "patient_input": patient_data,
        "patient_text": "",
        "similar_cases": [],
        "retrieved_context": "",
        "risk_level": "",
        "risk_analysis": "",
        "breakfast_plan": "",
        "lunch_plan": "",
        "dinner_plan": "",
        "snacks_plan": "",
        "complete_diet_plan": "",
        "lifestyle_recommendations": "",
        "foods_to_avoid": "",
        "final_output": "",
        "error": None
    }
    
    # Create and run workflow
    app = create_diet_plan_workflow()
    final_state = app.invoke(initial_state)
    
    print("\n✅ Workflow completed successfully!\n")
    
    return final_state["final_output"]

# ----------------------------- 
# EXAMPLE USAGE
# ----------------------------- 
if __name__ == "__main__":
    # Example patient data
    patient = DiabetesInput(
        Pregnancies=2,
        Glucose=148,
        BloodPressure=72,
        SkinThickness=35,
        Insulin=85,
        BMI=33.6,
        DiabetesPedigreeFunction=0.627,
        Age=50
    )
    
    # Optional: Convert CSV to text first (run once)
    # convert_csv_to_text("diabetes.csv", "patient_data.txt")
    
    # Generate comprehensive diet plan
    diet_plan = generate_personalized_diet_plan(patient)
    
    # Print the complete plan
    print(diet_plan)
    
    # Optional: Save to file
    with open("diet_plan_output.txt", "w", encoding="utf-8") as f:
        f.write(diet_plan)
    print("\n💾 Diet plan saved to 'diet_plan_output.txt'")