import streamlit as st
from langchain.chains import LLMChain
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
import os
import requests

model_url = "https://huggingface.co/Umar-Sanda/llama-2-7b-chat.Q4_K_M.gguf/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
model_path = "llama-2-7b-chat.Q4_K_M.gguf"

# Download the model if it doesn't exist
if not os.path.exists(model_path):
    with open(model_path, "wb") as f:
        response = requests.get(model_url, stream=True)
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

# Load the model
llm = LlamaCpp(model_path=model_path, temperature=0.7, max_tokens=2048)

# Function to test model output
def test_model():
    test_prompt = "What is 2 + 2?"
    result = llm.invoke(test_prompt)
    return result

# Resume prompt template (LLaMA instruction format)
prompt_template = PromptTemplate(
    input_variables=["full_name", "email", "phone", "summary", "experience", "education", "skills"],
    template="""
    [INST] You are a professional resume writer. 
    Write a detailed resume based on the following user input:
    
    Full Name: {full_name}
    Email: {email}
    Phone Number: {phone}

    Summary:
    {summary}

    Work Experience:
    {experience}

    Education:
    {education}

    Skills:
    {skills}
    [/INST]
    """
)

# Define the resume generation chain
resume_chain = LLMChain(llm=llm, prompt=prompt_template)

# Function to generate resume
def generate_resume(full_name, email, phone, summary, experience, education, skills):
    resume = resume_chain.run(
        full_name=full_name, 
        email=email, 
        phone=phone, 
        summary=summary, 
        experience=experience, 
        education=education, 
        skills=skills
    )
    return resume

# Streamlit UI
st.title("📄 Frontlett Resume Generator")
st.write("Generate a professional resume step-by-step!")

# Step-by-step user inputs
full_name = st.text_input("👤 Full Name:")
email = st.text_input("📧 Email Address:")
phone = st.text_input("📞 Phone Number:")
summary = st.text_area("📝 Summary:")
experience = st.text_area("💼 Work Experience:")
education = st.text_area("🎓 Education:")
skills = st.text_area("🛠 Skills:")

# Generate Resume Button
if st.button("🚀 Generate Resume"):
    if full_name and email and phone and summary and experience and education and skills:
        with st.spinner("Generating resume..."):
            resume = generate_resume(full_name, email, phone, summary, experience, education, skills)
            
            # Display Resume
            st.subheader("📄 Generated Resume")
            st.text_area("Resume Output", resume, height=400)

            # Debugging: Show raw model output
            st.subheader("🛠 Debug: Raw Model Output")
            st.code(resume)
    else:
        st.warning("⚠️ Please fill in all fields before generating your resume.")

# Model test button
if st.button("🛠 Test Model Output"):
    test_result = test_model()
    st.subheader("🔎 Model Test Result")
    st.write(test_result)
