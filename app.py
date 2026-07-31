import os
import re
import ssl
import nltk
from fastapi import FastAPI, Form, Request
import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure required NLTK resources are available silently
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
# ==========================================
# 🟢 CODE CHANGES: ROBUST NLTK DOWNLOAD & SSL FIX FOR RENDER
# Bypasses Linux SSL verification blocks and downloads required NLTK assets safely
# ==========================================
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

for resource in ['punkt', 'punkt_tab', 'stopwords']:
    try:
        nltk.download(resource, quiet=True)
    except Exception as e:
        print(f"Warning: Could not download {resource}: {e}")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# ==========================================

app = FastAPI(title="Smart ATS Resume Copilot")

# Sample Skill Database for Information Extraction (NER)
SKILL_DB = [
    "python", "javascript", "typescript", "react", "node.js", "express", "mongodb",
    "sql", "postgresql", "fastapi", "docker", "aws", "git", "machine learning",
    "nlp", "tensorflow", "pytorch", "pandas", "numpy", "html", "css", "tailwind"
]

def clean_and_tokenize(text: str):
    """Tokenization and Stop Word Removal Pipeline"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if w not in stop_words]
    return filtered_tokens

# ==========================================
# 🟢 CODE CHANGES: CORE ATS ENGINE LOGIC
# ==========================================
def extract_entities(text: str):
    """Extracts candidate contact details and skills (NER)"""
    # Extract Email using Regex
    emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    email = emails[0] if emails else "Not Found"

    # Extract Phone Number using Regex
    phones = re.findall(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
    phone = phones[0] if phones else "Not Found"

    # Extract Skills via Token Matching
    text_lower = text.lower()
    found_skills = [skill for skill in SKILL_DB if skill in text_lower]

    found_skills = [skill for skill in SKILL_DB if skill in text.lower()]
    return {
        "email": email,
        "phone": phone,
        "email": emails[0] if emails else "Not Found",
        "phone": phones[0] if phones else "Not Found",
        "skills": list(set(found_skills))
    }

def calculate_match_score(resume_text: str, job_text: str):
    """Calculates Match Percentage using TF-IDF and Cosine Similarity"""
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])
    
    # Cosine Similarity between vector 0 (Resume) and vector 1 (Job Description)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    match_percentage = round(similarity * 100, 2)
    return match_percentage
    return round(similarity * 100, 2)

def find_missing_keywords(resume_text: str, job_text: str):
    """Identifies missing skills required by the job description"""
    job_skills = [skill for skill in SKILL_DB if skill in job_text.lower()]
    resume_skills = [skill for skill in SKILL_DB if skill in resume_text.lower()]
    
    missing = set(job_skills) - set(resume_skills)
    return list(missing)
# ==========================================
    return list(set(job_skills) - set(resume_skills))

# --- Web UI Endpoint ---
@app.get("/", response_class=HTMLResponse)
def home():
    return """
@@ -84,16 +72,16 @@ def home():
    <body class="bg-slate-900 text-white min-h-screen p-8">
        <div class="max-w-4xl mx-auto">
            <h1 class="text-3xl font-bold mb-2 text-cyan-400">🤖 Smart ATS Resume Copilot</h1>
            <p class="text-slate-400 mb-8">Analyze candidate resumes against job descriptions using NLP, TF-IDF & Similarity Matching.</p>
            <p class="text-slate-400 mb-8">Analyze candidate resumes against job descriptions using NLP & TF-IDF.</p>
            
            <form action="/analyze" method="post" class="space-y-6">
                <div>
                    <label class="block text-sm font-medium mb-2 text-slate-300">Paste Candidate Resume Text</label>
                    <textarea name="resume_text" rows="6" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-3 text-slate-200 focus:outline-none focus:ring-2 focus:ring-cyan-500" required></textarea>
                    <textarea name="resume_text" rows="5" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-3 text-slate-200 focus:outline-none focus:ring-2 focus:ring-cyan-500" required></textarea>
                </div>
                <div>
                    <label class="block text-sm font-medium mb-2 text-slate-300">Paste Job Description</label>
                    <textarea name="job_text" rows="6" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-3 text-slate-200 focus:outline-none focus:ring-2 focus:ring-cyan-500" required></textarea>
                    <textarea name="job_text" rows="5" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-3 text-slate-200 focus:outline-none focus:ring-2 focus:ring-cyan-500" required></textarea>
                </div>
                <button type="submit" class="w-full bg-cyan-500 hover:bg-cyan-600 font-semibold py-3 px-6 rounded-lg transition duration-200">Run ATS Analysis</button>
            </form>
@@ -142,11 +130,20 @@ def analyze(resume_text: str = Form(...), job_text: str = Form(...)):

                <div class="bg-slate-800 border border-slate-700 rounded-lg p-6 space-y-3">
                    <h3 class="text-lg font-bold text-rose-400">Missing Required Skills</h3>
                    <p class="text-xs text-slate-400">Skills present in Job Description but missing from Resume:</p>
                    <div class="flex flex-wrap gap-2 pt-2">{missing_badge or '<span class="text-emerald-400 text-sm">None! Candidate covers all required skills.</span>'}</div>
                    <p class="text-xs text-slate-400">Skills in JD missing from Resume:</p>
                    <div class="flex flex-wrap gap-2 pt-2">{missing_badge or '<span class="text-emerald-400 text-sm">None! All matching.</span>'}</div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

# ==========================================
# 🟢 CODE CHANGES: RENDER PRODUCTION ENTRYPOINT
# Directly binds the app instance to host 0.0.0.0 and reads Render's assigned $PORT env var
# ==========================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
# ==========================================
