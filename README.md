# 🚀 ResearchForge - Setup & Installation Guide

## 📖 Overview

ResearchForge is an AI-powered research assistant that automates literature review, gap analysis, and hypothesis generation. This guide will help you get the project running on your local machine.

---

## 🎯 What This Project Does

ResearchForge performs a complete 4-phase research analysis pipeline:

1. **📄 Paper Collection** - Automatically gathers 10-15 research papers from arXiv, Google Scholar, and ClinicalTrials.gov
2. **📚 Literature Review** - Generates comprehensive academic review using GPT-3.5-turbo
3. **🔍 Gap Analysis** - Identifies research gaps using dual AI models (Flan-T5-Large + GPT-4)
4. **💡 Hypothesis Generation** - Creates actionable research proposals with novelty scoring

**Time savings:** What takes researchers weeks → Done in 2-3 minutes

---

## ⚙️ System Requirements

- **Python:** 3.11 or higher
- **RAM:** 16GB minimum (for Flan-T5-Large model)
- **Storage:** 5GB free space (for AI models)
- **OS:** Windows, macOS, or Linux
- **Internet:** Required for API calls and model downloads

---

## 🛠️ Prerequisites

Before starting, you'll need:

### 1. **OpenAI API Key** (Required)
- Sign up at: https://platform.openai.com/
- Create API key at: https://platform.openai.com/api-keys
- **Cost:** ~$0.42 per complete analysis

### 2. **Google Custom Search API** (Required)
- Create project at: https://console.cloud.google.com/
- Enable Custom Search API
- Create credentials: https://developers.google.com/custom-search/v1/introduction
- Create Custom Search Engine: https://programmablesearchengine.google.com/
- **Cost:** Free tier (100 queries/day)

### 3. **HuggingFace Token** (Optional)
- Sign up at: https://huggingface.co/
- Get token at: https://huggingface.co/settings/tokens
- **Used for:** Faster model downloads (optional)

---

## 📦 Installation Steps

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/Radia-987/ResearchForge.git
cd ResearchForge
```

### **Step 2: Create Virtual Environment**

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### **Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

**⏳ Note:** First run will download ~3.5GB of AI models:
- Flan-T5-Large (3.13 GB)
- Sentence-BERT (80 MB)
- This is a one-time download

### **Step 4: Configure API Keys**

Create a `.env` file in the project root directory:

```bash
# Windows
notepad .env

# macOS/Linux
nano .env
```

Add your API keys:

```env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxx
GOOGLE_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxx
GOOGLE_SEARCH_ENGINE_ID=xxxxxxxxxxxxxxxxxxxx
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

**⚠️ Important:** Never commit this file to Git (it's already in `.gitignore`)

### **Step 5: Verify Installation**

Check that all dependencies are installed:

```bash
pip list
```

You should see:
- `streamlit==1.31.0`
- `openai==1.12.0`
- `transformers>=4.41.0`
- `torch>=2.1.0`
- `sentence-transformers>=5.0.0`

---

## ▶️ Running the Application

### **Start the Streamlit App:**
```bash
streamlit run app.py
```

### **Expected Output:**
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

### **Open in Browser:**
The app will automatically open at `http://localhost:8501`

If it doesn't, manually navigate to the URL shown in the terminal.

---

## 🎨 Using the Application

1. **Enter Research Topic**
   - Example: "federated learning privacy"
   - Example: "machine learning for drug discovery"

2. **Click "🚀 Start Complete Analysis"**

3. **Wait 2-3 minutes** (progress shown in terminal)

4. **View Results in 3 Tabs:**
   - 📚 Literature Review
   - 🔍 Gap Analysis
   - 💡 Research Hypotheses
   - 🧪 Experiment Design

---

## 📊 What Gets Downloaded

On first run, the following models are automatically downloaded:

| Model | Size | Purpose |
|-------|------|---------|
| google/flan-t5-large | 3.13 GB | Gap analysis (local) |
| sentence-transformers/all-MiniLM-L6-v2 | 80 MB | Novelty scoring |

**Total:** ~3.2 GB (stored in `~/.cache/huggingface/`)

---

## 🔑 API Costs

Estimated costs per complete analysis:

| Service | Cost per Analysis |
|---------|-------------------|
| OpenAI GPT-3.5 (Literature Review) | ~$0.02 |
| OpenAI GPT-4 (Gap Analysis) | ~$0.15 |
| OpenAI GPT-4 (Hypothesis Generation) | ~$0.25 |
| Google Custom Search | Free (100/day) |
| arXiv API | Free |
| ClinicalTrials.gov API | Free |
| **Total** | **~$0.42** |

---

## 🧪 Testing the Setup

### **Quick Test:**
```bash
python -c "from multi_agent_system import MultiAgentSystem; print('✅ Import successful!')"
```

### **Check OpenAI Connection:**
```bash
python -c "from dotenv import load_dotenv; from openai import OpenAI; import os; load_dotenv(); client = OpenAI(api_key=os.getenv('OPENAI_API_KEY')); print('✅ OpenAI connected!')"
```

---

## 🔧 Troubleshooting

### **Issue: `ModuleNotFoundError: No module named 'streamlit'`**
**Solution:**
```bash
pip install -r requirements.txt
```

---

### **Issue: `OPENAI_API_KEY not found`**
**Solution:**
- Ensure `.env` file exists in project root (same folder as `app.py`)
- Check API key format: `OPENAI_API_KEY=sk-proj-...`
- Restart the Streamlit app after creating `.env`

---

### **Issue: `httpx.ConnectError` or API connection fails**
**Solution:**
```bash
pip uninstall httpx
pip install httpx==0.24.1
```

---

### **Issue: Model download fails or hangs**
**Solution:**
```bash
pip install --upgrade transformers torch sentence-transformers
```

Or manually clear cache and retry:
```bash
# Windows
rmdir /s /q %USERPROFILE%\.cache\huggingface

# macOS/Linux
rm -rf ~/.cache/huggingface
```

---

### **Issue: `Out of Memory` when loading Flan-T5**
**Solution:**
- Ensure you have 16GB+ RAM
- Close other applications
- If still failing, the model will automatically fall back to GPT-4 only

---

### **Issue: Streamlit won't start**
**Solution:**
```bash
# Check if Streamlit is installed
streamlit --version

# Reinstall Streamlit
pip uninstall streamlit
pip install streamlit==1.31.0

# Try running from project directory
cd ResearchForge
streamlit run app.py
```

---

## 📁 Project Structure

```
ResearchForge/
├── app.py                      # Streamlit web interface
├── multi_agent_system.py       # Core logic (1932 lines)
├── requirements.txt            # Python dependencies
├── .env                        # API keys (create this)
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation
├── PROJECT_DOCUMENTATION.md    # Detailed technical docs
└── images/                     # Architecture diagrams
```

---

## 🤝 Support

**Issues?** Open an issue at: https://github.com/Radia-987/ResearchForge/issues

**Questions?** Check the detailed documentation: `PROJECT_DOCUMENTATION.md`

---

## 🎯 Next Steps

After successful installation:
1. ✅ Test with a simple query (e.g., "machine learning")
2. ✅ Review the generated literature review
3. ✅ Examine the identified research gaps
4. ✅ Explore the generated hypotheses
5. ✅ Check experiment design suggestions

---

## 📝 License

This project is open source and available for educational and research purposes.

---

## 🙏 Acknowledgments

**AI Models Used:**
- OpenAI GPT-4 & GPT-3.5-turbo
- Google Flan-T5-Large
- Sentence-BERT (all-MiniLM-L6-v2)

**Data Sources:**
- arXiv API
- Google Custom Search
- ClinicalTrials.gov

---

**🎉 You're ready to start! Run `streamlit run app.py` and explore AI-powered research analysis!**
