Multimodal AI for Pre-TIA Detection

[Stroke Shield Demo](https://youtu.be/-seCH_FrJxw)

Stroke Shield is an AI-powered prototype system designed to assess stroke risk by analyzing three key data streams:

- PPG Signal Analysis – Uses a 1D Convolutional Neural Network to detect Atrial Fibrillation from PPG signals.
- Facial Asymmetry Analysis – Uses MediaPipe to detect facial droop and asymmetry.
- Voice Analysis – Uses Librosa to extract features for dysarthria and Parkinson-like patterns.

A fusion engine combines results from all three streams to generate a tiered risk alert: Normal, Advisory, or Critical.

 ⚠️Disclaimer:This is a research prototype and is not a medical device. It is intended for educational and demonstration purposes only.


📦 Features

- Cardiac Module
  - Load PPG/ECG data from PhysioNet (MIT-BIH AF, MIMIC-PERform-AF)
  - Upload custom CSV files
  - Simulate signals for testing
  - Bandpass filtering + CNN classification
  - AFib probability score

- Neurological Module
  - Face Test: Real-time facial landmark detection and asymmetry scoring (MediaPipe)
  - Voice Test: Record speech, extract features, score for dysarthria and Parkinson-like patterns (Librosa)
  - Baseline Setup: Build a personalized baseline using Isolation Forest for each user

- Neuro Tests
  - Reaction time test
  - Cognitive test (digit span)
  - Tap speed test (tremor proxy)

- Fusion Engine
  - Combines all stream scores with weighted logic
  - Applies corroboration bonuses and temporal trend analysis
  - Generates tiered risk alerts

- Demo Mode
  - Pre-built scenarios (pre-TIA, AFib-only, neuro-only) to simulate system behavior
  - Step-by-step walkthrough for presentations

🛠️ Tech Stack

Backend
- Framework: FastAPI
- Language: Python 3.11
- AI/ML: TensorFlow, scikit-learn, NumPy, SciPy
- Signal Processing: WFDB, Librosa
- Facial Analysis: MediaPipe, OpenCV
- Database: PostgreSQL (SQLAlchemy)

Frontend
- Framework: React + TypeScript
- Build Tool: Vite
- Styling: Tailwind CSS
- State Management: Redux Toolkit
- Charts: Recharts
- Media: React-Webcam, React-Audio-Recorder

🗂️ Project Structure
stroke-shield/
├── backend/
│ ├── app/
│ │ ├── api/endpoints/ # FastAPI route handlers
│ │ ├── core/ # Config, logging, security
│ │ ├── models/ # Pydantic schemas
│ │ ├── services/ # Business logic
│ │ ├── ml/ # CNN, Isolation Forest, evaluation
│ │ └── db/ # SQLAlchemy models
│ ├── tests/ # pytest suite
│ └── requirements.txt
├── frontend/
│ ├── src/
│ │ ├── api/ # Axios service layer
│ │ ├── components/ # Reusable UI components
│ │ ├── pages/ # Full page views
│ │ ├── store/ # Redux slices
│ │ └── types/ # TypeScript interfaces
│ ├── public/
│ └── package.json


🚀 Quick Start

Prerequisites
- Python 3.11+
- Node.js 18+

### Backend Setup
``bash
Clone the repository
git clone https://github.com/goldokereke/stroke_Risk.git
cd stroke-shield/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your values

# Run the server
uvicorn app.main:app --reload

Frontend Setup
bash
cd ../frontend
npm install
npm run dev
The app will be available at http://localhost:5173 and the API at http://localhost:8000

Running Tests
bash
# Backend
cd backend
pytest tests/ -v

# Frontend
cd frontend
npm run test


🗂️ Datasets
Cardiac
MIT-BIH Atrial Fibrillation Database
MIMIC-PERform-AF

Neurological
TORGO Dysarthric Speech Database
UCI Parkinson's Telemonitoring
Facial Palsy DB


📄 License
This project is for educational and research purposes.

⚠️ Medical Disclaimer
This software is for research and demonstration purposes only. It is not approved for clinical use. Always consult a qualified healthcare professional for medical decisions.

