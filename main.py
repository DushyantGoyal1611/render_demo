import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify, render_template


education_mapping = {
    'Engineering & Technology': [
        'B.E / B-Tech', 'M.E / M-Tech', 'B-Tech - INTEGRATED', 'BE - CSe and CDAC',
        'B.ENG', 'Bachelors of planning', 'B.Arch', 'Bachelor of Architecture', 'B.ARC'
    ],
    'Science & Medical': [
        'BSc or MSc', 'BSC', 'MS', 'Masters in biotechnology', 'PG - msc biochemistry',
        'Bio science', 'masters in biology', 'biotech', 'Masters of Statistics',
        'Bachelors in biotechnology', 'Ph D genetics', 'MBBS', 'Bachelor in medical lab',
        'B.Physiotherapist', 'B.pharma', 'M.pharma', 'B -pharm', 'B.PHARMA',
        'PHARMA D', 'pharm D', 'M.Pharma', 'Bachelors in dental',
        'Masters in Clinical Psychology', 'B.voc in applied clinical psychology'
    ],
    'Commerce & Business': [
        'B.com', 'BCOM COMPUTERS', 'Bcom computers', 'bcom -pa', 'M.com',
        'MBA', 'MBA- 2019', 'MBA agro business', 'PGDM', 'pgdm', 'PGDM 2022',
        'PGDM marketing', 'PG diploma in banking', 'PG DIPLOMA IN BANKING',
        'Masters in banking and insurance', 'BBA or BBM', 'BBM',
        'Bachelor of Business Economics (B.B.E.)', 'PGPM', 'PGPM in marketing',
        'PGPCEM', 'PGDBS', 'B.Voc - Banking & Finance'
    ],
    'Humanities & Social Sciences': [
        'BA/MA', 'LLB', 'LLM', 'MPhil', 'M.phil', 'Mphil 2020', 'PHD', 'doctorate',
        'Masters of Social Work (MSW)', 'msw', 'msw-hr-2014', 'MSW HRM',
        'Public administration', 'Masters in comparative religion', 'Masters in archeology',
        'English Honors', 'MASTERS IN SOCIAL SCIENCE', 'Bengali Honors'
    ],
    'Mass Communication & Media': [
        'B.comm (Bachelor of mass comm.)', 'Bachelor mass communication',
        'mass media', 'masss media', 'mass comm', 'Mass comm', 'BMM ( MAss Media)',
        'BJMC', 'MJMC 2020', 'Masters in mass communication', 'MASTERS IN MASS COM',
        'Masters in Convergent Journalism', 'master maas comm', 'master in electronic media'
    ],
    'Vocational & Professional Courses': [
        'B.ed (Teaching)', 'B.ed', 'B.ED', 'M.ed', 'D.el.ed', 'd el. ed', 'Dled',
        'Bachelors in fisheries science', 'Bachelors in tourism', 'Bachelor of Tourism',
        'travel & toursim', 'Bachelors in design', 'B.Design', 'Bachelors in hospitality',
        "Bachelor's in Hospitality", 'Hotel Management', 'Masters in fashion designing',
        'Masters in hospitality', 'Masters in design', 'food technology',
        'BVoc', 'B.VOC', 'B.vocational', 'B.voc - IT', 'B.voc networking and mobile applications',
        'bachelors of vocational - automobile', 'Bpharma', 'Mpharma', 'Diploma', 'Diploma in engr',
        'PGDCA', 'PGD', 'pgdca', 'PGD (Bachelor of Library Science)', 'BMIT',
        'PGDF', 'PGDA', 'MFC', 'MLIS', 'BMS', 'BMS marketing', 'bms', 'bms-2020',
        'Bachelor in IT', 'BCA/MCA', 'CSE', 'PDGM Advance Computing'
    ]
}

# Function to map education to categories
def categorize_education(education):
    for category, degrees in education_mapping.items():
        if education in degrees:
            return category
    return 'Other'  

def format_education(input_df):
    input_df = input_df.copy()
    input_df['Education'] = input_df['Education'].fillna('Unknown').apply(lambda x: categorize_education(x))
    return input_df

def convert_age(input_df):
    input_df = input_df.copy()
    input_df['Age'] = pd.to_numeric(input_df['Age'].astype(str).str.replace('+', '', regex=False), errors='coerce')
    input_df['Age'] = pd.cut(input_df['Age'], 
                       bins=[18, 22, 25, 28, 32, 35, float('inf')],
                       labels=['18-22', '23-25', '26-28', '29-32', '33-35', '35+'], 
                       right=True)
    return input_df

def label_encoder(input_df):
    input_df = input_df.copy()
    label_encoders = {}
    categorical_cols = [
        'Gender', 'Marital status', 'Mode of Interview',
        'Does the candidate has mother tongue influence while speaking english.',
        'Acquaintance and Referral', 'Currently Employed',
        'Candidate is willing to relocate'
    ]
    
    for col in categorical_cols:
        input_df[col] = input_df[col].fillna('Unknown')
        le = LabelEncoder()
        input_df[col] = le.fit_transform(input_df[col]) 
        label_encoders[col] = le  
        
    return input_df

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f) 

MODEL_PATH = "Candidate_Recommender_model.pth"
TEST_TENSOR_PATH = "X_test_tensor.pth"
TEST_INDEX_PATH = "X_test_index.pkl"

X_test_tensor = torch.load(TEST_TENSOR_PATH, map_location=torch.device("cpu"))
X_test_index = load_pickle(TEST_INDEX_PATH)  

class RecommenderModel(nn.Module):
    def __init__(self, input_size, hidden_units):
        super().__init__()

        # Fully Connected Layers
        self.fc1 = nn.Linear(input_size, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units // 2)
        self.fc3 = nn.Linear(hidden_units // 2, 1)

        # Activation Functions & Regularization
        self.dropout = nn.Dropout(0.3)
        self.batchnorm1 = nn.BatchNorm1d(hidden_units)
        self.batchnorm2 = nn.BatchNorm1d(hidden_units // 2)


    def forward(self, x):
        x = self.fc1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.dropout(x)

        return self.fc3(x).squeeze()
    
model = RecommenderModel(40, 128)

# Load model
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Load Pipeline
with open("pipeline.pkl", "rb") as pipeline_file:
    pipeline = pickle.load(pipeline_file)

# Flask App
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON. Set 'Content-Type: application/json'"}), 415

    data = request.get_json()
    try:
        # Convert JSON to DataFrame
        input_df = pd.DataFrame([data])

        # Apply preprocessing
        processed_input = pipeline.transform(input_df)
          
        # Convert to PyTorch tensor
        inputs = torch.tensor(processed_input, dtype=torch.float32)

        with torch.no_grad():
            scores = F.cosine_similarity(inputs, X_test_tensor)

        ranking_df = pd.DataFrame({
            'id' : X_test_index,
            'score': [round(float(s), 2) for s in scores.numpy()]
        })    

        ranked_candidates = ranking_df.sort_values(by='score', ascending=False).head(5)
        recommendations = ranked_candidates.to_dict(orient='records')

        return jsonify({"recommendations": recommendations})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)