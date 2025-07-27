# naive bayes classifier system

a modular fastapi-based system for training and using a naive bayes classifier on tabular data.  
split into two main components:

- **bayesian_model** – trains a model from csv and saves it as json
- **classifier** – loads a saved model and serves predictions

---

## 📁 project structure

.
├── bayesian_model/ # training service
│ ├── core_model/ # model building + testing logic
│ ├── data/ # csv datasets
│ ├── data_preparation/ # data loading + cleaning
│ ├── routes.py # / endpoint for training
│ ├── Dockerfile
│ └── requirements.txt
├── classifier/ # prediction service
│ ├── models/ # saved model jsons
│ ├── routes.py # / endpoint for prediction
│ ├── Dockerfile
│ ├── requirements.txt
└── README.md


---

## 💻 how to run

### 🚀 with docker (recommended)

```bash
# train model service (port 8001)
cd bayesian_model
docker build -t bayesian-model .
docker run -p 8000:8000 bayesian-model

# in a new terminal window or tab, run the classifier service (port 8002)
cd ../classifier
docker build -t classifier .
docker run -p 8002:8002 classifier



