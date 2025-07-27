# naive bayes classifier system

a modular fastapi-based system for training and using a naive bayes classifier on tabular data.  
split into two components/docker images:

- **bayesian_model** – trains a model from csv and saves it as json
- **classifier** – loads a saved model and serves predictions


## 💻 how to run

### 🚀 with docker (recommended)

```bash
# train model service (port 8000)
cd bayesian_model
docker build -t bayesian-model .
docker run -p 8000:8000 bayesian-model

# in a new terminal window or tab, run the classifier service (port 8001)
cd ../classifier
docker build -t classifier .
docker run -p 8001:8001 classifier



