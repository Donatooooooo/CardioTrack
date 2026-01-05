from locust import HttpUser, between, task
import random

def random_heart_sample() -> dict:
    return {
        "Age": random.randint(30, 80),
        "ChestPainType": random.choice(["TA", "ATA", "NAP", "ASY"]),
        "RestingBP": random.randint(90, 180),
        "Cholesterol": random.randint(150, 350),
        "FastingBS": random.choice([0, 1]),
        "RestingECG": random.choice(["Normal", "ST", "LVH"]),
        "MaxHR": random.randint(60, 200),
        "ExerciseAngina": random.choice(["Y", "N"]),
        "Oldpeak": round(random.uniform(-2, 4), 1),
        "ST_Slope": random.choice(["Up", "Flat", "Down"]),
    }
    
class WebsiteUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def home(self):
        self.client.get("/")

    @task(1)
    def get_dataset_card(self):
        self.client.get("/cards/dataset_card")

    @task(1)
    def get_model_card(self):
        self.client.get("/cards/model_card")

    @task(1)
    def get_model_hyperparameters(self):
        self.client.get("/model/hyperparameters")

    @task(2)
    def get_model_metrics(self):
        self.client.get("/model/metrics")

    @task(5)
    def predict_single(self):
        self.client.post(
            "/predictions",
            json=random_heart_sample(),
            name="/predictions (single)",
        )
            
    @task(2)
    def predict_batch(self):
        batch_size = random.randint(5, 20)
        payload = [random_heart_sample() for _ in range(batch_size)]

        self.client.post(
            "/batch-predictions",
            json=payload,
            name="/batch-predictions",
        )       


    @task(2)
    def explain_prediction(self):
        self.client.post(
            "/explanations",
            json=random_heart_sample(),
            name="/explanations",
        )
