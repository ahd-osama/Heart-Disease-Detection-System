
from experta import *

class HeartDiseaseExpert(KnowledgeEngine):
    def __init__(self):
        super().__init__()
        self.risk_count = 0 

    @Rule(Fact(cp=P(lambda x: x == 4)))
    def asymptomatic_chest_pain(self):
        self.risk_count += 1

    @Rule(Fact(age=MATCH.age) & Fact(thalach=MATCH.thalach))
    def low_max_heart_rate(self, age, thalach):
        if thalach < 0.85 * (220 - age):
           self.risk_count += 1

    @Rule(Fact(exang=P(lambda x: x == 1)))
    def exercise_induced_angina(self):
        self.risk_count += 1

    @Rule(Fact(oldpeak=P(lambda x: x > 2)))
    def st_depression(self):
        self.risk_count += 1

    @Rule(Fact(ca=P(lambda x: x > 0)))
    def blocked_vessels(self):
        self.risk_count += 1

  @Rule(Fact(thal=P(lambda x: x in [6, 7])))
    def thalassemia_defect(self):
        self.risk_count += 1

    def predict(self, patient_data):
        self.reset()
        self.risk_count = 0  
        
        for key, value in patient_data.items():
            self.declare(Fact(**{key: value}))
        
        self.run()

        return 0 if self.risk_count <= 2 else 1
