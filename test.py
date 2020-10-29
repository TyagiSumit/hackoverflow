import pickle
import ml
loaded_model = pickle.load(open('models/model', 'rb'))

s = ml.prepare_data(['abdomen acute', 'abortion', 'achalasia'])
result = loaded_model.predict(s)
print(result)