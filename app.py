from flask import Flask, jsonify, request
import pickle
import os
import traceback

app = Flask(__name__)

def make_prediction(input_values):
    dest = 'models\pkl_objects'
    rf = pickle.load(open(os.path.join(dest, 'rf_baseline.pkl'), 'rb'))

    return rf.predict_proba([input_values])

@app.route('/', methods=['GET', 'POST'])
def predict():
	try:
		json = request.get_json()	 
		# temp=list(json[0].values())
		temp = [148, 72, 0, 33.6, 50]
		prediction = make_prediction(temp)
		print("Prediction: ", prediction)        
		return jsonify({'prediction': str(prediction[0])})

	except:        
		return jsonify({'trace': traceback.format_exc()})
    


if __name__ == '__main__':
    app.run(debug=True)