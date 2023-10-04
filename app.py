from flask import Flask,request,render_template
from model import predict as model_predict,search_most_similar,calculate_emotion_vector,calculate_language_vector
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/result',methods=['GET',"POST"])
def result():
    if request.method == 'GET':
        return render_template('result.html')
    elif request.method == 'POST':
        input_word=request.form.getlist("word")
        #key_word=model_predict(input_word)
        key_word_vec=calculate_emotion_vector(input_word)
        place=search_most_similar(key_word_vec)
        return render_template("result.html",place = place)

if __name__ == "__main__":
    app.run()
