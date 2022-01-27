from prediction import * 
from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Welcome to API'

@app.route('/analyze', methods=["POST"])
def testpost():
    input_json = request.get_json(force=True) 
    sen1=input_json['standard_answer']
    sen2=input_json['student_answer']
    result=predict_sense(sen1,sen2,api_mode=True)
    return jsonify(result)

app.run(host='0.0.0.0',port=8080)
 
# https://www.okteto.com/blog/develop-and-deploy-a-flask-and-reactjs-app-in-okteto


# okteto context use https://cloud.okteto.com
# okteto stack deploy --build
# https://www.okteto.com/blog/building-and-deploying-a-fastapi-app-in-okteto-cloud


## testing json
# {
# "standard_answer":"An operating system is system software that manages computer hardware, software resources, and provides common services for computer programs.",
# "student_answer":"OS manages the all peripheral devices and all softwares "
# }