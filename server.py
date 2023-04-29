from flask import Flask, request

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        return "hello world"
    