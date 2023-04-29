from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)


@app.route("/")
def home():
    if request.method == "POST":
        return
    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)
