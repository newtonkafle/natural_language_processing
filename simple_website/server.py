from flask import Flask, request, render_template, redirect, url_for
from flask_wtf import FlaskForm, CSRFProtect
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Length
import numpy as np
import sys

sys.path.append("../")  # noqa: E402
from model import Model

app = Flask(__name__)
app.secret_key = "thisissecret"
csrf = CSRFProtect(app)


class Review(FlaskForm):
    review_field = StringField(
        "Review",
        validators=[DataRequired()],
        render_kw={"placeholder": "Enter a review here"},
    )
    submit = SubmitField("Analyze")


@app.route("/", methods=("GET", "POST"))
def home():
    form = Review()
    if request.method == "POST":
        print("hello")
        result = ""
        if form.validate_on_submit():
            print("hello")
            review = form.review_field.data.split(" ")
            review = [review]
            print(review)
            model = Model()
            loaded_model = model.load_model(path="../data/model_weights/model.h5")
            # remove the stop words
            # lemmitize the words

            # convert the text into the numerical data
            t_review = model.tokenize_and_pad_items(
                item_seq=review, max_length=70
            )  # noqa: E402

            print(t_review)

            predictions = np.argmax(loaded_model.predict(t_review), axis=1)
            print(predictions)

            result = "Postive" if predictions[0] == 1 else "Negative"
            print(result)
        return render_template("index.html", form=form,response=result) +'<div class="d-flex justify-content-center mt-4"><br/>Reviewed Text : '+ form.review_field.data + '<br/>Classification: ' + result +'</div>'
    return render_template("index.html", form=form,response='Neutral')


if __name__ == "__main__":
    app.run(debug=True)