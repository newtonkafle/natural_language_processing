from flask import Flask, request, render_template, redirect, url_for
from flask_wtf import FlaskForm, CSRFProtect
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Length
import numpy as np
import sys

sys.path.append("../")  # noqa: E402
from model import Model
from data_cleaning import DataPreparation

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
    # load the dataprep class
    data_prep = DataPreparation()

    # load the model class
    model = Model()
    loaded_model = model.load_model(model_name="LSTM", path="../data/model_weights/")
    form = Review()
    if request.method == "POST":
        print("hello")
        result = ""
        if form.validate_on_submit():
            # print("hello")
            review = form.review_field.data
            # # review = [review]
            print(review)

            # remove the stop words and lemitize the words
            cleaned_review = [data_prep.clean_and_lemmitize(review)]

            # convert the text into the numerical data
            t_review = data_prep.tokenize_and_pad_items(
                item_seq=cleaned_review, max_length=70
            )  # noqa: E402

            print(t_review)

            predictions = np.argmax(loaded_model.predict(t_review), axis=1)
            print(predictions)

            result = "Postive" if predictions[0] == 1 else "Negative"
            print(result)
        return (
            render_template("index.html", form=form, response=result)
            + '<div class="d-flex justify-content-center mt-4"><br/>Reviewed Text : '
            + form.review_field.data
            + "<br/>Classification: "
            + result
            + "</div>"
        )
    return render_template("index.html", form=form, response="Neutral")


if __name__ == "__main__":
    app.run(debug=True)
