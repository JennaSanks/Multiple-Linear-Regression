from flask import Flask, render_template, request
import pickle
import numpy as np


# ============================================================
# CREATE FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = pickle.load(
    open("UberFareModel.pkl", "rb")
)


# ============================================================
# HAVERSINE DISTANCE FUNCTION
# ============================================================

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):

    R = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(
        np.sqrt(a)
    )

    return R * c


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ----------------------------------------------------
        # GET USER INPUTS
        # ----------------------------------------------------

        passenger_count = float(
            request.form["passenger_count"]
        )

        pickup_latitude = float(
            request.form["pickup_latitude"]
        )

        pickup_longitude = float(
            request.form["pickup_longitude"]
        )

        dropoff_latitude = float(
            request.form["dropoff_latitude"]
        )

        dropoff_longitude = float(
            request.form["dropoff_longitude"]
        )

        hour = float(
            request.form["hour"]
        )

        day = float(
            request.form["day"]
        )

        month = float(
            request.form["month"]
        )

        year = float(
            request.form["year"]
        )

        weekday = float(
            request.form["weekday"]
        )


        # ----------------------------------------------------
        # CALCULATE DISTANCE
        # ----------------------------------------------------

        distance = haversine_distance(
            pickup_latitude,
            pickup_longitude,
            dropoff_latitude,
            dropoff_longitude
        )


        # ----------------------------------------------------
        # CREATE MODEL INPUT
        # ----------------------------------------------------

        input_data = np.array([[
            passenger_count,
            distance,
            hour,
            day,
            month,
            year,
            weekday
        ]])


        # ----------------------------------------------------
        # PREDICT FARE
        # ----------------------------------------------------

        prediction = model.predict(
            input_data
        )


        fare = float(
            prediction[0]
        )


        # ----------------------------------------------------
        # PREVENT NEGATIVE FARE
        # ----------------------------------------------------

        fare = max(
            fare,
            0
        )


        # ----------------------------------------------------
        # RETURN RESULT
        # ----------------------------------------------------

        return render_template(
            "index.html",

            prediction_text=
                f"${fare:.2f}",

            distance_text=
                f"{distance:.2f} km"
        )


    except Exception as e:

        return render_template(
            "index.html",

            error_text=
                "Please enter valid values."
        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )