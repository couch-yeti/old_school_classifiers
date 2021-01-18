from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import joblib

# set data
path = "./data/data.csv"
df = pd.read_csv(path, encoding="latin1")
df.columns = ["text", "label"]

df["text"] = df.text.apply(lambda x: x.replace("\r\n\r\n", " "))

x_train, x_test, y_train, y_test = train_test_split(df["text"], df["label"])


vec = TfidfVectorizer(stop_words="english", max_features=1000)
tfidf = vec.fit_transform(x_train)
# joblib.dump(vec, "tfidf.joblib")

model = LogisticRegression(multi_class="multinomial", solver="lbfgs")
model.fit(tfidf, y_train)
# joblib.dump(model, "model.joblib")


if __name__ == "__main__":

    # joblib.dump(model, "model.joblib")
    # tfidf = joblib.load("tfidf.joblib")
    # vec = TfidfVectorizer(stop_words="english", max_features=1000, vocabulary =tfidf.vocabulary_)
    # vec = joblib.load("tfidf.joblib")
    test = vec.transform(
        [
            """Apple has reportedly already begun developing a foldable iPhone, Bloomberg reported Friday.
            So far, the tech giant has only worked on a prototype display with no set plans for a launch date,
            the report said citing unnamed sources. The prototype foldable screen has an invisible hinge, Bloomberg said.
            An Apple patent for a foldable iPhone first appeared almost a year ago, but Apple has yet to announce any plans. """
        ],
    )
    # model = joblib.load("model.joblib")
    prediction = model.predict(test)[0]
    array = model.predict_proba(test)[0]
    print(int(prediction))
    print(array)
    print(array[prediction -1 ])