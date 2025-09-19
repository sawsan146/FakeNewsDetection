from flask import Flask, request, render_template
import pytesseract
from PIL import Image
import joblib
import os

# تحميل الموديل والـ Vectorizer
model = joblib.load("FakeNewsModel.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# تحديد مسار tesseract.exe (غيريه لو مختلف عندك)
pytesseract.pytesseract.tesseract_cmd = r"D:\NTI\Final Project2\FakeNews\tesseract.exe"

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    extracted_text = None
    error_message = None
    confidence = None   # نسبة الثقة

    if request.method == "POST":
        try:
            # النص اللي المستخدم يكتبه
            input_text = request.form.get("text_input")

            # الصورة اللي المستخدم يرفعها
            file = request.files.get("file")

            # لو دخل نص
            if input_text and input_text.strip() != "":
                extracted_text = input_text

            # لو رفع صورة
            elif file and file.filename != "":
                img = Image.open(file.stream)
                extracted_text = pytesseract.image_to_string(img)

            else:
                error_message = "⚠️ رجاءً أدخل نص أو ارفع صورة."

            # لو في نص مستخرج
            if extracted_text and not error_message:
                text_features = vectorizer.transform([extracted_text])
                proba = model.predict_proba(text_features)[0]  # [prob_fake, prob_real]

                result = model.predict(text_features)[0]

                if result == 1:
                    prediction = "✅ Real News"
                    confidence = round(proba[1] * 100, 2)   # نسبة الـ Real
                else:
                    prediction = "❌ Fake News"
                    confidence = round(proba[0] * 100, 2)   # نسبة الـ Fake

        except Exception as e:
            error_message = f"حدث خطأ: {str(e)}"

    return render_template("index.html",
                           prediction=prediction,
                           extracted_text=extracted_text,
                           error_message=error_message,
                           confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)
