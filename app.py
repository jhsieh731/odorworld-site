import os

from flask import Flask, render_template, request, redirect

# from inference import get_prediction
# from commons import format_class_name

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        print('1', request.form)
        print('2', request.data)
        for k, v in request.form.items():
            print('3', k, v)
        print('hellooooo')
        # return render_template('index.html')
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))