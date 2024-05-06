import os

from flask import Flask, render_template, request, redirect
import numpy as np
import utils
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
import base64
matplotlib.use('Agg')

app = Flask(__name__)

def create_heatmap(data):
    fig, ax = plt.subplots()
    heatmap = ax.imshow(data, cmap='hot', interpolation='nearest')
    ax.set_aspect(aspect=10)
    plt.colorbar(heatmap)

    # Convert plot to PNG image
    pngImage = BytesIO()
    FigureCanvas(fig).print_png(pngImage)
    plt.close(fig)  # Close the figure to free memory

    # Encode PNG image to base64 string
    pngImage.seek(0)
    base64_data = base64.b64encode(pngImage.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{base64_data}"


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        print('1', request.form)
        print('2', request.data)
        encoding = request.form.get('encoding')
        odors = list(request.form.keys())[:-1]
        print('3', odors)
        input, predictions, top_15_labels, top_15_probs, categories = utils.get_pred(len(odors), encoding, odors)
        top_15 = zip(top_15_labels, [round(x, 4) for x in top_15_probs], categories)
        # image_data = create_heatmap(input)
        print(input.shape)
        image_data = create_heatmap(input.reshape(13, 137))
        return render_template('result.html', input_odors=odors, input=image_data, odors=predictions, top15=top_15)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))