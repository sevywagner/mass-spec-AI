from flask import Flask, jsonify, request, Response, send_file
from flask_headers import headers
from flask_cors import CORS
from Helpers.preprocessingHelpers import find_peaks
from Helpers.postprocessingHelpers import processMassValue, getMostLikelyCompounds
from Objects.ExtremeGradientBoosting.XGBoostRegressor import XGBoostRegressor
from matplotlib.figure import Figure
import pandas as pd
import base64
import os
import io

model = XGBoostRegressor.loadModel('./Models/ExtremeGradientBoosting/combFeat-3subsample-5learners.pkl')
app = Flask(__name__)
CORS(app)

@app.route('/mass-val', methods=['POST'])
def processMassVal():
    c, s, uc, crit_enc, ppms = processMassValue(model=model,
                                            value=float(request.json['massVal']),
                                            mode='comb')

    response = jsonify({
        'compounds': c,
        'scores': s,
        'uCompounds': uc,
        'critereaEncodings': crit_enc,
        'ppms': ppms
    })

    response.headers['access-control-allow-origin'] = '*'
    
    return response

@app.route('/graph-upload', methods=['POST'])
def graphUpload():
    av = request.files['av']
    base = request.files['base']

    av_path = os.path.join(os.path.dirname('./'), 'data', 'peakGraph', 'graphs', '3', 'mz_av')
    base_path = os.path.join(os.path.dirname('./'), 'data', 'peakGraph', 'graphs', '3', 'mz_base')

    av.save(av_path)
    base.save(base_path)
    
    av = list(pd.read_csv(av_path)['mz_av'])
    base = list(pd.read_csv(base_path)['mz_base'])

    fig = Figure(figsize=(15, 5))
    ax = fig.subplots()
    ax.plot(base, av)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    data = base64.b64encode(buf.read()).decode("ascii")

    return Response(data, mimetype='image/png')

@app.route('/fit', methods=['GET'])
def fit():
    av_path = os.path.join(os.path.dirname('./'), 'data', 'peakGraph', 'graphs', '3', 'mz_av')
    base_path = os.path.join(os.path.dirname('./'), 'data', 'peakGraph', 'graphs', '3', 'mz_base')
    find_peaks(base_path, av_path, 'fitPeaks')
    return jsonify({ 'message': 'Success...' })

@app.route('/predict', methods=['GET'])
def fitPredict():
    av_path = os.path.join(os.path.dirname('./'), 'data', 'peakGraph', 'graphs', '3', 'mz_av')
    base_path = os.path.join(os.path.dirname('./'), 'data', 'peakGraph', 'graphs', '3', 'mz_base')
    find_peaks(base_path, av_path, 'fitPeaks')
    getMostLikelyCompounds(model, 'fitPeaks.txt', 'predictions.txt')
    return send_file(os.path.join(os.path.dirname('./'), 'data', 'output', 'predictions.txt'))

if __name__ == '__main__':
    app.run(debug=True, port=8080)