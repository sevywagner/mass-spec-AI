from Helpers.postprocessingHelpers import processMassValue, getMostLikelyCompounds
from Objects.ExtremeGradientBoosting.XGBoostRegressor import XGBoostRegressor
from flask import Flask, jsonify, request, Response, send_file
from Helpers.preprocessingHelpers import find_peaks
from matplotlib.figure import Figure
from flask_headers import headers
from flask_cors import CORS
import pandas as pd
import datetime
import base64
import os
import io

model = XGBoostRegressor.loadModel('./Models/ExtremeGradientBoosting/combFeat-3subsample-5learners.pkl')
app = Flask(__name__)
CORS(app)

@app.route('/mass-val', methods=['POST'])
@headers({ 'access-control-allow-origin': '*' })
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
    
    return response

@app.route('/graph-upload', methods=['POST'])
def graphUpload():
    av = request.files['av']
    base = request.files['base']

    av_path = os.path.join(os.path.dirname('./'), 'data', 'peakGraph', 'graphs', '', 'mz_av')
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

@app.route('/predict', methods=['GET'])
@headers({ 'access-control-allow-origin': '*' })
def fitPredict():
    time = datetime.datetime.now()
    av_path = os.path.join(os.path.join(os.path.dirname('./'), 'data', 'UserFiles', 'graphs' f'mz_av-{time}.csv'))
    base_path = os.path.join(os.path.join(os.path.dirname('./'), 'data', 'UserFiles', 'graphs' f'mz_base-{time}.csv'))
    peaks_path = os.path.join(os.path.join(os.path.dirname('./'), 'data', 'UserFiles', 'peaks' f'peaks-{time}.csv'))
    pred_path = os.path.join(os.path.dirname('./'), 'data', 'UserFiles', 'graphPreds' f'predictions-{time}.txt')
    find_peaks(base_path, av_path, peaks_path)
    getMostLikelyCompounds(model, peaks_path, pred_path)

    response = send_file(pred_path)

    os.remove(av_path)
    os.remove(base_path)
    os.remove(pred_path)
    os.remove(peaks_path)

    return response


@app.route('/download-table', methods=['POST'])
@headers({ 'access-control-allow-origin': '*' })
def downloadTable():
    compounds = request.json['compounds']
    preds = request.json['preds']
    ppms = request.json['ppms']

    path = os.path.join(os.path.dirname('./'), 'data', 'UserFiles', 'tables', f'table-{datetime.datetime.now()}')

    with open(path, 'w+') as f:
        f.write("Compound\tPrediction\tPPM\n")
        for i in range(len(compounds)):
            f.write(str(compounds[i]) + '\t' + str(preds[i]) + '\t' + str(ppms[i]) + '\n')
        f.close()

    response = send_file(path)
    os.remove(path)

    return response

if __name__ == '__main__':
    app.run(debug=True, port=8080)