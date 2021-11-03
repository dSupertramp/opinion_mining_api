from opinion_miner import *
from flask import Flask, jsonify, request

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False


@app.route("/extract/<review>", methods=['GET'])
def main(review):
    return jsonify(process_review(review))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
