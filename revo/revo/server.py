import json
import os, sys 
root = os.path.join(os.path.dirname(__file__), "../")
sys.path.append(root)

from flask import Flask
from flask import request
app = Flask(__name__)

from revo.share.dialState import initial_dial_state
from revo.nlu.model.model import NLUModel

NLU_MODEL = NLUModel()
DST_MODEL = None
POLICY_MODEL = None

DIAL_STATE_CACHE = {}



@app.route('/')
def bot_predict():
    user_id = request.args.get("user_id")
    user_utterance = request.args.get("utterance")
    user_dial_state = DIAL_STATE_CACHE.get("user_id", initial_dial_state() )
    tokened = NLU_MODEL.tokenize(user_utterance)
    # nlu_rst = NLU_MODEL(user_utterance)
    # user_dial_state = DST_MODEL(user_dial_state)
    # action = POLICY_MODEL(user_dial_state)
    action = None
    return str(tokened)

if __name__ == '__main__':
    app.run(host="127.0.0.1")