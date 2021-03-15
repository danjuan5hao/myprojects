import json

from flask import Flask
from flask import request
app = Flask(__name__)

from revo.share.core.dialState import initial_dial_state

NLU_MODEL = None
DST_MODEL = None
POLICY_MODEL = None

DIAL_STATE_CACHE = {}

def load_nlu_model():
    global NLU_MODEL

    

def load_dst_model():
    pass 

def load_policy_model():
    pass 

@app.route('/')
def bot_redict():
    user_id = request.args.get("user_id")
    user_utterance = request.args.get("utterance")
    user_dial_state = DIAL_STATE_CACHE.get("user_id", initial_dial_state() )

    # nlu_rst = NLU_MODEL(user_utterance)
    # user_dial_state = DST_MODEL(user_dial_state)
    # action = POLICY_MODEL(user_dial_state)
    action = None
    return action

if __name__ == '__main__':
    app.run(host="127.0.0.1:5000")