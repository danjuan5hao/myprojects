import os,sys
ROOT = os.path.dirname(__file__)
sys.path.append(os.path.join(ROOT, "../../..")) 

from zwznlp.preprocessor import NerPreprocessor, BertNerPreprocessor


class TestNerPreprocessor:
    pass

if __name__ == "__main__":
    print("DONE")