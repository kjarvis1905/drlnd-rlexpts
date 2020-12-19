import os
import mlflow
import shutil
#from contextlib import AbstractContextManager

class ArtifactHandler():
    def __init__(self):
        pass

    def __enter__(self):
        if not os.path.exists('./artifacts'):
            os.makedirs('./artifacts')


    def __exit__(self, *exc):
        print(mlflow.get_artifact_uri())
        print("Logging artifacts")
        mlflow.log_artifacts('./artifacts')
        print("Cleaning up...")
        shutil.rmtree('./artifacts/')
