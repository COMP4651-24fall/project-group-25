from function.core.train import *
result = mymodel()

def handle(event, context):
    return {
        "statusCode": 200,
        "body":  result
    }
