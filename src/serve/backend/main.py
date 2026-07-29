import io
import time
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype, Lambda

import mlflow
from mlflow.tracking import MlflowClient


MLFLOW_TRACKING_URI = 'http://uri'
MLFLOW_EXPERIMENT_NAME = 'MLOPS_PROJECT'


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
metric_to_optimize = 'metrics.val_f1_score'
sort_order = 'DESC'

def load_best_model():
    client = MlflowClient()
    
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f'Experiment {MLFLOW_EXPERIMENT_NAME} not found in MLflow.')
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f'{metric_to_optimize} {sort_order}'],
        max_results=1
    )
        
    best_run = runs[0]
    best_run_id = best_run.info.run_id
    
    metric_key = metric_to_optimize.replace('metrics.', '')
    best_score = best_run.data.metrics.get(metric_key, 'Unknown')
    print(f'Best Run: {best_run_id} | {metric_key}: {best_score}')
    
    model_uri = f'runs:/{best_run_id}/resnet18'
    best_model = mlflow.pytorch.load_model(model_uri).to('cuda').to(device)
    best_model.eval()
    
    return best_model, model_uri

try:
    model, loaded_model_uri = load_best_model()
except Exception as e:
    print(f'CRITICAL ERROR loading model from MLflow: {e}')
    model, loaded_model_uri = None, None


preprocess = Compose([
    Resize((224, 224)),
    Lambda(lambda img: img.convert('RGB')),
    ToImage(),
    ToDtype(torch.float32, scale=True)
])


@app.post('/classify')
async def classify_image(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail='Model is not loaded.')
        
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='Invalid file type.')

    start_time = time.time()
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top3_prob, top3_class = torch.topk(probabilities, 3)

    predictions = [
        {
            'label': f'Class {top3_class[i].item()}', 
            'confidence': round(top3_prob[i].item(), 4)
        }
        for i in range(top3_prob.size(0))
    ]

    return {
        'predictions': predictions,
        'metrics': {
            'source_run': loaded_model_uri,
            'total_backend_ms': round((time.time() - start_time) * 1000, 2)
        }
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)