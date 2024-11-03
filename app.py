from transformers import pipeline
import gradio as gr

class_descriptions = {
    "-K": "Potassium deficiency", 
    "-N": "Azote deficiency",
    "-P": "Phosphorus deficiency",
    "FN": "Healthy"
}

pipe = pipeline("image-classification", model="AbdoulayeDIOP/lettuce-npk-vit")

def prediction(img):
    pred = pipe(img)
    # text = "\n".join([
    #     f"Prediction: {class_descriptions[pred[0]['label']]}",
    #     f"Confidence: {pred[0]['score']:.2f}",
    # ])
    return {class_descriptions[v['label']]: v['score'] for v in pred}

app = gr.Interface(prediction, inputs=gr.Image(label="Input Image", type="pil"), outputs=gr.Label(label="Model prediction"))
app.launch(share=True)