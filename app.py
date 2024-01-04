__all__ = ['is_cat', 'learn', 'classify_image', 'categories', 'image', 'label', 'examples', 'intf']

from fastai.vision.all import *
import gradio as gr

def is_cat(x): return x[0].isupper() 

learn = load_learner('model.pkl')

labels = learn.dls.vocab

def classify_image(img):
    pred,pred_idx,probs = learn.predict(img)
    return dict(zip(labels, map(float, probs)))

image = gr.Image(height=224, width=224)
label = gr.Label()
examples = ['siamese.jpg', 'beagle.jpg', 'bagel.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)