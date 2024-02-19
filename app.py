#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai.vision.all import *


# In[ ]:


from pathlib import Path
from fastcore.xtras import Path
from fastai.learner import load_learner
from fastcore.foundation import L


# In[ ]:


img_path = Path('./static')
images_lis = [ i for i in list(img_path.ls()) if i.is_file()]
images_lis[:3], len(images_lis)


# In[ ]:


def label_to_class(f:str)->L:
    return L(f)


# In[ ]:


learner = load_learner('./model/model_asl_sign03.pkl')


# In[ ]:


import gradio as gr
import os


def pred_image(image):
    dict = {}
    for i, j in zip(learner.dls.vocab, learner.predict(images_lis[i])[2]):
        dict[i] = round(j.item(), 2)
    
    return dict


demo = gr.Interface(
    fn=pred_image,
    inputs='image',
    outputs=gr.Label(num_top_classes=28),
    examples=images_lis,
)

if __name__ == "__main__":
    demo.launch(debug=True)



# In[ ]:





# In[ ]:




