#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai.vision.all import *
import gradio as gr
import os


# In[ ]:


from pathlib import Path
from fastcore.xtras import Path
from fastai.learner import load_learner
from fastcore.foundation import L


# In[2]:


img_path = Path('./static')
images_lis = [ i for i in list(img_path.ls()) if i.is_file()]
images_lis[:3], len(images_lis)


# In[3]:


def label_to_class(f:str)->L:
    return L(f)


# In[4]:


learner = load_learner('./model/asl_sign_multi_resnet18_03.pkl')


# In[ ]:


def pred_image(image):
    dict = {}
    for i, j in zip(learner.dls.vocab, learner.predict(image)[2]):
        dict[i] = round(j.item(), 2)
    
    return dict


demo = gr.Interface(
    fn=pred_image,
    inputs='image',
    outputs=gr.Label(num_top_classes=29),
    examples=images_lis,
)

if __name__ == "__main__":
    demo.launch(debug=True)



# In[ ]:





# In[ ]:




