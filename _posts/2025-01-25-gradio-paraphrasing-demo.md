---
layout: post
title: Gradio Paraphrasing Demo
date: 2025-01-25 13:10 -0500
---


## Paraphrasing Text with T5 Model Using Gradio

Paraphrasing is a crucial task in natural language processing (NLP) that involves rephrasing text while retaining its original meaning. 
It has numerous applications, such as content generation, text simplification, and improving readability. In this post,
 we will demonstrate how to build a simple yet effective paraphrasing tool using the T5 (Text-to-Text Transfer Transformer) model, 
 fine-tuned on the PAWS (Paraphrase Adversaries from Word Scrambling) dataset.

We'll leverage the **Hugging Face Transformers** library to load a pre-trained T5 model and the **Gradio** 
library to create an interactive web-based interface that allows users to input text and generate multiple paraphrased outputs.
 Gradio provides an intuitive and user-friendly way to deploy machine learning models with minimal effort.

### Key Features of the Paraphrasing Tool
- **Model**: We use the `Vamsi/T5_Paraphrase_Paws` model, fine-tuned to generate high-quality paraphrases.
- **Input**: Users can enter text into a textbox to be paraphrased.
- **Output**: The tool generates multiple paraphrased versions of the input text.
- **Customization Options**: Users can adjust the number of paraphrases and beam search size to fine-tune the results.

### How It Works
1. The input text is tokenized and processed using the pre-trained T5 tokenizer.
2. The model generates paraphrased outputs by applying beam search, allowing the user to control the number of generated sequences.
3. The results are displayed in a user-friendly interface powered by Gradio.
4. Example inputs are provided to help users explore the tool's capabilities.

Below is the implementation of the paraphrasing tool as well you can try it yourself:

---

```python
pip install transformers torch
pip install sentencepiece protobuf

```
```python
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")

def paraphrase_text(input_text, num_return_sequences=3, num_beams=5):
    input_ids = tokenizer.encode("paraphrase: " + input_text, return_tensors="pt", truncation=True)

    outputs = model.generate(
        input_ids,
        max_length=256,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    paraphrased_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return paraphrased_texts

# Personal Information Section
title = "Paraphrasing with T5 Model - Created by Mohsen Dehghani"
description = """
## Welcome to the Paraphrasing Demo
This demo uses the T5 model fine-tuned on PAWS to generate paraphrases of input text.
Simply enter your text and click 'Submit' to see the results.

**Created by:** Mohsen Dehghani  
**Email:** mohsen.dehghani@gmail.com  
**Location:** Montreal, Quebec, Canada  
"""

article = """
<p style='text-align: center'>
<a href='https://huggingface.co/Vamsi/T5_Paraphrase_Paws'>Hugging Face Model</a> | 
<a href='https://github.com/huggingface/transformers'>Transformers Library</a>
</p>
"""

examples = [
    ["Intelligence Analyst Trainer - 24-968-08-086 All information required to determine suitability for employment
     with the Canadian Security Intelligence Service is collected under the authority of the Canadian Security Intelligence Service Act."],
    ["The Privacy Act allows candidates to review collected information and request amendments."],
    ["Machine learning models are improving rapidly, and their applications are expanding across different industries."]
]

iface = gr.Interface(
    fn=paraphrase_text,
    inputs=[
        gr.Textbox(lines=5, placeholder="Enter text to paraphrase here..."),
        gr.Slider(1, 5, value=3, label="Number of paraphrases"),
        gr.Slider(1, 10, value=5, label="Beam search size")
    ],
    outputs=gr.Textbox(label="Paraphrased Outputs"),
    title=title,
    description=description,
    article=article,
    examples=examples
)

iface.launch(share=True)

```
    Running on local URL:  http://127.0.0.1:7860
    Running on public URL: https://2c03b010d116b96362.gradio.live
    
    This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run `gradio deploy` from Terminal to deploy to Spaces (https://huggingface.co/spaces)

    
<a href="https://huggingface.co/spaces/MohsenDehghani/paraphrasing" target="_blank">
    <button style="padding: 10px 20px; font-size: 16px;">Visit the Paraphrasing Tool on my Hugging Face</button>
</a>


