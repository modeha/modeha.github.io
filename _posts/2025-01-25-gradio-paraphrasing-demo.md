---
layout: post
title: Gradio Paraphrasing Demo
date: 2025-01-25 13:10 -0500
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
    # Preprocess the text
    input_ids = tokenizer.encode("paraphrase: " + input_text, return_tensors="pt", truncation=True)

    # Generate paraphrased versions
    outputs = model.generate(
        input_ids,
        max_length=256,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    # Decode the generated sequences
    paraphrased_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return paraphrased_texts

title = "Paraphrasing with T5 Model"
description = "This demo uses the T5 model fine-tuned on PAWS for generating paraphrases of input text. Simply enter your text and click 'Submit' to generate multiple paraphrased versions."
article = "<p style='text-align: center'><a href='https://huggingface.co/Vamsi/T5_Paraphrase_Paws'>Hugging Face Model</a> | <a href='https://github.com/huggingface/transformers'>Transformers Library</a></p>"

examples = [
    ["Intelligence Analyst Trainer - 24-968-08-086 All information required to determine suitability for employment with the Canadian Security Intelligence Service is collected under the authority of the Canadian Security Intelligence Service Act."],
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
    examples=examples,
)


# iface.launch(debug=True)


```

    You are using the default legacy behaviour of the <class 'transformers.models.t5.tokenization_t5.T5Tokenizer'>. This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thoroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565
    c:\Users\mohse\.conda\envs\cdo_idp\lib\site-packages\torchvision\io\image.py:13: UserWarning: Failed to load image Python extension: Could not find module 'C:\Users\mohse\.conda\envs\cdo_idp\Lib\site-packages\torchvision\image.pyd' (or one of its dependencies). Try using the full path with constructor syntax.
      warn(f"Failed to load image Python extension: {e}")
    

    Running on local URL:  http://0.0.0.0:7860
    Running on public URL: https://f1f8800d90672a6ac0.gradio.live
    
    This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run `gradio deploy` from Terminal to deploy to Spaces (https://huggingface.co/spaces)
    


<div><iframe src="https://f1f8800d90672a6ac0.gradio.live" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>





    




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
    ["Intelligence Analyst Trainer - 24-968-08-086 All information required to determine suitability for employment with the Canadian Security Intelligence Service is collected under the authority of the Canadian Security Intelligence Service Act."],
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
    


<div><iframe src="https://2c03b010d116b96362.gradio.live" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>





    



    Traceback (most recent call last):
      File "c:\Users\mohse\.conda\envs\cdo_idp\lib\site-packages\gradio\queueing.py", line 536, in process_events
        response = await route_utils.call_process_api(
      File "c:\Users\mohse\.conda\envs\cdo_idp\lib\site-packages\gradio\route_utils.py", line 322, in call_process_api
        output = await app.get_blocks().process_api(
      File "c:\Users\mohse\.conda\envs\cdo_idp\lib\site-packages\gradio\blocks.py", line 1935, in process_api
        result = await self.call_function(
      File "c:\Users\mohse\.conda\envs\cdo_idp\lib\site-packages\gradio\blocks.py", line 1520, in call_function
        prediction = await anyio.to_thread.run_sync(  # type: ignore
      File "c:\Users\mohse\.conda\envs\cdo_idp\lib\site-packages\anyio\to_thread.py", line 33, in run_sync
        return await get_asynclib().run_sync_in_worker_thread(
      File "c:\Users\mohse\.conda\envs\cdo_idp\lib\site-packages\anyio\_backends\_asyncio.py", line 877, in run_sync_in_worker_thread
        return await future
      File "c:\Users\mohse\.conda\envs\cdo_idp\lib\site-packages\anyio\_backends\_asyncio.py", line 807, in run
        result = context.run(func, *args)
      File "c:\Users\mohse\.conda\envs\cdo_idp\lib\site-packages\gradio\utils.py", line 826, in wrapper
        response = f(*args, **kwargs)
      File "C:\Users\mohse\AppData\Local\Temp\ipykernel_29716\3481243923.py", line 11, in paraphrase_text
        outputs = model.generate(
      File "c:\Users\mohse\.conda\envs\cdo_idp\lib\site-packages\torch\utils\_contextlib.py", line 116, in decorate_context
        return func(*args, **kwargs)
      File "c:\Users\mohse\.conda\envs\cdo_idp\lib\site-packages\transformers\generation\utils.py", line 2227, in generate
        beam_scorer = BeamSearchScorer(
      File "c:\Users\mohse\.conda\envs\cdo_idp\lib\site-packages\transformers\generation\beam_search.py", line 200, in __init__
        raise ValueError(
    ValueError: `num_beams` has to be an integer strictly greater than 1, but is 4.25. For `num_beams` == 1, one should make use of `greedy_search` instead.
    
