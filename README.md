## CookBookCreater


The code has been tested on PyTorch 2.1.0
To install the dependencies, run <pre/>pip install -r requirements.txt</pre> 

### Recipe Inference demo:
To run inference on any food image run:
<pre>python inference.py --weight "path to weights" --input "path to image"</pre> 

### Recipe Title Captioning:
1. Download Recipe1M datasets from the original <a href="http://pic2recipe.csail.mit.edu/">website</a>, and set 'image_root' in configs/custom_recipe1m.yaml accordingly.
2. To evaluate the finetuned BLIP model on Recipe1M, run:
<pre>python -m torch.distributed.run --nproc_per_node=1 evaluate.py </pre> 
3. To finetune the pre-trained checkpoint on a single GPU, first set 'pretrained' in configs/custom_recipe1m.yaml as "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth". Then run:
<pre>python -m torch.distributed.run --nproc_per_node=1 train_caption_recipe.py </pre> 


### Acknowledgement
The implementation of CookBook Creater relies on resources from <a href="https://huggingface.co/meta-llama/Llama-2-7b-chat-hf">LLaMa from META</a>, <a href="https://huggingface.co/lmsys/vicuna-7b-v1.5-16k">Vicuna</a>, <a href="https://github.com/salesforce/BLIP">BLIP</a>, <a href="https://github.com/huggingface/transformers">Huggingface Transformers</a>, and <a href="https://github.com/rwightman/pytorch-image-models/tree/master/timm">timm</a>. We thank the original authors for their open-sourcing.
