## CookBookCreater


The code has been tested on PyTorch 2.1.0
To install the dependencies, run <pre/>pip install -r requirements.txt</pre> 

### Recipe1M Dataset
1. Download Recipe1M datasets from the original <a href="http://pic2recipe.csail.mit.edu/">website</a> [please note that you may need to email the authors to get access to the dataset]
2. To attain annotation files from dataset files in the appropriate format for training and evaluation use: <pre>dataset_cleaning.ipynb</pre> Set the paths to your data files accordingly.

### Finetuned checkpoints:
BLIP w/ ViT-B | BLIP w/ ViT-L 
--- | :---:
<a href="https://mbzuaiac-my.sharepoint.com/:u:/g/personal/ayesha_ishaq_mbzuai_ac_ae/ET55uP9BoHpJqwrX2Z63USQB99VvxHh3GZtZJejxkyCjEw?e=7cVm0T">Download</a>| <a href="https://mbzuaiac-my.sharepoint.com/:u:/g/personal/ayesha_ishaq_mbzuai_ac_ae/EWvdy_BnFHNNseo7PNDnzugBYxnTY57NfnuYgZBOrz6s3g?e=mN4gfu">Download</a> | 

### Pretrained checkpoints:
BLIP w/ ViT-B | BLIP w/ ViT-L 
--- | :---:
<a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth">Download</a>| <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth">Download</a> | 

### Recipe Inference demo:
To run inference on any food image run:
<pre>python inference.py --weight "path to weights" --input "path to image"</pre> 

### Recipe Title Captioning:
1. Set 'image_root' in configs/custom_recipe1m.yaml accordingly.
2. To evaluate the finetuned BLIP model on Recipe1M, first set 'pretrained' in configs/custom_recipe1m.yaml to the appropriate weights from above and vit as 'base' or 'large' corresponding to the weights. Then run:
<pre>python -m torch.distributed.run --nproc_per_node=1 evaluate.py </pre> 
3. To finetune the pre-trained checkpoint on a single GPU, first set 'pretrained' in configs/custom_recipe1m.yaml to pretrained checkpoints from above. Then run:
<pre>python -m torch.distributed.run --nproc_per_node=1 train_caption_recipe.py </pre> 


### Acknowledgement
The implementation of CookBook Creater relies on resources from <a href="https://huggingface.co/meta-llama/Llama-2-7b-chat-hf">LLaMa from META</a>, <a href="https://huggingface.co/lmsys/vicuna-7b-v1.5-16k">Vicuna</a>, <a href="https://github.com/salesforce/BLIP">BLIP</a>, <a href="https://github.com/huggingface/transformers">Huggingface Transformers</a>, and <a href="https://github.com/rwightman/pytorch-image-models/tree/master/timm">timm</a>. We thank the original authors for their open-sourcing.

Please note that LLaMa-2 7B from META is a gated model and to use it you must request access on this <a href="https://huggingface.co/meta-llama/Llama-2-7b-chat-hf">link</a>. 
