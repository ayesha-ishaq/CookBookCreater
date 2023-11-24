import argparse
import os
import ruamel.yaml as yaml
import json
from pathlib import Path
from pathlib import Path
import os
import argparse
import torch
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu

import LLaMa 
import Vicuna

def main(args):
        
    if args.model_llm == "llama":
        generator = LLaMa.RecipeGenerator(token="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")
    else:
        generator = Vicuna.RecipeGenerator()
    
    # Initialize ROUGE and BLEU calculators
    rouge_calculator = Rouge()

    # Calculate scores for each pair
    rouge_scores = []
    bleu_scores = []

    f = open(args.gt_path,) 
    ground_truth = json.load(f)

    for i, gt in enumerate(ground_truth):
        recipe = gt['recipe']
        title = gt['title']
        print(title)
        predicted_recipe_l = generator.generate(text=title)

        # Calculate ROUGE scores
        rouge_l_score_l = rouge_calculator.get_scores(predicted_recipe_l, recipe)[0]['rouge-l']['f']
        
        rouge_scores.append(rouge_l_score_l)
        
        # Calculate BLEU score
        bleu_score_l =  sentence_bleu([predicted_recipe_l], [recipe])
        bleu_scores.append(bleu_score_l)

        if i%10 == 0:
        # Aggregate scores
            aggregate_rouge_scores_l = sum(rouge_scores) / len(rouge_scores)

            aggregate_bleu_score_l = sum(bleu_scores) / len(bleu_scores)

            scores = {'ROUGE_L':aggregate_rouge_scores_l, 'BLEU': aggregate_bleu_score_l}
            
            log_stats = {**{f'{k}': v for k, v in scores.items()}}
            with open(os.path.join(args.output_dir, "log_eval_recipe_{}.txt".format(args.model_llm)),"a") as f:
                f.write(json.dumps(log_stats) + "\n")   


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', default='/home/ayesha.ishaq/Desktop/AI701/CookBookCreater/val_recipe.json')
    parser.add_argument('--output_dir', default='./output/evalution_val')       
    parser.add_argument('--model_llm', default='llama')
    args = parser.parse_args()

    # config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True) 
    
    main(args)