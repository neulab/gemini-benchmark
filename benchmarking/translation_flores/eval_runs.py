import subprocess
import argparse
import pandas as pd


def populate_metrics_across_prompts(metrics_name, metrics_xp_dict, prompt_name, prompt_dict):
    for key in metrics_xp_dict.keys():
        metrics_xp_dict[key][prompt_name] = prompt_dict[key][metrics_name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="flores101")
    parser.add_argument("--langs_file" , type=str , required=True)
    args = parser.parse_args()
    results_dir = args.results_dir
    tokenizer = args.tokenizer

    langs = [line.strip() for line in open(args.langs_file, "r")]
    # prompt_datasets = ["tp3", "tt-zero", "tt-one", "tt-three", "tt-five"]
    prompt_datasets = ["tt-zero"]

    name_prefix = "gpt-3.5-turbo tt-def temperature=0.3 context_length=-1"

    bleu_across_prompts = {}
    spbleu200_across_prompts = {}
    chrf_across_prompts = {}
    ter_across_prompts = {}
    slr_across_prompts = {}

    metrics_tuple = ((bleu_across_prompts, "BLEU"), (spbleu200_across_prompts, 'spBLEU200') ,(chrf_across_prompts, "chrF2++"), (ter_across_prompts, "TER"),
                     (slr_across_prompts, "SLR"))

    for lang in langs:
        for item in metrics_tuple:
            item[0][lang] = {}
    temp_scores_fname = "temp_scores.txt"
    for prompt in prompt_datasets:
        results = {}

        for lang in langs:
            fname = f"{results_dir}/{prompt}/{name_prefix}_{prompt}_{lang}.tsv"
            out_file = open(temp_scores_fname, "w")
            subprocess.call(["python", "score.py", fname, "--tokenizer", tokenizer], stdout=out_file)
            out_file.close()
            output = open(temp_scores_fname, "r").read()
            bleu = float(output.split("BLEU = ")[1].split()[0])
            spbleu200  = float(output.split("sp200BLEU = ")[1].split()[0])
            slr = float(output.split("ratio = ")[1].split()[0])
            chrf = float(output.split("chrF2++ = ")[1].split()[0])
            ter = float(output.split("TER = ")[1].split()[0])

            results[lang] = {"BLEU": bleu, "spBLEU200" : spbleu200, "chrF2++": chrf, "TER": ter, "SLR": slr}
        df = pd.DataFrame(results)
        df =df.T
        df.to_csv(prompt + "_scores.tsv", sep="\t")

        for metric in metrics_tuple:
            populate_metrics_across_prompts(metric[1], metric[0], prompt, results)

    for metric in metrics_tuple:
        df = pd.DataFrame(metric[0])
        df.to_csv(metric[1] + "_scores.tsv", sep="\t")


if __name__ == "__main__":
    main()
