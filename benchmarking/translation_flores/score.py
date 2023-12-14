import argparse
import numpy as np
import pandas as pd
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF, TER

COMET_MODEL = "wmt20-comet-da"
COMET_BATCH_SIZE = 64
BLEURT_BATCH_SIZE = 64


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("res", type=str)
    parser.add_argument("--tokenizer", type=str, default="flores101")
    parser.add_argument("--no-lexical-metrics", action="store_true")
    parser.add_argument("--comet-dir", type=str, default=None)
    parser.add_argument("--bleurt-dir", type=str, default=None)
    parser.add_argument("--src", type=str)
    parser.add_argument("--save-segment-level", default=None)

    args = parser.parse_args()

    df = pd.read_csv(args.res, sep="\t")
    hyps = [str(line) for line in df["label"].tolist()]
    refs = [str(line) for line in df["predictions"].tolist()]

    sentence_metrics = [[] for _ in range(len(refs))]

    if not args.no_lexical_metrics:

        # gets corpus-level non-ml evaluation metrics
        # corpus-level BLEU
        tokenize = args.tokenizer
        try:
            print(sacrebleu.corpus_bleu(hyps, [refs], tokenize=tokenize).format())
        except:
            print("BLEU = -1")
        try:
            print("sp200"+sacrebleu.corpus_bleu(hyps, [refs], tokenize='flores200').format())
        except:
            print("spBLEU200 = -1")
        chrf = CHRF(word_order=2)
        try:
            print(chrf.corpus_score(hyps, [refs]))
        except:
            print("chrF2++ = -1")
        try:
            print(sacrebleu.corpus_ter(hyps, [refs]).format())
        except:
            print("TER = -1")
        if args.save_segment_level is not None:
            # gets sentence-level non-ml metrics
            for i, (hyp, ref) in enumerate(zip(hyps, refs)):
                sentence_metrics[i].append(
                    ("bleu", sacrebleu.sentence_bleu(hyp, [ref]).score)
                )
                sentence_metrics[i].append(
                    ("chrf", sacrebleu.sentence_chrf(hyp, [ref]).score)
                )
                sentence_metrics[i].append(
                    ("ter", sacrebleu.sentence_ter(hyp, [ref]).score)
                )

    if args.comet_dir is not None:
        from comet import download_model, load_from_checkpoint

        assert args.src is not None, "source needs to be provided to use COMET"
        with open(args.src) as src_f:
            srcs = [line.strip() for line in src_f.readlines()]

        # download comet and load
        comet_path = download_model(COMET_MODEL, args.comet_dir)
        comet_model = load_from_checkpoint(comet_path)

        print("Running COMET evaluation...")
        comet_input = [
            {"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(srcs, hyps, refs)
        ]
        # sentence-level and corpus-level COMET
        comet_sentscores, comet_score = comet_model.predict(
            comet_input, batch_size=COMET_BATCH_SIZE, sort_by_mtlen=True
        )
        for i, comet_sentscore in enumerate(comet_sentscores):
            sentence_metrics[i].append(("comet", comet_sentscore))

        print(f"COMET = {comet_score:.4f}")

    # gets BLEURT scores
    if args.bleurt_dir is not None:
        from bleurt import score

        checkpoint = args.bleurt_dir

        bleurt_scorer = score.LengthBatchingBleurtScorer(checkpoint)
        bleurt_scores = bleurt_scorer.score(
            references=refs, candidates=hyps, batch_size=BLEURT_BATCH_SIZE
        )
        assert type(bleurt_scores) == list
        # corpus-level BLEURT
        print(f"BLEURT = {np.array(bleurt_scores).mean():.4f}")
        for i, bleurt_score in enumerate(bleurt_scores):
            sentence_metrics[i].append(("bleurt", bleurt_score))

    # saves segment-level scores to the disk
    if args.save_segment_level is not None:
        with open(args.save_segment_level, "w") as f:
            for metrics in sentence_metrics:
                print(
                    " ".join(
                        f"{metric_name}={value}" for metric_name, value in metrics
                    ),
                    file=f,
                )


if __name__ == "__main__":
    main()
