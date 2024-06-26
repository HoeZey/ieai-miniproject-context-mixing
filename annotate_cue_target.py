import argparse
import csv
import pickle
import spacy
from transformers import WhisperProcessor, Wav2Vec2Processor
from utils import MODEL_PATH

ORIGINAL_TRANSCRIPTIONS_PATH = "./directory/mfa/common_voice/test/corpus/"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="wav2vec2-large-xlsr-53-french")
    parser.add_argument("--model-output-dir", default="directory/predictions/common_voice/test/wav2vec2-large-xlsr-53-french")
    parser.add_argument("--output-file", default="directory/predictions/common_voice/test/wav2vec2-large-xlsr-53-french/annotated_det_noun.csv")
    parser.add_argument("--target-tag", default="NOUN")
    parser.add_argument("--cue-tag", default="DET")
    parser.add_argument("--allowed-cues", nargs='+', default=['le', 'la', 'les', 'un', 'une', 'des', 'du'])
    parser.add_argument("--banned-cues", nargs='+', default=["de", "d'", "de", "en", "leur"])
    parser.add_argument("--min-wer", type=float, default=0)
    parser.add_argument("--max-wer", type=float, default=100)
    args = parser.parse_args()

    nlp = spacy.load("fr_core_news_md")

    if args.model_name.split('-')[0] == "whisper":
        processor = WhisperProcessor.from_pretrained(MODEL_PATH[args.model_name], task='transcribe', language='french')
    else:
        processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH[args.model_name])

    with open(f"{args.model_output_dir}/wers.pkl", "rb") as f:
        wers = pickle.load(f)
    with open(f"{args.model_output_dir}/generated_ids.pkl", "rb") as f:
        generated_ids = pickle.load(f)

    with open(args.output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'sentence_true', 'sentence_pred', 'target_true', 'target_pred', 'cue'])

        for i, (wer, gen_ids) in enumerate(zip(wers, generated_ids)):
            if args.min_wer < wer < args.max_wer:
                predicted_transcript = processor.decode(gen_ids)
                if "whisper" in args.model_name:
                    predicted_transcript = predicted_transcript.split(">")[3].split("<")[0].strip()

                predicted_doc = nlp(predicted_transcript)
                predicted_words = [word.text.lower() for word in predicted_doc]
                predicted_tags = [word.tag_ for word in predicted_doc]

                with open(f"{ORIGINAL_TRANSCRIPTIONS_PATH}/{i}.txt") as f:
                    true_transcript = f.readline().strip()

                true_doc = nlp(true_transcript)
                true_words = [word.text.lower() for word in true_doc]
                true_tags = [word.tag_ for word in true_doc]

                for j, (word, tag) in enumerate(zip(true_words, true_tags)):
                    if (len(true_doc) == len(predicted_doc) + 1 and
                        word not in predicted_words and
                        tag == args.target_tag and
                        (word.rsplit('s', 1)[0] in predicted_words or
                         word + 's' in predicted_words or
                         word.rsplit('es', 1)[0] in predicted_words or
                         word + 'es' in predicted_words or
                         word.rsplit('nt', 1)[0] in predicted_words or
                         word + 'nt' in predicted_words or
                         word.rsplit('ent', 1)[0] in predicted_words or
                         word + 'ent' in predicted_words)):

                        if true_doc[j - 1].pos_ == args.cue_tag or true_doc[j - 1].pos_ in args.allowed_cues:
                            cue_word = true_words[j - 1]
                            if true_words[j - 1] != predicted_words[j - 1]:
                                continue
                        elif true_doc[j - 2].pos_ == args.cue_tag or true_doc[j - 2].pos_ in args.allowed_cues:
                            cue_word = true_words[j - 2]
                            if true_words[j - 2] != predicted_words[j - 2]:
                                continue
                        else:
                            continue

                        if cue_word in args.banned_cues:
                            continue

                        pred_word = predicted_words[j]
                        writer.writerow([i, true_transcript, predicted_transcript, word, pred_word, cue_word])
                        print(i, true_transcript, predicted_transcript)
