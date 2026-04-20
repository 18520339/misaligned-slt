'''Trim mBART vocabulary to tokens used in a specific dataset's target text.

Usage:
    python trim_mbart.py \
        --data-path data/PHOENIX-2014-T-release-v3/phoenix_train.gz \
        --mbart-name facebook/mbart-large-cc25 \
        --tgt-lang de_DE \
        --output-dir ./trimmed_mbart
'''
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import MBartTokenizer, MBartForConditionalGeneration
from hftrim.TokenizerTrimmer import TokenizerTrimmer
from hftrim.ModelTrimmers import MBartTrimmer

from loader import load_dataset_file


def collect_texts(data_path: str) -> list:
    '''Collect all target texts from the dataset pickle, in deterministic order.'''
    raw = load_dataset_file(data_path)
    keys = sorted(raw.keys())
    texts = [raw[k]['text'] for k in keys]
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True, help='Path to dataset pickle (gzip or plain).')
    parser.add_argument('--mbart-name', default='facebook/mbart-large-cc25')
    parser.add_argument('--tgt-lang', required=True, help="Target language code (e.g. 'de_DE').")
    parser.add_argument('--output-dir', default='./trimmed_mbart')
    args = parser.parse_args()

    texts = collect_texts(args.data_path)
    print(f'Collected {len(texts)} texts from {args.data_path}')

    tokenizer = MBartTokenizer.from_pretrained(
        args.mbart_name, src_lang=args.tgt_lang, tgt_lang=args.tgt_lang, use_fast=False,
    )

    tt = TokenizerTrimmer(tokenizer)
    tt.make_vocab(texts)

    # Ensure special tokens and target language code stay in the trimmed vocab
    extra_ids = set()
    for tok in (tokenizer.pad_token, tokenizer.bos_token, tokenizer.eos_token,
                tokenizer.unk_token, tokenizer.mask_token):
        if tok is not None:
            extra_ids.add(tokenizer.convert_tokens_to_ids(tok))
    if hasattr(tokenizer, 'lang_code_to_id'):
        for code, idx in tokenizer.lang_code_to_id.items():
            if code == args.tgt_lang:
                extra_ids.add(idx)

    for idx in extra_ids:
        if idx not in tt.trimmed_vocab_ids:
            tt.trimmed_vocab_ids.append(idx)
    tt.trimmed_vocab_ids = sorted(set(tt.trimmed_vocab_ids))

    tt.make_tokenizer()
    tt.trimmed_tokenizer.save_pretrained(args.output_dir)
    print(f'Saved trimmed tokenizer to {args.output_dir}')

    model = MBartForConditionalGeneration.from_pretrained(args.mbart_name)
    mt = MBartTrimmer(model, model.config, tt.trimmed_tokenizer)
    mt.make_weights(tt.trimmed_vocab_ids)
    mt.make_model()
    mt.trimmed_model.save_pretrained(args.output_dir)
    print(f'Saved trimmed model to {args.output_dir}')


if __name__ == '__main__':
    main()
