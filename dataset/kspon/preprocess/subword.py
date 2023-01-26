import os
import sentencepiece as spm


def train_sentencepiece(transcripts, datapath: str = './data', vocab_size: int = 5000):
    print('generate_sentencepiece_input..')

    if not os.path.exists(datapath):
        os.mkdir(datapath)

    with open('sentencepiece_input.txt', 'w', encoding="utf-8") as f:
        for transcript in transcripts:
            f.write(f'{transcript}\n')

    spm.SentencePieceTrainer.Train(
        f"--input={datapath}/sentencepiece_input.txt "
        f"--model_prefix=kspon_sentencepiece "
        f"--vocab_size={vocab_size} "
        f"--model_type=bpe "
        f"--pad_id=0 "
        f"--bos_id=1 "
        f"--eos_id=2 "
        f"--unk_id=3 "
        f"--user_defined_symbols={blank_token}"
    )


def sentence_to_subwords(audio_paths: list, transcripts: list, datapath: str = './data'):
    subwords = list()

    print('sentence_to_subwords...')

    sp = spm.SentencePieceProcessor()
    vocab_file = "kspon_sentencepiece.model"
    sp.load(vocab_file)

    with open(f'{datapath}/transcripts.txt', 'w') as f:
        for audio_path, transcript in zip(audio_paths, transcripts):
            audio_path = audio_path.replace('txt', 'pcm')
            subword_transcript = " ".join(sp.EncodeAsPieces(transcript))
            subword_id_transcript = " ".join([str(item) for item in sp.EncodeAsIds(transcript)])
            f.write(f'{audio_path}\t{subword_transcript}\t{subword_id_transcript}\n')

    return subwords
