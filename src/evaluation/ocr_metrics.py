import editdistance

def cer(reference: str, hypothesis: str) -> float:
    ref = reference.replace(" ", "")
    hyp = hypothesis.replace(" ", "")
    return editdistance.eval(ref, hyp) / max(len(ref), 1)


def wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    return editdistance.eval(ref_words, hyp_words) / max(len(ref_words), 1)

