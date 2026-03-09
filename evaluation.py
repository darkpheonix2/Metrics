"""
=============================================================================
  Comprehensive RAG Evaluation Pipeline
=============================================================================

  Metrics Covered
  ───────────────────────────────────────────────────────────────────────────
  GROUP A — Standard NLP  (answer vs Response, token-overlap)
  ───────────────────────────────────────────────────────────────────────────
  • EM                  Exact Match (SQuAD normalisation)
  • Token_F1            Token-overlap F1  (SQuAD style)
  • Token_Precision     Token-overlap Precision
  • Token_Recall        Token-overlap Recall
  • METEOR              NLTK METEOR (unigram F-measure fallback)
  • BLEU                Sentence BLEU-4 with smoothing
  • BERTScore_P/R/F1    Deep-contextual P/R/F1  (bert-score, GPU batch)
  
  • AUC_ROC             Batch ROC-AUC  (label=EM, score=token-F1)
  • LCS                 Longest Common Subsequence ratio  (ROUGE-L style)
  • Jaccard_Similarity  Token-set Jaccard
  • Hamming_Distance    Character-level Hamming  (padded)

  GROUP B — LLM-based Answer P / R / F1  (answer vs Response)
  ───────────────────────────────────────────────────────────────────────────
  • NLI_Precision       LLM: does Response entail answer?   (0/1)
  • NLI_Recall          LLM: does answer   entail Response? (0/1)
  • NLI_F1              Harmonic mean of NLI P and R
  • GEval_Precision     LLM CoT score 0–1: how much of Response is correct
                        relative to answer  (G-Eval / Prometheus style)
  • GEval_Recall        LLM CoT score 0–1: how much of answer is covered
                        by Response
  • GEval_F1            Harmonic mean of GEval P and R
  • Emb_Precision       Sentence-embedding greedy-match P  (BERTScore-style,
                        all-MiniLM-L6-v2)
  • Emb_Recall          Sentence-embedding greedy-match R
  • Emb_F1              Harmonic mean of Emb P and R

  GROUP C — Retrieval P / R / F1  (context vs supporting_context, text-only)
  ───────────────────────────────────────────────────────────────────────────
  • Retrieval_Precision  fraction of context sentences that are semantically
                         matched in supporting_context  (cosine ≥ threshold)
  • Retrieval_Recall     fraction of supporting_context sentences covered by
                         retrieved context
  • Retrieval_F1         harmonic mean

  GROUP D — RAG-specific  (RAGAS / Adaptive-RAG / IrCoT / GraphRAG)
  ───────────────────────────────────────────────────────────────────────────
  • Faithfulness         RAGAS §3.1 — statement-level NLI vs context
  • Context_Precision    RAGAS §3.2 — MAP over context sentences
  • Context_Recall       RAGAS §3.3 — sentence attribution to context
  • Answer_Relevance     RAGAS §3.4 — reverse-question cosine similarity

  GROUP E — Accuracy  (existing class, unchanged)
  ───────────────────────────────────────────────────────────────────────────
  • Token_Accuracy       Token containment accuracy
  • Fuzzy_Accuracy       Fuzzy sliding-window accuracy
  • Meaning_Accuracy_LLM LLM 0–1 meaning score
  • Meaning_Accuracy_Emb Cosine semantic similarity

  GROUP F — Error metrics
  ───────────────────────────────────────────────────────────────────────────
  • Is_Error             1 if token-F1 < threshold (row-level)
  • Global_Error_Rate    fraction of erroneous rows (batch)
  • Error_Detection_Rate TP / (TP + FN)
  • Error_Rejection_Rate correctly-rejected errors / total errors

  DataFrame columns expected
  ──────────────────────────
  context | answer | supporting_context | Response |
  context_token_count | Analysis | Aspect

  Usage
  ─────
  # Full pipeline (all LLM metrics)
  python rag_evaluation.py --data data.csv --model mistralai/Mistral-7B-Instruct-v0.2 --gpu 0

  # Embedding-only (no LLM calls, much faster)
  python rag_evaluation.py --data data.csv --no-llm

  # Custom retrieval similarity threshold
  python rag_evaluation.py --data data.csv --ret-threshold 0.65
=============================================================================
"""

# ── std-lib ───────────────────────────────────────────────────────────────────
import re
import os
import argparse
import warnings
from collections import Counter
from difflib import SequenceMatcher

warnings.filterwarnings("ignore")

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine


# =============================================================================
# 0.  DEVICE HELPER
# =============================================================================

def _get_cuda(gpu_id: int = 0) -> str:
    """
    CUDA_VISIBLE_DEVICES remaps physical GPUs to logical indices starting at 0.
    After setting CUDA_VISIBLE_DEVICES=N, the only valid device is cuda:0.
    The gpu_id arg now selects which physical GPU to expose (set in env), 
    but the logical device inside the process is always cuda:0.
    """
    if not torch.cuda.is_available():
        return "cpu"
    n_visible = torch.cuda.device_count()
    logical_id = min(gpu_id, n_visible - 1)   # clamp to valid range
    return f"cuda:{logical_id}"

# =============================================================================
# 1.  EXISTING CLASSES  ── kept 100 % identical to uploaded code
#     Only additions: max_new_tokens in .generate() (OOM safety)
#                     gpu_id param (multi-GPU support)
# =============================================================================

class Faithfulness:
    """
    RAGAS §3.1 Faithfulness.
    Step 1 — decompose Response into atomic statements (LLM).
    Step 2 — NLI-style verdict per statement vs context (LLM).
    Score = |supported statements| / |total statements|
    """
    def __init__(self, question, answer, llm, tokenizer, context, gpu_id=0):
        self.question  = question
        self.answer    = answer
        self.llm       = llm
        self.tokenizer = tokenizer
        self.context   = context
        self.dev       = _get_cuda(gpu_id)

    def statement_creator(self):
        prompt = (
            "You are an expert at generating statements.\n\n"
            "Given a question and answer, create one or more statements from "
            "each sentence in the given answer.\n"
            f"question: {self.question}\nanswer: {self.answer}\nStatement:"
        )
        inputs       = self.tokenizer(prompt, return_tensors="pt",
                                      truncation=True, max_length=2048)
        input_ids    = inputs.input_ids.to(self.dev)
        attn_mask    = inputs.attention_mask.to(self.dev)
        pad_token_id = self.tokenizer.eos_token_id
        output_ids   = self.llm.generate(
            input_ids, attention_mask=attn_mask,
            pad_token_id=pad_token_id, max_new_tokens=256
        )
        output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if "Statement:" in output:
            statement = output.split("Statement:")[-1].strip().split("\n")[0]
        else:
            statement = output.strip().split("\n")[-1]
        return statement.replace('"', '').replace("'", "")

    def calculate_faithfulness_score(self, output):
        lines = output.strip().split('\n')
        yes_count, total_statements = 0, 0
        verdict_patterns = [
            r'Verdict:\s*(Yes|No|Correct|Incorrect)',
            r'Statement\s+\d+(?:-\d+)?:\s*(Yes|No|Correct|Incorrect)',
            r'Final verdict:\s*(Yes|No|Correct|Incorrect)',
        ]
        for line in lines:
            line = line.strip()
            for pattern in verdict_patterns:
                matches = re.findall(pattern, line, re.IGNORECASE)
                for match in matches:
                    total_statements += 1
                    if match.lower() in ['yes', 'correct']:
                        yes_count += 1
                    break

        negative_indicators = [
            r'not directly supported', r'not supported',
            r'verdict:\s*no', r'false', r'incorrect'
        ]
        positive_indicators = [
            r'verdict:\s*yes', r'verdict:\s*correct',
            r'supported by', r'true', r'correct'
        ]

        if total_statements == 0:
            statement_count, yes_from_explanations = 0, 0
            for line in lines:
                if re.match(r'^\d+\.', line.strip()):
                    statement_count += 1
            for line in lines:
                ll      = line.lower()
                has_neg = any(re.search(i, ll) for i in negative_indicators)
                has_pos = any(re.search(i, ll) for i in positive_indicators)
                if has_pos and not has_neg:
                    yes_from_explanations += 1
                elif ll.strip().endswith('yes.'):
                    yes_from_explanations += 1
            if statement_count > 0:
                total_statements = statement_count
                yes_count        = yes_from_explanations

        if total_statements == 0:
            return 0.0
        return yes_count / total_statements

    def faithfulness(self):
        statements = self.statement_creator()
        prompt = (
            "You are an expert at rating the statements to their context.\n\n"
            "Consider the given context and following statements, then determine "
            "whether they are supported by the information present in the context. "
            "Provide a brief explanation for each and every statement before "
            "arriving at the verdict (Yes/No). Provide a final verdict for each "
            "and every statement in order at the end in the given format. "
            "The verdict should be binary output yes or no. Stick to instructions.\n"
            f"Context: {self.context}\nStatement: {statements}"
        )
        inputs       = self.tokenizer(prompt, return_tensors="pt",
                                      truncation=True, max_length=3000)
        input_ids    = inputs.input_ids.to(self.dev)
        attn_mask    = inputs.attention_mask.to(self.dev)
        pad_token_id = self.tokenizer.eos_token_id
        output_ids   = self.llm.generate(
            input_ids, attention_mask=attn_mask,
            pad_token_id=pad_token_id, max_new_tokens=512
        )
        output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        output = output.strip().replace('"', '').replace("'", "")
        return self.calculate_faithfulness_score(output)


class Relevance:
    """
    RAGAS §3.4 Answer Relevance.
    Generates a reverse question from the answer; measures cosine-sim
    with the original question using sentence embeddings.
    """
    def __init__(self, question, answer, llm, tokenizer, gpu_id=0):
        self.question  = question
        self.answer    = answer
        self.llm       = llm
        self.tokenizer = tokenizer
        self.dev       = _get_cuda(gpu_id)

    def question_embedding(self, question):
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        return model.encode(question)

    def answer_relevance(self):
        prompt = (
            "You are an expert at creating questions based on the answer.\n\n"
            "Generate one question for the given answer. "
            "Do not add any extra information.\n"
            f"answer: {self.answer}\nQuestion:"
        )
        inp          = self.tokenizer(prompt, return_tensors="pt",
                                      truncation=True, max_length=2048)
        input_ids    = inp.input_ids.to(self.dev)
        attn_mask    = inp.attention_mask.to(self.dev)
        pad_token_id = self.tokenizer.eos_token_id
        output_ids   = self.llm.generate(
            input_ids, attention_mask=attn_mask,
            pad_token_id=pad_token_id, max_new_tokens=128
        )
        pred_ques = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        pred_ques = pred_ques.strip('Question:').replace('"', '').replace("'", "")
        aq  = self.question_embedding(self.question).reshape(1, -1)
        gq  = self.question_embedding(pred_ques).reshape(1, -1)
        return float(sk_cosine(aq, gq)[0][0])


class Retrieval_metrics:
    """Document-level retrieval P/R/F1 — pass document ID lists."""
    def __init__(self, retrieved_docs, actual_relevant_docs):
        self.retrieved_docs       = retrieved_docs
        self.actual_relevant_docs = actual_relevant_docs

    def calculate_retrieval_metrics(self):
        retrieved_set = set(self.retrieved_docs)
        relevant_set  = set(self.actual_relevant_docs)
        tp  = len(retrieved_set & relevant_set)
        fp  = len(retrieved_set - relevant_set)
        fn  = len(relevant_set  - retrieved_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0)
        return dict(precision=precision, recall=recall,
                    f1_score=f1, tp=tp, fp=fp, fn=fn)


class Accuracy:
    """
    Four-way accuracy suite: token containment, fuzzy, LLM meaning, embedding.
    Kept identical to the uploaded class.
    """
    def __init__(self, llm, tokenizer, predicted_answer,
                 actual_answer, threshold=0.8, gpu_id=0):
        self.predicted_answer = predicted_answer
        self.actual_answer    = actual_answer
        self.threshold        = threshold
        self.llm              = llm
        self.tokenizer        = tokenizer
        self.dev              = _get_cuda(gpu_id)

    def fuzzy_containment_accuracy(self):
        pred   = str(self.predicted_answer or "").lower()
        actual = str(self.actual_answer    or "").lower()
        if actual in pred:
            return 1.0
        pred_words   = pred.split()
        actual_words = actual.split()
        actual_len   = len(actual_words)
        best_sim     = 0.0
        for i in range(max(1, len(pred_words) - actual_len + 1)):
            window   = ' '.join(pred_words[i:i + actual_len])
            best_sim = max(best_sim, SequenceMatcher(None, actual, window).ratio())
        return best_sim

    def token_containment_accuracy(self):
        pred_tok   = set(re.findall(r'\w+', str(self.predicted_answer or "").lower()))
        actual_tok = set(re.findall(r'\w+', str(self.actual_answer    or "").lower()))
        if not actual_tok:
            return 1.0 if not pred_tok else 0.0
        return len(actual_tok & pred_tok) / len(actual_tok)

    def meaning_based_accuracy(self):
        pred   = str(self.predicted_answer or "").lower()
        actual = str(self.actual_answer    or "").lower()
        prompt = (
            "You are an expert at evaluating the accuracy of the answer.\n\n"
            "Given two sets of answers, provide a score between 0 and 1 on how "
            "much the predicted answer matches with the actual answer.\n\n"
            "Instructions:\n"
            "1. The score should only be based on the meaning of the answers and "
            "not on the structure or format of the answers.\n"
            "2. The score should only be between 0 and 1 with 1 indicating the same "
            "meaning of both the answers and 0 indicating no relation.\n"
            f"Predicted answer: {pred}\nActual answer: {actual}\nScore:"
        )
        inp       = self.tokenizer(prompt, return_tensors="pt",
                                   truncation=True, max_length=2048)
        input_ids = inp.input_ids.to(self.dev)
        attn_mask = inp.attention_mask.to(self.dev)
        output    = self.llm.generate(
            input_ids, attention_mask=attn_mask,
            pad_token_id=self.tokenizer.eos_token_id, max_new_tokens=16
        )
        out = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        out = out.replace('"', '').replace("'", "")
        out = out.split("Score:")[-1].strip().split("\n")[0] \
              if "Score:" in out else out.strip().split("\n")[-1]
        try:
            return float(re.findall(r'\d+\.?\d*', out)[0])
        except Exception:
            return float('nan')

    def question_embedding(self, question):
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        return model.encode(question)

    def meaning_based_accuracy_controlled(self):
        model   = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        act_emb = model.encode([str(self.actual_answer    or "")]).reshape(1, -1)
        pre_emb = model.encode([str(self.predicted_answer or "")]).reshape(1, -1)
        return float(sk_cosine(act_emb, pre_emb)[0][0])

    def evaluate_rag_accuracy(self):
        return dict(
            Token_level_accuracy              = self.token_containment_accuracy(),
            Fuzzy_based_accuracy              = self.fuzzy_containment_accuracy(),
            Meaning_based_accuracy            = self.meaning_based_accuracy(),
            Meaning_based_accuracy_controlled = self.meaning_based_accuracy_controlled(),
        )


# =============================================================================
# 2.  STANDARD NLP METRICS  (GROUP A)
# =============================================================================

def _normalize(text: str) -> str:
    """SQuAD-style normalisation: lowercase, strip articles and punctuation."""
    if not isinstance(text, str):
        text = str(text) if text else ""
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def _tokens(text: str):
    return _normalize(text).split()


def exact_match(pred: str, gold: str) -> float:
    return float(_normalize(pred) == _normalize(gold))


def token_prf(pred: str, gold: str):
    """Token-level Precision, Recall, F1 — SQuAD / RAGAS convention."""
    pc = Counter(_tokens(pred))
    gc = Counter(_tokens(gold))
    common = sum((pc & gc).values())
    if common == 0:
        return 0.0, 0.0, 0.0
    p = common / sum(pc.values())
    r = common / sum(gc.values())
    f = 2 * p * r / (p + r)
    return p, r, f


def meteor_score(pred: str, gold: str) -> float:
    """Full METEOR via NLTK; falls back to harmonic-mean unigram F-measure."""
    try:
        import nltk
        from nltk.translate.meteor_score import single_meteor_score
        for res in ('wordnet', 'omw-1.4', 'punkt'):
            try:
                nltk.data.find(res)
            except LookupError:
                nltk.download(res, quiet=True)
        return float(single_meteor_score(_tokens(gold), _tokens(pred)))
    except Exception:
        p, r, _ = token_prf(pred, gold)
        return (10 * p * r) / (r + 9 * p) if (r + 9 * p) > 0 else 0.0


def bleu_score(pred: str, gold: str) -> float:
    """Sentence BLEU-4 with smoothing. Falls back to 1-gram BLEU."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        sf = SmoothingFunction().method1
        return float(sentence_bleu([_tokens(gold)], _tokens(pred),
                                   smoothing_function=sf))
    except Exception:
        pt = _tokens(pred)
        gt = _tokens(gold)
        if not pt:
            return 0.0
        bp      = min(1.0, len(pt) / max(len(gt), 1))
        matches = sum(1 for t in pt if t in gt)
        return bp * matches / len(pt)


def lcs_score(pred: str, gold: str) -> float:
    """ROUGE-L style: LCS_length / max(|pred|, |gold|). Used in IrCoT eval."""
    pt, gt = _tokens(pred), _tokens(gold)
    if not pt or not gt:
        return 0.0
    m, n = len(pt), len(gt)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            curr[j] = prev[j-1] + 1 if pt[i-1] == gt[j-1] \
                      else max(prev[j], curr[j-1])
        prev = curr
    return prev[n] / max(m, n)


def jaccard_similarity(pred: str, gold: str) -> float:
    ps, gs = set(_tokens(pred)), set(_tokens(gold))
    if not ps and not gs:
        return 1.0
    return len(ps & gs) / len(ps | gs)


def hamming_distance(pred: str, gold: str) -> int:
    """Character-level Hamming distance (strings padded to equal length)."""
    p, g   = _normalize(pred), _normalize(gold)
    maxlen = max(len(p), len(g))
    return sum(c1 != c2 for c1, c2 in zip(p.ljust(maxlen), g.ljust(maxlen)))


def bertscore_batch(preds, golds, device: str = "cuda:0", lang: str = "en"):
    """BERTScore P/R/F1 over full batch on GPU."""
    try:
        from bert_score import score as _bs
        P, R, F = _bs(preds, golds, lang=lang, device=device, verbose=False)
        return P.tolist(), R.tolist(), F.tolist()
    except ImportError:
        print("[WARN] bert_score not installed → pip install bert-score")
        nan = [float('nan')] * len(preds)
        return nan, nan, nan


def auc_roc_batch(preds, golds):
    """Batch ROC-AUC. Label = EM, Score = token-F1. Needs ≥1 pos + ≥1 neg."""
    try:
        from sklearn.metrics import roc_auc_score
        labels = [exact_match(p, g) for p, g in zip(preds, golds)]
        scores = [token_prf(p, g)[2]  for p, g in zip(preds, golds)]
        if len(set(labels)) < 2:
            return float('nan')
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float('nan')


# =============================================================================
# 3.  LLM-BASED ANSWER P / R / F1  (GROUP B)
#     Three independent methods — NLI, G-Eval, Embedding
# =============================================================================

class LLMAnswerPRF:
    """
    Three complementary methods to compute Precision, Recall, F1
    purely from (gold_answer, model_response) without any retrieval context.

    ┌─────────────┬────────────────────────────────────────────────────────────┐
    │ Method      │ What it measures                                            │
    ├─────────────┼────────────────────────────────────────────────────────────┤
    │ NLI         │ Binary entailment verdicts (LLM-as-NLI-classifier)          │
    │             │  Precision: Response → answer  (is Response a subset?)      │
    │             │  Recall   : answer → Response  (is answer covered?)         │
    ├─────────────┼────────────────────────────────────────────────────────────┤
    │ G-Eval      │ Continuous 0–1 CoT scores (G-Eval / Prometheus style)       │
    │             │  Precision: "what fraction of Response matches the answer?" │
    │             │  Recall   : "what fraction of answer is in Response?"       │
    ├─────────────┼────────────────────────────────────────────────────────────┤
    │ Embedding   │ Sentence-level greedy cosine matching (BERTScore-style)     │
    │             │  Uses all-MiniLM-L6-v2, no LLM call needed                 │
    └─────────────┴────────────────────────────────────────────────────────────┘
    """

    def __init__(self, llm, tokenizer, gpu_id: int = 0):
        self.llm       = llm
        self.tokenizer = tokenizer
        self.dev       = _get_cuda(gpu_id)
        self._st       = None

    @property
    def st(self):
        if self._st is None:
            self._st = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        return self._st

    def _generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        inp       = self.tokenizer(prompt, return_tensors="pt",
                                   truncation=True, max_length=2048)
        input_ids = inp.input_ids.to(self.dev)
        attn_mask = inp.attention_mask.to(self.dev)
        out_ids   = self.llm.generate(
            input_ids,
            attention_mask=attn_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
        )
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

    # ── 3.1  NLI-based ───────────────────────────────────────────────────────
    def nli_prf(self, gold: str, response: str):
        """
        Precision: does the model Response entail the gold answer?
                   i.e., is everything in Response correct w.r.t. answer?
        Recall:    does the gold answer entail the model Response?
                   i.e., does Response cover what the answer says?
        Both are binary (0 or 1) from a single LLM verdict.
        F1 = harmonic mean.
        """
        # Precision — Response as hypothesis, answer as premise
        p_prompt = (
            "You are an expert at Natural Language Inference.\n\n"
            "Premise: {answer}\n"
            "Hypothesis: {response}\n\n"
            "Does the Premise ENTAIL the Hypothesis? "
            "Answer ONLY 'Yes' or 'No'.\nVerdict:"
        ).format(answer=gold, response=response)

        # Recall — answer as hypothesis, Response as premise
        r_prompt = (
            "You are an expert at Natural Language Inference.\n\n"
            "Premise: {response}\n"
            "Hypothesis: {answer}\n\n"
            "Does the Premise ENTAIL the Hypothesis? "
            "Answer ONLY 'Yes' or 'No'.\nVerdict:"
        ).format(answer=gold, response=response)

        p_out  = self._generate(p_prompt, max_new_tokens=8).lower()
        r_out  = self._generate(r_prompt, max_new_tokens=8).lower()
        prec   = 1.0 if "yes" in p_out else 0.0
        rec    = 1.0 if "yes" in r_out else 0.0
        f1     = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    # ── 3.2  G-Eval ──────────────────────────────────────────────────────────
    def geval_prf(self, gold: str, response: str):
        """
        G-Eval style (Wei et al., 2023; Prometheus eval):
        Ask the LLM to reason step-by-step and provide a continuous 0–1 score.

        Precision score: "What fraction of the Response content is correct /
                          consistent with the gold Answer?"
        Recall score:    "What fraction of the gold Answer content is covered
                          by the Response?"
        """
        p_prompt = (
            "You are an expert evaluator.\n\n"
            "Gold Answer : {answer}\n"
            "Model Response : {response}\n\n"
            "Task: Evaluate PRECISION — the fraction of the Model Response that "
            "is factually correct and consistent with the Gold Answer.\n"
            "Think step by step, then provide a final score between 0.0 (nothing "
            "correct) and 1.0 (fully correct).\n"
            "Respond in this exact format:\n"
            "Reasoning: <your reasoning>\n"
            "Score: <number between 0.0 and 1.0>"
        ).format(answer=gold, response=response)

        r_prompt = (
            "You are an expert evaluator.\n\n"
            "Gold Answer : {answer}\n"
            "Model Response : {response}\n\n"
            "Task: Evaluate RECALL — the fraction of the Gold Answer content "
            "that is covered or addressed by the Model Response.\n"
            "Think step by step, then provide a final score between 0.0 (nothing "
            "covered) and 1.0 (fully covered).\n"
            "Respond in this exact format:\n"
            "Reasoning: <your reasoning>\n"
            "Score: <number between 0.0 and 1.0>"
        ).format(answer=gold, response=response)

        def _parse_score(raw: str) -> float:
            if "Score:" in raw:
                snippet = raw.split("Score:")[-1].strip().split("\n")[0]
            else:
                snippet = raw.strip().split("\n")[-1]
            nums = re.findall(r'\d+\.?\d*', snippet)
            if nums:
                val = float(nums[0])
                return min(max(val, 0.0), 1.0)   # clamp to [0,1]
            return float('nan')

        p_raw  = self._generate(p_prompt, max_new_tokens=200)
        r_raw  = self._generate(r_prompt, max_new_tokens=200)
        prec   = _parse_score(p_raw)
        rec    = _parse_score(r_raw)
        if not (np.isnan(prec) or np.isnan(rec)) and (prec + rec) > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = float('nan')
        return prec, rec, f1

    # ── 3.3  Embedding P / R / F1  (BERTScore-style, local embeddings) ───────
    def embedding_prf(self, gold: str, response: str):
        """
        Sentence-level greedy cosine matching using all-MiniLM-L6-v2.
        Same algorithm as BERTScore but with lighter embeddings:

        Precision: for each response sentence, find max cosine-sim to any
                   gold sentence → average these maxima.
        Recall:    for each gold sentence, find max cosine-sim to any
                   response sentence → average these maxima.
        F1:        harmonic mean.

        No LLM call needed — fully embedding-based, runs on GPU via
        sentence-transformers.
        """
        def _sent_split(text: str):
            sents = re.split(r'(?<=[.!?])\s+', text.strip())
            return [s.strip() for s in sents if len(s.strip()) > 3] or [text]

        gold_sents = _sent_split(gold)
        resp_sents = _sent_split(response)

        gold_emb = self.st.encode(gold_sents, convert_to_numpy=True,
                                  show_progress_bar=False)
        resp_emb = self.st.encode(resp_sents, convert_to_numpy=True,
                                  show_progress_bar=False)

        # cosine similarity matrix  [resp × gold]
        sim_matrix = sk_cosine(resp_emb, gold_emb)   # shape (R, G)

        # Precision: each response sentence matched to its best gold sentence
        prec = float(sim_matrix.max(axis=1).mean())

        # Recall: each gold sentence matched to its best response sentence
        rec  = float(sim_matrix.max(axis=0).mean())

        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        return prec, rec, f1


# =============================================================================
# 4.  RETRIEVAL P / R / F1  (GROUP C — text-only, no doc IDs)
# =============================================================================

class TextRetrievalMetrics:
    """
    Computes Retrieval Precision, Recall, F1 purely from text when
    document IDs are not available.

    Strategy
    ────────
    1. Split `context`            into sentences  → "retrieved set"
    2. Split `supporting_context` into sentences  → "relevant set"
    3. For each retrieved sentence find the maximum cosine-similarity
       to any relevant sentence.
       A retrieved sentence is a True Positive if max-sim ≥ threshold.
    4. Precision = |TP_retrieved| / |retrieved|
       Recall    = |TP_relevant|  / |relevant|      (symmetric pass)
       F1        = harmonic mean

    The threshold default (0.70) follows the convention used in the
    RAGAS retrieval evaluation and Adaptive-RAG context quality checks.
    """

    def __init__(self, st_model: SentenceTransformer,
                 sim_threshold: float = 0.70):
        self.st        = st_model
        self.threshold = sim_threshold

    @staticmethod
    def _split_sentences(text: str):
        sents = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sents if len(s.strip()) > 5]

    def compute(self, context: str, supporting_context: str):
        """
        Returns dict with keys:
          Retrieval_Precision, Retrieval_Recall, Retrieval_F1,
          Retrieval_TP, Retrieval_FP, Retrieval_FN
        Returns NaN values when either text is empty.
        """
        nan_result = dict(
            Retrieval_Precision=float('nan'),
            Retrieval_Recall=float('nan'),
            Retrieval_F1=float('nan'),
            Retrieval_TP=float('nan'),
            Retrieval_FP=float('nan'),
            Retrieval_FN=float('nan'),
        )

        retrieved = self._split_sentences(context)
        relevant  = self._split_sentences(supporting_context)

        if not retrieved or not relevant:
            return nan_result

        ret_emb = self.st.encode(retrieved, convert_to_numpy=True,
                                 show_progress_bar=False)
        rel_emb = self.st.encode(relevant,  convert_to_numpy=True,
                                 show_progress_bar=False)

        sim_matrix = sk_cosine(ret_emb, rel_emb)   # shape (R_retrieved, R_relevant)

        # ── Precision pass: are retrieved sentences covered by relevant? ──────
        ret_tp_flags = (sim_matrix.max(axis=1) >= self.threshold).astype(int)
        tp_ret  = int(ret_tp_flags.sum())
        fp      = len(retrieved) - tp_ret
        prec    = tp_ret / len(retrieved)

        # ── Recall pass: are relevant sentences covered by retrieved? ─────────
        rel_tp_flags = (sim_matrix.max(axis=0) >= self.threshold).astype(int)
        tp_rel  = int(rel_tp_flags.sum())
        fn      = len(relevant) - tp_rel
        rec     = tp_rel / len(relevant)

        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        return dict(
            Retrieval_Precision = round(prec, 6),
            Retrieval_Recall    = round(rec,  6),
            Retrieval_F1        = round(f1,   6),
            Retrieval_TP        = tp_ret,
            Retrieval_FP        = fp,
            Retrieval_FN        = fn,
        )


# =============================================================================
# 5.  RAGAS-STYLE RAG METRICS  (GROUP D)
# =============================================================================

class RAGASMetrics:
    """
    RAGAS-compatible implementations using your local LLM + tokenizer.

    References
    ──────────
    RAGAS        — https://arxiv.org/abs/2309.15217
    Adaptive-RAG — https://arxiv.org/abs/2403.14403
    IrCoT        — https://arxiv.org/abs/2212.10509
    GraphRAG     — https://arxiv.org/abs/2404.16130
    """

    def __init__(self, llm, tokenizer, gpu_id: int = 0):
        self.llm        = llm
        self.tokenizer  = tokenizer
        self.dev        = _get_cuda(gpu_id)
        self._st_model  = None

    @property
    def st(self):
        if self._st_model is None:
            self._st_model = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2'
            )
        return self._st_model

    def _embed(self, texts) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self.st.encode(texts, convert_to_numpy=True)

    def _generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        inp       = self.tokenizer(prompt, return_tensors="pt",
                                   truncation=True, max_length=2048)
        input_ids = inp.input_ids.to(self.dev)
        attn_mask = inp.attention_mask.to(self.dev)
        out_ids   = self.llm.generate(
            input_ids,
            attention_mask=attn_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
        )
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

    # Faithfulness — delegates to existing class
    def faithfulness(self, question, response, context):
        f = Faithfulness(question, response, self.llm,
                         self.tokenizer, context, gpu_id=0)
        return f.faithfulness()

    # Answer Relevance — N reverse-questions averaged (RAGAS §3.4)
    def answer_relevance(self, question, response, n_reverse: int = 3):
        q_emb = self._embed(question).reshape(1, -1)
        sims  = []
        for _ in range(n_reverse):
            prompt = (
                "You are an expert at creating questions based on answers.\n"
                "Generate exactly ONE question for the given answer. "
                "Do not add extra information.\n"
                f"answer: {response}\nQuestion:"
            )
            out   = self._generate(prompt, max_new_tokens=128)
            gen_q = out.split("Question:")[-1].strip() if "Question:" in out \
                    else out.strip().split("\n")[-1]
            gen_q = gen_q.replace('"', '').replace("'", "")
            ge    = self._embed(gen_q).reshape(1, -1)
            sims.append(float(sk_cosine(q_emb, ge)[0][0]))
        return float(np.mean(sims))

    # Context Precision — MAP (RAGAS §3.2 / IrCoT)
    def context_precision(self, question, context, answer=""):
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', context)
                     if len(s.strip()) > 10]
        if not sentences:
            return float(sk_cosine(
                self._embed(question).reshape(1, -1),
                self._embed(context).reshape(1, -1)
            )[0][0])

        if self.llm is None:
            q_emb     = self._embed(question).reshape(1, -1)
            rel_flags = [
                1 if float(sk_cosine(q_emb, self._embed(s).reshape(1, -1))[0][0]) > 0.4
                else 0
                for s in sentences
            ]
        else:
            rel_flags = []
            for sent in sentences:
                prompt = (
                    "Is the following context sentence useful for answering the question? "
                    "Answer ONLY 'Yes' or 'No'.\n"
                    f"Question: {question}\nContext sentence: {sent}\nAnswer:"
                )
                out = self._generate(prompt, max_new_tokens=8).lower()
                rel_flags.append(1 if "yes" in out else 0)

        total_rel = sum(rel_flags)
        if total_rel == 0:
            return 0.0
        precisions_at_k, running_rel = [], 0
        for k, flag in enumerate(rel_flags, start=1):
            if flag:
                running_rel += 1
                precisions_at_k.append(running_rel / k)
        return float(np.mean(precisions_at_k)) if precisions_at_k else 0.0

    # Context Recall — sentence attribution (RAGAS §3.3 / Adaptive-RAG)
    def context_recall(self, gold_answer, context):
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', gold_answer)
                     if len(s.strip()) > 5]
        if not sentences:
            return float('nan')

        if self.llm is None:
            ctx_emb    = self._embed(context).reshape(1, -1)
            attributed = sum(
                1 for s in sentences
                if float(sk_cosine(self._embed(s).reshape(1, -1), ctx_emb)[0][0]) > 0.4
            )
        else:
            attributed = 0
            for sent in sentences:
                prompt = (
                    "Can the following sentence be inferred or attributed to the given "
                    "context? Answer ONLY 'Yes' or 'No'.\n"
                    f"Context: {context}\nSentence: {sent}\nAnswer:"
                )
                out = self._generate(prompt, max_new_tokens=8).lower()
                if "yes" in out:
                    attributed += 1
        return attributed / len(sentences)

    # Error Rate
    @staticmethod
    def error_rate(pred_list, gold_list, threshold: float = 0.5) -> float:
        if not pred_list:
            return float('nan')
        errors = sum(1 for p, g in zip(pred_list, gold_list)
                     if token_prf(p, g)[2] < threshold)
        return errors / len(pred_list)

    # Error Detection Rate
    @staticmethod
    def error_detection_rate(detected, actual_errors):
        tp = sum(d == 1 and a == 1 for d, a in zip(detected, actual_errors))
        fn = sum(d == 0 and a == 1 for d, a in zip(detected, actual_errors))
        return tp / (tp + fn) if (tp + fn) > 0 else float('nan')

    # Error Rejection Rate
    @staticmethod
    def error_rejection_rate(rejected, actual_errors):
        tp           = sum(r == 1 and a == 1 for r, a in zip(rejected, actual_errors))
        total_errors = sum(actual_errors)
        return tp / total_errors if total_errors > 0 else float('nan')


# =============================================================================
# 6.  MODEL LOADER
# =============================================================================

def load_model(model_name: str, gpu_id: int):
    dev = _get_cuda(gpu_id)
    print(f"[INFO] Loading tokenizer : {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[INFO] Loading model     : {model_name}  →  {dev}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map  = "auto",
    )
    model.eval()
    return model, tokenizer


# =============================================================================
# 7.  MAIN EVALUATION PIPELINE
# =============================================================================

def run_evaluation(
    df:               pd.DataFrame,
    model_name:       str   = "mistralai/Mistral-7B-Instruct-v0.2",
    output_path:      str   = "evaluation_results.csv",
    gpu_id:           int   = 0,
    error_threshold:  float = 0.5,
    ret_threshold:    float = 0.70,    # cosine threshold for retrieval P/R/F1
    run_llm_metrics:  bool  = True,
    # ── column names ──────────────────────────────────────────────────────────
    col_answer:       str   = "answer",
    col_response:     str   = "Response",
    col_context:      str   = "context",
    col_sup_ctx:      str   = "supporting_context",
    col_question:     str   = "Analysis",
    col_aspect:       str   = "Aspect",
) -> pd.DataFrame:

    dev = _get_cuda(gpu_id)
    print(f"\n{'='*70}")
    print(f"  RAG Evaluation  |  rows={len(df)}  |  device={dev}")
    print(f"{'='*70}\n")

    for col in [col_answer, col_response]:
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' not found. "
                f"Available: {list(df.columns)}"
            )

    # ── load models ───────────────────────────────────────────────────────────
    llm, tokenizer = None, None
    if run_llm_metrics:
        llm, tokenizer = load_model(model_name, gpu_id)

    ragas    = RAGASMetrics(llm, tokenizer, gpu_id=gpu_id)
    llm_prf  = LLMAnswerPRF(llm, tokenizer, gpu_id=gpu_id) if run_llm_metrics else None
    ret_eval = TextRetrievalMetrics(ragas.st, sim_threshold=ret_threshold)

    results   = df.copy()
    pred_list = [str(r or "") for r in df[col_response].tolist()]
    gold_list = [str(r or "") for r in df[col_answer].tolist()]

    # ── BERTScore batch (GPU) ─────────────────────────────────────────────────
    print("[INFO] Computing BERTScore (GPU batch) …")
    bert_p, bert_r, bert_f = bertscore_batch(pred_list, gold_list, device=dev)

    # ── AUC-ROC (batch) ───────────────────────────────────────────────────────
    auc_val = auc_roc_batch(pred_list, gold_list)
    print(f"[INFO] Batch AUC-ROC: {auc_val:.4f}")

    # ── Row-level containers ──────────────────────────────────────────────────
    # Group A
    em_l, f1_l, pr_l, rc_l         = [], [], [], []
    met_l, bleu_l, lcs_l           = [], [], []
    jac_l, ham_l, err_l            = [], [], []
    # Group B — NLI
    nli_p_l, nli_r_l, nli_f_l      = [], [], []
    # Group B — G-Eval
    gev_p_l, gev_r_l, gev_f_l      = [], [], []
    # Group B — Embedding PRF
    emb_p_l, emb_r_l, emb_f_l      = [], [], []
    # Group C — Retrieval
    ret_p_l, ret_r_l, ret_f_l      = [], [], []
    ret_tp_l, ret_fp_l, ret_fn_l   = [], [], []
    # Group D — RAGAS
    faith_l, ctx_p_l, ctx_r_l, ar_l = [], [], [], []
    # Group E — Accuracy
    tok_acc_l, fuz_acc_l            = [], []
    mng_llm_l, mng_emb_l           = [], []

    print("[INFO] Computing row-level metrics …")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        pred = str(row.get(col_response, "") or "")
        gold = str(row.get(col_answer,   "") or "")
        q    = str(row.get(col_question, "") or "")
        ctx  = str(row.get(col_context,  "") or "")
        sctx = str(row.get(col_sup_ctx,  "") or "")

        # ── GROUP A: Standard NLP ─────────────────────────────────────────────
        em        = exact_match(pred, gold)
        p, r, f1  = token_prf(pred, gold)
        met       = meteor_score(pred, gold)
        bleu      = bleu_score(pred, gold)
        lcs       = lcs_score(pred, gold)
        jac       = jaccard_similarity(pred, gold)
        ham       = hamming_distance(pred, gold)
        is_err    = int(f1 < error_threshold)

        em_l.append(em);   f1_l.append(f1)
        pr_l.append(p);    rc_l.append(r)
        met_l.append(met); bleu_l.append(bleu)
        lcs_l.append(lcs); jac_l.append(jac)
        ham_l.append(ham); err_l.append(is_err)

        # ── GROUP B: LLM Answer P/R/F1 ───────────────────────────────────────

        # B.3 — Embedding PRF (no LLM needed, always computed)
        _ep = LLMAnswerPRF(None, None, gpu_id=gpu_id)
        _ep._st = ragas.st           # reuse already-loaded SentenceTransformer
        ep, er, ef = _ep.embedding_prf(gold, pred)
        emb_p_l.append(ep); emb_r_l.append(er); emb_f_l.append(ef)

        if run_llm_metrics and llm_prf is not None:
            # B.1 — NLI
            np_, nr, nf = llm_prf.nli_prf(gold, pred)
            nli_p_l.append(np_); nli_r_l.append(nr); nli_f_l.append(nf)
            # B.2 — G-Eval
            gp, gr, gf  = llm_prf.geval_prf(gold, pred)
            gev_p_l.append(gp); gev_r_l.append(gr); gev_f_l.append(gf)
        else:
            nli_p_l.append(float('nan')); nli_r_l.append(float('nan'))
            nli_f_l.append(float('nan'))
            gev_p_l.append(float('nan')); gev_r_l.append(float('nan'))
            gev_f_l.append(float('nan'))

        # ── GROUP C: Retrieval P/R/F1  (text-based) ───────────────────────────
        if ctx.strip() and sctx.strip():
            ret_res = ret_eval.compute(ctx, sctx)
        else:
            ret_res = dict(
                Retrieval_Precision=float('nan'), Retrieval_Recall=float('nan'),
                Retrieval_F1=float('nan'),        Retrieval_TP=float('nan'),
                Retrieval_FP=float('nan'),        Retrieval_FN=float('nan'),
            )
        ret_p_l.append(ret_res["Retrieval_Precision"])
        ret_r_l.append(ret_res["Retrieval_Recall"])
        ret_f_l.append(ret_res["Retrieval_F1"])
        ret_tp_l.append(ret_res["Retrieval_TP"])
        ret_fp_l.append(ret_res["Retrieval_FP"])
        ret_fn_l.append(ret_res["Retrieval_FN"])

        # ── GROUP D: RAGAS metrics ────────────────────────────────────────────
        # Faithfulness
        if ctx and q and run_llm_metrics:
            faith = ragas.faithfulness(q, pred, ctx)
        else:
            faith = float('nan')

        # Context Precision
        ctx_p = ragas.context_precision(q, ctx, gold) if (ctx and q) \
                else float('nan')

        # Context Recall (prefer supporting_context)
        recall_ctx = sctx.strip() if sctx.strip() else ctx
        ctx_r = ragas.context_recall(gold, recall_ctx) if (recall_ctx and gold) \
                else float('nan')

        # Answer Relevance
        if q and run_llm_metrics:
            ar_score = ragas.answer_relevance(q, pred, n_reverse=3)
        elif q:
            ar_score = float(sk_cosine(
                ragas._embed(q).reshape(1, -1),
                ragas._embed(pred).reshape(1, -1)
            )[0][0])
        else:
            ar_score = float('nan')

        faith_l.append(faith); ctx_p_l.append(ctx_p)
        ctx_r_l.append(ctx_r); ar_l.append(ar_score)

        # ── GROUP E: Accuracy class ───────────────────────────────────────────
        acc = Accuracy(llm, tokenizer, pred, gold, gpu_id=gpu_id)
        if run_llm_metrics:
            ar = acc.evaluate_rag_accuracy()
            tok_acc_l.append(ar["Token_level_accuracy"])
            fuz_acc_l.append(ar["Fuzzy_based_accuracy"])
            mng_llm_l.append(ar["Meaning_based_accuracy"])
            mng_emb_l.append(ar["Meaning_based_accuracy_controlled"])
        else:
            tok_acc_l.append(acc.token_containment_accuracy())
            fuz_acc_l.append(acc.fuzzy_containment_accuracy())
            mng_llm_l.append(float('nan'))
            mng_emb_l.append(acc.meaning_based_accuracy_controlled())

    # ── Attach all columns ────────────────────────────────────────────────────
    # Group A
    results["EM"]                    = em_l
    results["Token_F1"]              = f1_l
    results["Token_Precision"]       = pr_l
    results["Token_Recall"]          = rc_l
    results["METEOR"]                = met_l
    results["BLEU"]                  = bleu_l
    results["BERTScore_P"]           = bert_p
    results["BERTScore_R"]           = bert_r
    results["BERTScore_F1"]          = bert_f
    results["LCS"]                   = lcs_l
    results["Jaccard_Similarity"]    = jac_l
    results["Hamming_Distance"]      = ham_l
    # Group B — LLM Answer PRF
    results["NLI_Precision"]         = nli_p_l
    results["NLI_Recall"]            = nli_r_l
    results["NLI_F1"]                = nli_f_l
    results["GEval_Precision"]       = gev_p_l
    results["GEval_Recall"]          = gev_r_l
    results["GEval_F1"]              = gev_f_l
    results["Emb_Precision"]         = emb_p_l
    results["Emb_Recall"]            = emb_r_l
    results["Emb_F1"]                = emb_f_l
    # Group C — Retrieval
    results["Retrieval_Precision"]   = ret_p_l
    results["Retrieval_Recall"]      = ret_r_l
    results["Retrieval_F1"]          = ret_f_l
    results["Retrieval_TP"]          = ret_tp_l
    results["Retrieval_FP"]          = ret_fp_l
    results["Retrieval_FN"]          = ret_fn_l
    # Group D — RAGAS
    results["Faithfulness"]          = faith_l
    results["Context_Precision"]     = ctx_p_l
    results["Context_Recall"]        = ctx_r_l
    results["Answer_Relevance"]      = ar_l
    # Group E — Accuracy
    results["Token_Accuracy"]        = tok_acc_l
    results["Fuzzy_Accuracy"]        = fuz_acc_l
    results["Meaning_Accuracy_LLM"]  = mng_llm_l
    results["Meaning_Accuracy_Emb"]  = mng_emb_l
    # Group F — Error
    results["Is_Error"]              = err_l

    # ── Global / batch error metrics ──────────────────────────────────────────
    actual_errors   = err_l
    errors_detected = err_l
    errors_rejected = err_l
    global_error_rate = RAGASMetrics.error_rate(pred_list, gold_list, error_threshold)
    edr               = RAGASMetrics.error_detection_rate(errors_detected, actual_errors)
    err_rej           = RAGASMetrics.error_rejection_rate(errors_rejected, actual_errors)

    # ── Summary ───────────────────────────────────────────────────────────────
    numeric_cols = [
        # A
        "EM", "Token_F1", "Token_Precision", "Token_Recall",
        "METEOR", "BLEU", "BERTScore_P", "BERTScore_R", "BERTScore_F1",
        "LCS", "Jaccard_Similarity", "Hamming_Distance",
        # B
        "NLI_Precision", "NLI_Recall", "NLI_F1",
        "GEval_Precision", "GEval_Recall", "GEval_F1",
        "Emb_Precision", "Emb_Recall", "Emb_F1",
        # C
        "Retrieval_Precision", "Retrieval_Recall", "Retrieval_F1",
        # D
        "Faithfulness", "Context_Precision", "Context_Recall", "Answer_Relevance",
        # E
        "Token_Accuracy", "Fuzzy_Accuracy",
        "Meaning_Accuracy_LLM", "Meaning_Accuracy_Emb",
    ]
    summary = {col: float(pd.Series(results[col]).mean())
               for col in numeric_cols if col in results.columns}
    summary["AUC_ROC"]              = auc_val
    summary["Global_Error_Rate"]    = global_error_rate
    summary["Error_Detection_Rate"] = edr
    summary["Error_Rejection_Rate"] = err_rej

    # ── Pretty-print summary ──────────────────────────────────────────────────
    # Group headers for readability
    groups = {
        "── GROUP A: Standard NLP ──────────────────────────────────────": [
            "EM","Token_F1","Token_Precision","Token_Recall","METEOR","BLEU",
            "BERTScore_P","BERTScore_R","BERTScore_F1","LCS",
            "Jaccard_Similarity","Hamming_Distance","AUC_ROC",
        ],
        "── GROUP B: LLM Answer P/R/F1 ─────────────────────────────────": [
            "NLI_Precision","NLI_Recall","NLI_F1",
            "GEval_Precision","GEval_Recall","GEval_F1",
            "Emb_Precision","Emb_Recall","Emb_F1",
        ],
        "── GROUP C: Retrieval P/R/F1 (text) ───────────────────────────": [
            "Retrieval_Precision","Retrieval_Recall","Retrieval_F1",
        ],
        "── GROUP D: RAG / RAGAS ────────────────────────────────────────": [
            "Faithfulness","Context_Precision","Context_Recall","Answer_Relevance",
        ],
        "── GROUP E: Accuracy ───────────────────────────────────────────": [
            "Token_Accuracy","Fuzzy_Accuracy",
            "Meaning_Accuracy_LLM","Meaning_Accuracy_Emb",
        ],
        "── GROUP F: Error Metrics ──────────────────────────────────────": [
            "Global_Error_Rate","Error_Detection_Rate","Error_Rejection_Rate",
        ],
    }
    print(f"\n{'='*70}")
    print("  EVALUATION SUMMARY")
    print(f"{'='*70}")
    for header, keys in groups.items():
        print(f"\n  {header}")
        for k in keys:
            v = summary.get(k, float('nan'))
            if isinstance(v, float) and not np.isnan(v):
                if 0.0 <= v <= 1.0:
                    filled = int(v * 24)
                    bar    = "  [" + "█"*filled + "░"*(24-filled) + f"]  {v:.4f}"
                else:
                    bar = f"  {v:.2f}"
            else:
                bar = f"  {v}"
            print(f"    {k:<32}{bar}")
    print(f"\n{'='*70}\n")

    # ── Save ─────────────────────────────────────────────────────────────────
    results.to_csv(output_path, index=False)
    print(f"[INFO] Full results saved  →  {output_path}")
    summary_path = output_path.replace(".csv", "_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"[INFO] Summary saved       →  {summary_path}")

    return results


# =============================================================================
# 8.  CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Comprehensive RAG Evaluation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",          required=True)
    p.add_argument("--model",         default="mistralai/Mistral-7B-Instruct-v0.2")
    p.add_argument("--output",        default="evaluation_results.csv")
    p.add_argument("--gpu",           type=int,   default=0)
    p.add_argument("--threshold",     type=float, default=0.5,
                   help="Token-F1 threshold below which a response is an error")
    p.add_argument("--ret-threshold", type=float, default=0.70,
                   help="Cosine-sim threshold for text-based retrieval P/R/F1")
    p.add_argument("--no-llm",        action="store_true",
                   help="Skip LLM calls; use embeddings only")
    p.add_argument("--col-answer",    default="answer")
    p.add_argument("--col-response",  default="Response")
    p.add_argument("--col-context",   default="context")
    p.add_argument("--col-supctx",    default="supporting_context")
    p.add_argument("--col-question",  default="Analysis")
    p.add_argument("--col-aspect",    default="Aspect")
    os.environ["CUDA_VISIBLE_DEVICES"] = "3,1"
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    

    if args.data.endswith(".csv"):
        df = pd.read_csv(args.data)
    elif args.data.endswith((".xlsx", ".xls")):
        df = pd.read_excel(args.data)
    else:
        raise ValueError("Supported formats: .csv, .xlsx, .xls")

    print(f"[INFO] Loaded {len(df)} rows  |  columns: {list(df.columns)}")

    run_evaluation(
        df              = df,
        model_name      = args.model,
        output_path     = args.output,
        gpu_id          = args.gpu,
        error_threshold = args.threshold,
        ret_threshold   = args.ret_threshold,
        run_llm_metrics = not args.no_llm,
        col_answer      = args.col_answer,
        col_response    = args.col_response,
        col_context     = args.col_context,
        col_sup_ctx     = args.col_supctx,
        col_question    = args.col_question,
        col_aspect      = args.col_aspect,
    )


# =============================================================================
# NOTES
# =============================================================================
#
# Metric groups at a glance
# ─────────────────────────
#  GROUP A  Token_F1 / Token_Precision / Token_Recall
#           → SQuAD-style token overlap between `answer` and `Response`
#             (ground-truth vs model output, NOT retrieval)
#
#  GROUP B  NLI / G-Eval / Embedding  P / R / F1
#           → Three independent LLM / embedding methods measuring
#             answer ↔ Response quality purely from text
#           NLI     : binary entailment verdict (0 or 1 per direction)
#           G-Eval  : continuous 0–1 CoT score (chain-of-thought reasoning)
#           Emb PRF : sentence-level greedy cosine matching (no LLM needed)
#
#  GROUP C  Retrieval_Precision / Retrieval_Recall / Retrieval_F1
#           → Sentence-level cosine matching between `context` (retrieved)
#             and `supporting_context` (relevant ground truth)
#           Threshold default = 0.70 (adjust with --ret-threshold)
#           Only computed for rows where BOTH columns are non-empty.
#
#  GROUP D  Faithfulness / Context_Precision / Context_Recall / Answer_Relevance
#           → Standard RAGAS metrics (LLM-graded)
#
# Error Detection / Rejection Rates
# ───────────────────────────────────
# Without a separate binary ground-truth error column, EDR and ERR are
# self-referential (both = 1.0).  To use real labels, e.g.:
#   actual_errors = df["is_hallucination"].astype(int).tolist()
# and pass to RAGASMetrics.error_detection_rate() directly.
# =============================================================================