# Hinglish Offensive Language Classifier

Comparing basic TF-IDF models with a fine-tuned multilingual transformer (XLM-RoBERTa) to spot offensive content in code-mixed Hindi-English (Hinglish) text.

## Problem

Most of the tools for detecting sentiment or toxicity are built for English, and tested on English. But South Asian social media—think group chats and comment threads—is full of sentences where Hindi and English mix freely, usually all written out in Roman script. Research hasn’t really caught up to this reality. Here, I looked at three different ways to classify Hinglish sentences as offensive or not: two TF-IDF variations and one transformer-based model.

## Dataset

- **Size:** Started out with 1,366 Hinglish sentences. Removed 30 duplicates, so 1,336 examples left.
- **Columns:** Just two: `text` (the Hinglish sentence in Roman script) and `label` (`offensive` or `not offensive`).
- **Class balance:** Almost even—695 offensive, 671 not offensive before deduplication. So accuracy actually means something; no rebalancing needed.
- **Text length:** Mostly short. On average, just 33 characters per sample, maxing out at 81 characters—think more like tweets or quick comments than paragraphs.

### What stands out in the data

A close look showed two things shaping the whole project:

1. **Intentional obfuscation.** People use leetspeak—swapping in numbers and symbols for letters—in offensive words. This helps them sneak past keyword-based filters. Because of this, I wanted to compare basic word-level vs. character-level features, since the latter are more flexible to these tricks.
2. **Synthetic patterns.** A ton of offensive examples look copy-pasted, or at least follow a template.This is not how people really swear online, and probably means the data was generated or heavily templated. On the non-offensive side, most texts are lighthearted roasting between friends, rather than general, topic-neutral content.

**Upshot:** High test set accuracy on this data doesn’t mean a model is genuinely robust. To check, I stress-tested the models on some custom sentences that didn’t fit these templates—see below.

## Methods

### Preprocessing
Lowercased everything; stripped punctuation except `@`, `!`, and `*`, since those often carry meaning in obfuscated words. Turned labels into binary (`not offensive`: 0, `offensive`: 1). Then split into train and test (80/20, stratified).

### Model 1: Word-level TF-IDF + Logistic Regression
Classic bag-of-words baseline. Used unigrams and bigrams, ignoring terms that show up only once.

### Model 2: Character-level TF-IDF + Logistic Regression
Broke text into character n-grams (3–5 characters, aware of word boundaries). The idea: even if you obfuscate "lode" as "l0d3," they still share some character chunks.

### Model 3: Fine-tuned XLM-RoBERTa (xlm-roberta-base)
Trained for 4 epochs, sequence length capped at 64 (more than enough for these short sentences), learning rate set to 2e-5, batch size 16, using HuggingFace's Trainer.

## Results

### Test set performance (in-distribution)

| Model | Accuracy | F1 Score |
|---|---|---|
| Word TF-IDF | 0.9800 | 0.9800 |
| Char TF-IDF | 0.9900 | 0.9900 |
| XLM-RoBERTa | 0.9925 | 0.9926 |

All models did great on the test set, with character-level TF-IDF beating word-level features, and XLM-RoBERTa edging out both. This matches the intuition: char and subword features can handle obfuscation, while plain words can’t.

Interestingly, the only mistakes XLM-RoBERTa made were marking a couple of benign slang phrases as offensive. For example, "mast joke maara" or "bakchodi mat kar"—probably because those words sometimes appear in worse contexts in the training data.

### Out-of-distribution stress test: hand-written Hinglish

Since the data looked synthetic, I wrote nine new Hinglish sentences that didn’t fit the old templates and ran all three models on them:

| Sentence | True label | Word TF-IDF | Char TF-IDF | XLM-RoBERTa |
|---|---|---|---|---|
| yaar tu kitna ghatiya insaan hai | offensive (implicit) | not offensive | not offensive | not offensive |
| bhai mast match tha aaj | not offensive | not offensive | not offensive | not offensive |
| teri shakal dekh ke hi pata chal jata hai | ambiguous | offensive | not offensive | not offensive |
| chal hatt yaha se bsdk | offensive | offensive | offensive | offensive |
| Chill bhai kya kar raha hai | not offensive | not offensive | not offensive | not offensive |
| Abbe jaana chutiye | offensive | offensive | offensive | offensive |
| bhag gandu ma chuda jaake | offensive | offensive | offensive | offensive |
| Yo i love that pizza bhai it is so good | not offensive | not offensive | not offensive | not offensive |
| Yeah I know usko farak nahi padta bhai | not offensive | not offensive | not offensive | not offensive |
| Uske mama ka bh0s*a uski ma k! ch!t | offensive | offensive | offensive |

**Main takeaway:** None of the models caught the offensive sentence without explicit slurs ("yaar tu kitna ghatiya insaan hai"). They only flagged sentences with clear trigger words—doesn’t matter if you use TF-IDF or a transformer.

The bottom line is all three models basically just picked up on explicit offensive words. They didn’t build any sense of context or more subtle, implied insult—mainly because the data didn't require that skill. If you want a model to spot less obvious offense, you first need examples of that in the training set.

## Conclusion

On paper, these models look impressive: 98–99% accuracy. But the task ends up being "can you spot a known slur," not "can you judge real-world offensiveness." Yes, the transformer did a bit better than basic models, but even it failed to recognize offense delivered with ordinary words. Anyone building a deployable Hinglish abuse detector shouldn’t take these numbers at face value.

## Limitations & What’s Next

- **Dataset realism:** We need datasets scraped from actual Hinglish conversations (e.g., YouTube or Twitter) to test if any of this holds up with “in the wild” language.
- **Implicit offense:** No model here handles offense that doesn’t rely on clear trigger words. For that, you need proper examples in the data, not just better models.
- **Dataset size:** 1,300 examples is tiny, especially for transformers. Results might change with a big, naturally diverse dataset.
- **Single split:** Everything here is from one random 80/20 split. Running cross-validation would show how stable these results are, especially since the models made so few mistakes (2 out of ~270).

## Repro Notes

- Used `xlm-roberta-base`: 4 epochs, lr=2e-5, batch size 16, max_length=64.
- If you’re short on compute (like using Colab’s free tier), `distilbert-base-multilingual-cased` mostly works as a replacement.
- Dropped duplicates before splitting, to make sure the test set didn’t leak training examples.
