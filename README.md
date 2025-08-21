# Subtask 2: Passage Retrieval 

The subtask is defined as follows: Given a free-text question posed in MSA, a collection of Qur'anic passages (that cover the Holy Qur'an) and a collection of Hadiths from Sahih Bukhari, a system is required to retrieve a ranked list of up-to 20 answer-bearing Qur'anic passages or Hadiths (i.e., Islamic sources that potentially enclose the answer(s) to the given question) from the two collections. The question can be a factoid or non-factoid question. 
To make the task more realistic (thus challenging), some questions may not have an answer in the Holy Qur'an and Sahih Al-Bukhari. In such cases, the ideal system should return no answers; otherwise, it returns a ranked list of up to 20 answer-bearing sources.

# Data

You can find both the official subtask data as well as external datasets used (i.e. TyDi, QuQA, HaQA, Jalalayn Tafseer). Pre-processing of these datasets can be found under: src\Utils

# Official Runs

To run the three configs submitted for test data, run:

```bash
python src/Cross-encoder/crossenc_test.py
```

```bash
python src/Gemini/reranking_gemini.py
```
