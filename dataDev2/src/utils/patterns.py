# From FLAN Templates (flan_templates):
# https://github.com/google-research/FLAN/blob/main/flan/v2/templates.py
# Licensed under the Apache License, Version 2.0 (the "License")

PATTERNS = {
    "rte": [
        ("{premise}\n\nQuestion with options: Based on the paragraph above can"
         " we conclude that \"{hypothesis}\"?\n\n{options_}", "{answer}"),
        ("{premise}\n\nBased on that paragraph can we conclude that the "
         "sentence below is true?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("{premise}\n\nQ with options: Can we draw the following "
         "conclusion?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("{premise}\nDoes this next sentence follow, given the preceding "
         "text?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("{premise}\n{options_}\nQuestion: Can we infer the "
         "following?\n{hypothesis}", "{answer}"),
        ("Read the following paragraph and determine if the hypothesis is "
         "true. Select from options at the end:\n\n{premise}\n\nHypothesis: "
         "{hypothesis}\n{options_}\nThe answer is", "{answer}"),
        ("Read the text and determine if the sentence is "
         "true:\n\n{premise}\n\nSentence: {hypothesis}\n{options_}\nA:",
         "{answer}"),
        ("Question with options: can we draw the following hypothesis from the"
         " context? \n\nContext:\n\n{premise}\n\nHypothesis: "
         "{hypothesis}\n{options_}\nA:", "{answer}"),
        ("Determine if the sentence is true based on the text below. Choose "
         "from options.\n{hypothesis}\n\n{premise}\n{options_}", "{answer}"),
        ("Generate a context and a hypothesis.",
         "Context: {premise}\n\nHypothesis: {hypothesis}"),
    ],
    "wsc": [
        ("{context}\n\nWhich option(s) below is/are correct for question: are "
         "\"{text1}\" and \"{text2}\" the same entity?\n\n{options_}",
         "{answer}"),
        ("{context}\n\nMulti-choice question: Do \"{text1}\" and \"{text2}\" "
         "have the same meaning?\n\n{options_}", "{answer}"),
        ("Multi-choice problem: Given the following "
         "context\n\n{context}\n\nAre \"{text1}\" and \"{text2}\" the "
         "same?\n\n{options_}\nA:", "{answer}"),
        ("Choose your answer.{options_}.\n\n{context}\n\nDo \"{text2}\" and "
         "\"{text1}\" mean the same thing?", "{answer}"),
        ("{context}\n\nAre \"{text2}\" and \"{text1}\" the same thing in the "
         "aforementioned sentence (choose from options)?\n\n{options_}",
         "{answer}"),
        ("Context:{context}\n\nIs \"{text2}\" the same as \"{text1}\"? "
         "Possible answers:{options_}\n\nAnswer:", "{answer}"),
        ("Consider this sentence: {context}\n\nAre \"{text2}\" and \"{text1}\""
         " the same (see options)?\n\n{options_}", "{answer}"),
        ("Are \"{text1}\" and \"{text2}\" the same in this "
         "sentence?\n{context}\n\n{options_}\nThe answer is:", "{answer}"),
        ("See context followed by options. Is \"{text1}\" the same as "
         "\"{text2}\" in this sentence?\n{context}\n\n{options_}", "{answer}"),
        ("Choose your answer: Do \"{text1}\" and \"{text2}\" point to the same"
         " thing in the following sentence?\n\n{context}\n\n{options_}",
         "{answer}"),
    ],
    "wsc273": [
        ("Multi-choice problem: {context}\n{options_}", "{answer}"),
        ("Complete the passage.\n\n{context}\n{options_}", "{answer}"),
        ("How does this following sentence end (see "
         "options)?\n\n{context}\n{options_}", "{answer}"),
        ("What is the most logical completion for the following text (see "
         "options)?\n\n{context}\n{options_}", "{answer}"),
        ("Multi-choice problem: How does this text "
         "end?\n\n{context}\n{options_}", "{answer}"),
        ("Choose from the options on what happens "
         "next.\n\n{context}\n{options_}", "{answer}"),
        ("Complete the following sentence.\n\n{context}\n{options_}",
         "{answer}"),
        ("Choose from options: Fill in the remainder of the "
         "sentence.\n\n{context}\n{options_}", "{answer}"),
        ("What is the next event listed in the options is "
         "correct?\n\n{context}\n{options_}\nA:", "{answer}"),
        ("Complete the rest of the sentence by choosing from "
         "options.\n\n{context}\n{options_}", "{answer}"),
    ],
    "wic": [
        ("{sentence1}\n{sentence2}\nChoose your answer: Does the word "
         "\"{word}\" mean the same thing in the above two "
         "sentences?\n{options_}", "{answer}"),
        ("Sentence 1: {sentence1}\nSentence 2: {sentence2}\nMulti-choice "
         "problem: Does {word} mean the same thing in these two "
         "sentences?\n{options_}", "{answer}"),
        ("Here is one sentence: {sentence1}\nHere is another sentence: "
         "{sentence2}\nQ: Does the term {word} mean the same thing in both "
         "these sentences?\n{options_}", "{answer}"),
        ("In these two sentences (1) {sentence1} (2) {sentence2}, does the "
         "word {word} mean the same thing?\n{options_}.... A:", "{answer}"),
        ("Multi-choice problem: does word \"{word}\" have the same meaning in "
         "the following two "
         "sentences?\n\n{sentence1}\n\n{sentence2}\n\n{options_}", "{answer}"),
        ("This question has options. Is the word \"{word}\" used in the same "
         "way in the following two "
         "sentences?\n\n{sentence1}\n\n{sentence2}\n\n{options_}", "{answer}"),
        ("This question has options. Does the word \"{word}\" have the same "
         "definition in the next two "
         "sentences?\n\n{sentence1}\n\n{sentence2}\n\n{options_}", "{answer}"),
        ("Is {word} used to mean the same thing in the next two sentences (see"
         " options)?\n\n{sentence1}\n\n{sentence2}\n\n{options_}", "{answer}"),
        ("Does \"{word}\" mean the same thing in these two sentences? See "
         "options at the end. \n{sentence1}\n{sentence2}\n{options_}..Answer:",
         "{answer}"),
        ("(options at the end). Does the word \"{word}\" mean the same thing "
         "in \"{sentence1}\" and \"{sentence2}\"?\n{options_}", "{answer}"),
    ],
    "record": [
        ("Complete the passage: pick from possible "
         "candidates.\n\n{passage}\n\n{query}\n\n{options_str}\n\n",
         "{answer}"),
        ("{passage}\n\n{query}\n\n{options_str}\n\n", "{answer}"),
        ("Find the right ending to this "
         "passage.\n\n{passage}\n\n{query}\n\n{options_str}\n\n", "{answer}"),
        ("What's the most logical way to complete this "
         "passage?\n\n{passage}\n\n{query}\n\n{options_str}\n\n", "{answer}"),
        ("Choose the next sentence."
         "{passage}\n\n{query}\n\n{options_str}\n\n", "{answer}"),
        ("Choose how you want this story to "
         "end.\n\n{passage}\n\n{query}\n\n{options_str}\n\n", "{answer}"),
        ("Write the last sentence in this "
         "story.\n\n{passage}\n\n{query}\n\n{options_str}\n\n", "{answer}"),
        ("Choose the next sentence for this "
         "paragraph.\n\n{passage}\n\n{query}\n\n{options_str}\n\n", "{answer}"),
        ("What is the most logical completion of this news "
         "story?.\n\n{passage}\n\n{query}\n\n{options_str}\n\n", "{answer}"),
        ("How does the sentence end?\n\n"
         "{passage}\n\n{query}\n\n{options_str}\n\n", "{answer}"),
    ],
    "natural_questions": [
        ("Question: {question}?\nAnswer:", "{answer}"),
        ("{question}?", "{answer}"),
        ("Answer the following question:\n\n{question}", "{answer}"),
        ("Answer this question:\n\n{question}?", "{answer}"),
        ("Please answer this question: {question}", "{answer}"),
        ("Answer the question...{question}?", "{answer}"),
        ("What is the answer to this question? {question}\n\n", "{answer}"),
        ("Can you tell me the answer to {question}?", "{answer}"),
        ("Next question: {question}\n\n", "{answer}"),
        ("Q: {question} A:", "{answer}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "synth_cot_natural_questions": [
        ("Question: {question}?\nCoT:", "{cot}\nThe answer is {answer}"),
        ("{question}? Let's think step by step.",
         "{cot}\n\nThe answer is {answer}"),
        ("Answer the following question carefully:\n\n{question}",
         "\n{cot}\nThe answer is {answer}"),
        ("Answer this question:\n\n{question}? Think out loud!",
         "{cot}\nSo, the answer is {answer}"),
        ("Please answer this question: {question}\nGive your reasons first.",
         "{cot}\n\nThe answer: {answer}"),
        ("Answer the question...{question}? Give your explanation afterwards",
         "The answer: {answer}\nExplanation: {cot}"),
        ("What is the answer to this question? {question}\nLet's think...",
         "{cot}. So the answer is {answer}."),
        ("Can you tell me the logic and answer to {question}?",
         "logic: {cot}\n\nThe final answer: {answer}"),
        ("Next question: {question}\n\nSolution:",
         "{cot}\nThe answer is {answer}"),
        ("Q: {question} Step-by-step reasoning process:",
         "{cot} The answer is {answer}"),
    ],
    "trivia_qa": [
        ("Please answer this question: {question}", "{answer}"),
        ("{question}", "{answer}"),
        ("Write the answer: {question}", "{answer}"),
        ("What is the answer: {question}", "{answer}"),
        ("Answer this question.\n\n{question}", "{answer}"),
        ("Answer the following question. {question}", "{answer}"),
        ("Question: {question}\nAnswer:", "{answer}"),
        ("{question}???", "{answer}"),
        ("Trivia question: {question}\nAnd the answer is?", "{answer}"),
        ("{question}\nWhat is the answer?", "{answer}"),
    ],
    "math_dataset": [
        ("{question}", "{answer}"),
        ("Solve this math problem\n\n{question}", "{answer}"),
        ("What is the solution?\n\n{question}", "{answer}"),
        ("Math Problem\n{question}", "{answer}"),
        ("Write down the solution for this math problem: {question}",
         "{answer}"),
        ("What is the solution to this math problem?\n{question}", "{answer}"),
        ("Math problem: {question}\nWhat is the solution?", "{answer}"),
        ("{question}\nSolve this problem.", "{answer}"),
        ("Problem: {question}\nAnd the answer is...", "{answer}"),
        ("{question}. What is the answer??", "{answer}"),
    ],
    "aeslc": [
        ("What is the subject line for this email?\n\n{body}\n\nSubject Line:",
         "{subject}"),
        ("Write a subject line for this message:\n\n{body}\n\nSubject Line:",
         "{subject}"),
        ("{body}\nWrite a subject line for this email.", "{subject}"),
        ("Here is an email: {body}\nWhat is a potential subject line for this "
         "email?", "{subject}"),
        ("{body}\nPropose a subject line for this email?", "{subject}"),
        ("This is the content of an email: {body}\nWhat was the subject line "
         "for this email?", "{subject}"),
        ("This is an email\n{body}\n\nWhat is the subject of this email?",
         "{subject}"),
        ("{body}\n\nGenerate a subject line for this email.", "{subject}"),
        ("Write an email with the following subject:\n\n{subject}\n\nEmail:",
         "{body}"),
        ("Write an email with the subject line \"{subject}\".", "{body}"),
    ],
    "cnn_dailymail": [
        ("Write highlights for this article:\n\n{text}\n\nHighlights:",
         "{highlights}"),
        ("Write some highlights for the following "
         "article:\n\n{text}\n\nHighlights:", "{highlights}"),
        ("{text}\n\nWrite highlights for this article.", "{highlights}"),
        ("{text}\n\nWhat are highlight points for this article?",
         "{highlights}"),
        ("{text}\nSummarize the highlights of this article.", "{highlights}"),
        ("{text}\nWhat are the important parts of this article?",
         "{highlights}"),
        ("{text}\nHere is a summary of the highlights for this article:",
         "{highlights}"),
        ("Write an article using the following "
         "points:\n\n{highlights}\n\nArticle:", "{text}"),
        ("Use the following highlights to write an "
         "article:\n\n{highlights}\n\nArticle:", "{text}"),
        ("{highlights}\n\nWrite an article based on these highlights.",
         "{text}"),
    ],
    "gigaword": [
        ("Write a short summary for this text: {text}\n\nSummary:",
         "{summary}"),
        ("Briefly summarize this sentence: {text}\n\nSummary:", "{summary}"),
        ("Generate a short summary this sentence:\n{text}\n\nSummary:",
         "{summary}"),
        ("What is a shorter version of this:\n\n{text}\n\nSummary:",
         "{summary}"),
        ("{text}\n\nWrite a brief summary in a sentence or less.", "{summary}"),
        ("{text}\n\nWhat is a very short summary of the above text?",
         "{summary}"),
        ("{text}\nSummarize the aforementioned text in a single phrase.",
         "{summary}"),
        ("{text}\nCan you generate a short summary of the above paragraph?",
         "{summary}"),
        ("Write a text based on this summary: {summary}\n\nText:", "{text}"),
        ("Write a text based on \"{summary}\"\n\nText:", "{text}"),
    ],
    "multi_news": [
        ("Summarize this article:\n\n{text}\n\nSummary:", "{summary}"),
        ("Write a summary based on this article:\n\n{text}\n\nSummary:",
         "{summary}"),
        ("Article:\n\n{text}\nWhat is a summary?", "{summary}"),
        ("{text}\nWhat is a one-paragraph summary of the above article?",
         "{summary}"),
        ("Here is a news article: {text}\nA summary of this is?", "{summary}"),
        ("News article:\n\n{text}\nWhat is a shorter version of the above "
         "article?", "{summary}"),
        ("{text}\n\nWrite a summary.", "{summary}"),
        ("Article:\n{text}Summary:", "\n{summary}"),
        ("Write an article based on this summary:\n\n{summary}\n\nArticle:",
         "\n{text}"),
        ("{summary}\n\nExpand this summary.", "{text}"),
    ],
    "newsroom": [
        ("{title}\n\n{text}\n\nWrite a one or two sentence summary.",
         "{summary}"),
        ("Please write a short summary for the following "
         "article:\n\n{title}\n\n{text}\n\nSummary:", "{summary}"),
        ("Please briefly summarize this news "
         "article:\n\n{title}\n\n{text}\n\nSummary:", "{summary}"),
        ("{title}\n{text}\nWhat was this article about?", "{summary}"),
        ("{title}\n{text}\nWhat is a short summary of the above article?",
         "{summary}"),
        ("{title}\n\n{text}\nWhat are the most important parts of this text?",
         "{summary}"),
        ("News article: {title}\n\n{text}\nWhat are the most important parts "
         "of this news article?", "{summary}"),
        ("Write an article with the title: \"{title}\"\n\nArticle:",
         "\n{text}"),
        ("Write a title for this article:\n\n{text}\n\nTitle:", "{title}"),
        ("Here is an article:\n\n{text}\n\nWrite a title for it.", "{title}"),
    ],
    "samsum": [
        ("{dialogue}\n\nBriefly summarize that dialogue.", "{summary}"),
        ("Here is a dialogue:\n{dialogue}\n\nWrite a short summary!",
         "{summary}"),
        ("Dialogue:\n{dialogue}\n\nWhat is a summary of this dialogue?",
         "{summary}"),
        ("{dialogue}\n\nWhat was that dialogue about, in two sentences or less?",
         "{summary}"),
        ("Here is a dialogue:\n{dialogue}\n\nWhat were they talking about?",
         "{summary}"),
        ("Dialogue:\n{dialogue}\nWhat were the main points in that "
         "conversation?", "{summary}"),
        ("Dialogue:\n{dialogue}\nWhat was going on in that conversation?",
         "{summary}"),
        ("Write a dialog about anything you want.", "{dialogue}"),
        ("Write a dialog based on this summary:\n{summary}.", "{dialogue}"),
        ("Write a dialog with this premise \"{summary}\".", "{dialogue}"),
    ],
    "xsum": [
        ("Summarize:\n\n{text}\n\nSummary:", "{summary}"),
        ("Summarize this article:\n\n{text}\n\nSummary:", "{summary}"),
        ("Summarize this article in one sentence.\n\n{text}\n\nSummary:",
         "{summary}"),
        ("{text}\nWhat is a summary of this text?", "{summary}"),
        ("{text}\nWhat was that article about?", "{summary}"),
        ("{text}\n\nThis article was about:", "{summary}"),
        ("Article:{text}\n\nA summary of the above article is?", "{summary}"),
        ("Article:{text}\n\nSummarize the main points of that article.",
         "{summary}"),
        ("Write an article based on this summary:\n\n{summary}\n\nArticle:",
         "{text}"),
        ("Write an article based on this \"{summary}\"\n\nArticle:", "{text}"),
    ],
    "squad_v1": [
        ("Please answer a question about the following article about "
         "{title}:\n\n{context}\n\n{question}", "{answer}"),
        ("Read this and answer the question\n\n{context}\n\n{question}",
         "{answer}"),
        ("{context}\n{question}", "{answer}"),
        ("Answer a question about this article:\n{context}\n{question}",
         "{answer}"),
        ("Here is a question about this article: {context}\nWhat is the answer"
         " to this question: {question}", "{answer}"),
        ("Article: {context}\n\nQuestion: {question}", "{answer}"),
        ("Article: {context}\n\nNow answer this question: {question}",
         "{answer}"),
        ("{title}\n{context}\n\nQ: {question}", "{answer}"),
        ("Ask a question about {title}.", "{question}"),
        ("What is the title of this article:\n\n{context}\n\nTitle:",
         "{title}"),
    ],
    "squad_v2": [
        ("{title}:\n\n{context}\n\nPlease answer a question about this "
         "article. If the question is unanswerable, say \"unanswerable\". "
         "{question}", "{answer}"),
        ("Read this and answer the question. If the question is unanswerable, "
         "say \"unanswerable\".\n\n{context}\n\n{question}", "{answer}"),
        ("What is a question about this article? If the question is "
         "unanswerable, say \"unanswerable\".\n\n{context}\n\n{question}",
         "{answer}"),
        ("{context}\n{question} (If the question is unanswerable, say "
         "\"unanswerable\")", "{answer}"),
        ("{context}\nTry to answer this question if possible (otherwise reply "
         "\"unanswerable\"): {question}", "{answer}"),
        ("{context}\nIf it is possible to answer this question, answer it for "
         "me (else, reply \"unanswerable\"): {question}", "{answer}"),
        ("{context}\n\nAnswer this question, if possible (if impossible, reply"
         " \"unanswerable\"): {question}", "{answer}"),
        ("Read this: {context}\n\n{question}\nWhat is the answer? (If it "
         "cannot be answered, return \"unanswerable\")", "{answer}"),
        ("Read this: {context}\nNow answer this question, if there is an "
         "answer (If it cannot be answered, return \"unanswerable\"): "
         "{question}", "{answer}"),
        ("{context}\nIs there an answer to this question (If it cannot be "
         "answered, say \"unanswerable\"): {question}", "{answer}"),
    ],
    "drop": [
        ("Answer based on context:\n\n{context}\n\n{question}", "{answer}"),
        ("{context}\n\nAnswer this question based on the article: {question}",
         "{answer}"),
        ("{context}\n\n{question}", "{answer}"),
        ("{context}\nAnswer this question: {question}", "{answer}"),
        ("Read this article and answer this question {context}\n{question}",
         "{answer}"),
        ("{context}\n\nBased on the above article, answer a question. "
         "{question}", "{answer}"),
        ("Context: {context}\n\nQuestion: {question}\n\nAnswer:", "{answer}"),
        ("Write an article that answers the following question: {question}",
         "{context}"),
        ("Write a question about the following article: {context}\n\nQuestion "
         "about the article:", "{question}"),
        ("{context}\n\nAsk a question about this article.", "{question}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "synth_cot_drop": [
        ("Answer based on context:\n\n{context}\n\n{question}.\n\n"
         "Let's think step by step:", "{cot}\nThe answer is {answer}"),
        ("{context}\n\nAnswer this question by reasoning step-by-step based on "
         "the article: {question}", "{cot} The answer is {answer}"),
        ("{context}\n\n{question}\nStep-by-step reasoning process:",
         "\n{cot} The answer is {answer}"),
        ("{context}\nAnswer this question: {question}. Now, let me think...",
         "\n{cot}\nThe answer is {answer}"),
        ("Read the following article then answer the question. "
         "Explain your answer afterwards.\n{context}\n{question}\n",
         "The answer is {answer}.\nExplanation: {cot}"),
        ("{context}\n\nBased on the above article, answer a question. "
         "{question}\nI need you to give me your thought process first.",
         "{cot}\nThe answer is {answer}"),
        ("Context: {context}\n\nQuestion: {question}\n\nYour thought:",
         "{cot} The answer is {answer}"),
        ("{context}\n{question}\nWhat do you think? I think:",
         "{cot} The answer is {answer}"),
        ("{context} {question} Chain-of-thought:",
         "{cot} The answer is {answer}"),
        ("{context} {question}\n Let's think step by step:\n",
         "{cot} The answer is {answer}"),
    ],
    "quac": [
        ("{background}\n\n{context}\n\nAnswer the following question by taking"
         " a quote from the article: {question}", "{answer}"),
        ("{background}\n\n{context}\n\nUsing a quote from the above article, "
         "answer the following question: {question}", "{answer}"),
        ("Answer by taking a quote from the following "
         "article:\n\n{background}\n\n{context}\n\n{question}", "{answer}"),
        ("{background}\n\n{context}\n\n{question}", "{answer}"),
        ("Background: {background}\nContext: {context}\nQuestion: "
         "{question}\n\nAnswer:", "{answer}"),
        ("Background: {background}\nContext: {context}\nQuestion: {question}. "
         "Whats the answer?", "{answer}"),
        ("{context}\n\nAnswer this question \"{question}\" by extracting the "
         "answer from the text above.", "{answer}"),
        ("{background}\n\n{context} Answer this question using a quote from"
         " the following article:\n\n{question}", "{answer}"),
        ("Which entity is this text "
         "about?\n\n{background}\n\n{context}\n\nEntity:", "{title}"),
        ("{background}\n\n{context}\n\nAsk a question about this article.",
         "{question}"),
    ],
    "para_crawl": [
        ("How do you say \"{sent1}\" in {lang2}?", "{sent2}"),
        ("{sent2} How do you say this sentence in {lang1}?", "{sent1}"),
        ("{sent1} Say this using {lang2}.", "{sent2}"),
        ("Translate from {lang1} to {lang2}:\n\n{sent1}\n\n{lang2}:",
         "{sent2}"),
        ("Translate from {lang2} to {lang1}:\n\n{sent2}\n\n{lang1}:",
         "{sent1}"),
        ("Translate \"{sent2}\" from {lang2} to {lang1}.", "{sent1}"),
        ("Translate \"{sent1}\" to {lang2}.", "{sent2}"),
        ("Translate the following.\n\n{lang1}: {sent1}\n\n{lang2}:", "{sent2}"),
        ("Write a sentence in {lang1}.", "{sent1}"),
        ("Write a sentence in {lang2}.", "{sent2}"),
    ],
    "wmt16_translate": [
        ("{sent1}\n\nTranslate to {lang2}\n\n{lang2}:", "{sent2}"),
        ("{sent2}\n\nTranslate to {lang1}\n\n{lang1}:", "{sent1}"),
        ("{sent2}\n\nCould you please translate this to {lang1}?", "{sent1}"),
        ("{sent2}\n\nTranslate this to {lang1}?", "{sent1}"),
        ("Translate to {lang2}:\n\n{sent1}\n\n{lang2}:", "{sent2}"),
        ("Translate the following sentence to {lang2}:\n{sent1}\n\n{lang2}:",
         "{sent2}"),
        ("How is \"{sent1}\" said in {lang2}?", "{sent2}"),
        ("Translate \"{sent1}\" to {lang2}?", "{sent2}"),
        ("Write a sentence not in {lang1}.", "{sent2}"),
        ("{sent2}\n\nWhich language is this?", "{lang2}"),
    ],
    "wmt14_enfr": [
        ("{sent1}\n\nTranslate to {lang2}.", "{sent2}"),
        ("{sent2}\n\nTranslate to {lang1}.", "{sent1}"),
        ("{sent2}\n\nCould you please translate this to {lang1}?", "{sent1}"),
        ("{sent2}\n\nTranslate this to {lang1}?", "{sent1}"),
        ("Translate to {lang2}:\n\n{sent1}\n\n", "{sent2}"),
        ("Translate the following sentence to {lang2}:\n{sent1}\n\n",
         "{sent2}"),
        ("How is \"{sent1}\" said in {lang2}?", "{sent2}"),
        ("Translate \"{sent1}\" to {lang2}?", "{sent2}"),
        ("Write a sentence not in {lang1}.", "{sent2}"),
        ("{sent2}\n\nWhich language is this?", "{lang2}"),
    ],
    "true_case": [
        ("{lower}\n\nPlease write the text above using proper case.",
         "{answer}"),
        ("{lower}\n\nWrite the above sentence using proper case.", "{answer}"),
        ("{lower}\n\nHow would the previous sentence be correctly capitalized?",
         "{answer}"),
        ("{lower}\nCapitalize this past sentence correctly.", "{answer}"),
        ("{lower}\nRepeat this setence, but with the correct capitalization.",
         "{answer}"),
        ("{lower}\nCan you repeat this sentence, but capitalize it correctly?",
         "{answer}"),
        ("{lower}\n\nThis is all lower case. Can you fix that?", "{answer}"),
        ("{lower}\n\nMake this proper case.", "{answer}"),
        ("Please capitalize where necessary: {lower}", "{answer}"),
        ("{answer}\n\nMake this lower case.", "{lower}"),
    ],
    "fix_punct": [
        ("{no_punct}\n\nAdd punctuation.", "{answer}"),
        ("{no_punct}\n\nCan you repeat this sentence, but add in punctuation?",
         "{answer}"),
        ("{no_punct}\n\nWhat is the correctly punctuated version of this "
         "sentence?", "{answer}"),
        ("{no_punct}\n\nPlease fix the punctuation.", "{answer}"),
        ("{no_punct}\n\nCould you correct the punctuation please?", "{answer}"),
        ("Please add punctuation to this: {no_punct}\n\nPunctuation version:",
         "{answer}"),
        ("Add punctuation: {no_punct}\n\n", "{answer}"),
        ("Add punctuation to the following sentence: {no_punct}\n\n",
         "{answer}"),
        ("Generate a correctly punctuated version of the following text: "
         "{no_punct}\n\n", "{answer}"),
        ("What is the version of the following sentence with correct "
         "punctuation?\n\n{no_punct}\n\n", "{answer}"),
    ],
    # "word_segment": [
    #     ("{no_space}\nGenerate a sentence using the above characters:",
    #      "{answer}"),
    #     ("{no_space}\nWhat's a sentence that uses these characters?",
    #      "{answer}"),
    #     ("{no_space}\n\nPlease segment the words:", "{answer}"),
    #     ("Add spaces: {no_space}\n\n", "{answer}"),
    #     ("Please add spaces between words: {no_space}\n\n", "{answer}"),
    #     ("This text is missing some spaces, please add them: {no_space}\n\n",
    #      "{answer}"),
    #     ("Add spaces between the words in the following text: {no_space}\n\n",
    #      "{answer}"),
    #     ("Write the following list of characters into a correctly formed "
    #      "sentence: {no_space}\n\n", "{answer}"),
    #     ("{answer}\n\nPlease remove spaces between words.", "{no_space}"),
    #     ("Remove the spaces from the following sentence: {answer}",
    #      "{no_space}"),
    # ],
    "cosmos_qa": [
        ("{context}\n\nQuestion with options to choose from: "
         "{question}\n{options_}", "{answer}"),
        ("{context}\n\n{options_}\nQ: {question}", "{answer}"),
        ("{context}\n\n{options_}\nAnswer the following question: {question}\n",
         "{answer}"),
        ("{context}\n\nBased on the preceding passage, choose your answer for "
         "question {question}\n{options_}\nThe answer is:", "{answer}"),
        ("{context}\n\nQ with options: Give answer the following question "
         "using evidence from the above passage: {question}\n{options_}",
         "{answer}"),
        ("Context: {context}\nQuestion {question}\nPossible "
         "answers:\n{options_}\nThe answer:", "{answer}"),
        ("Read the following article and answer the question by choosing from "
         "the options.\n\n{context}\n\n{question}\n{options_}...A:",
         "{answer}"),
        ("This question has options. Answer the question about "
         "text:\n\n{context}\n\n{question}\n{options_}", "{answer}"),
        ("Write a question about the following article."
         "\n\n{context}\n\nQuestion:", "{question}\n{options_}"),
        ("{context}\n\nGenerate a question about the above context.",
         "{question}\n{options_}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "synth_cot_cosmos_qa": [
        ("{context}\n\nQuestion: {question}\n{options_}\n"
         "Let's answer step by step.", "{cot} So the answer is {answer}"),
        ("{context}\n\n{options_}\nQ: {question}\nStep by step reasoning:",
         "{cot} The answer is {answer}"),
        ("{context}\n\n{options_}\nLet's answer this carefully: {question}\n",
         "{cot}\nThe answer is {answer}"),
        ("{context}\n\nBased on the preceding passage, answer question "
         "{question}\n{options_}\nLet's solve slowly:",
         "{cot} The answer is {answer}"),
        ("{context}\nSolve the following question "
         "thinking out loud: {question}\n{options_}",
         "{cot} So, the answer is {answer}"),
        ("Context: {context}\nQuestion: {question}\n"
         "\n{options_}\nLet's think:", "{cot}... So the answer is {answer}"),
        ("Read the following article and answer the question."
         "\n{context}\n\n{question}\n{options_}..."
         "Chain-of-thought:", "{cot}\nThe answer is {answer}"),
        ("Answer the question about text:\n\n{context}\n\n{question}\n"
         "{options_}\nCoT:", "{cot} The answer is {answer}"),
        ("{context}\nQuestion: {question}\n{options_}\nChain-of-thought:",
         "{cot} The answer is {answer}"),
        ("Context: {context}\nQ: {question}\n{options_}\nStep-by-step "
         "reasoning process:", "{cot}\nThe answer is {answer}"),
    ],
    "ag_news_subset": [
        ("{title}\n\n{text}\n\nMulti-choice problem: What is this text "
         "about?\n{options_}", "{answer}"),
        ("Choose your answer. {title}\n\n{text}\n\nWhich topic is this article"
         " about?\n{options_}", "{answer}"),
        ("{text}\nQ: Which is the best summary of this article?\n{options_}\nI"
         " think the answer is", "{answer}"),
        ("{text}\nChoose your answer. What is this text "
         "about?\n{options_}\nAnswer:", "{answer}"),
        ("{text}\n\nWhat best summarizes the content of the above "
         "article?\n{options_}", "{answer}"),
        ("Select your answer: Which is this about?\n\n{text}\n\n{options_}",
         "{answer}"),
        ("Select the correct answer: Which is an appropriate title for this "
         "article?\n\n{text}\n\n{options_}", "{answer}"),
        ("Note the options at the end. Select the topic that this "
         "about:\n\n{text}\n\n{options_}", "{answer}"),
        ("Write a title:\n{text}\nTitle:", "{title}"),
        ("{text}\n\nWhat is a good title for this?", "{title}"),
    ],
    "bool_q": [
        ("{text}\n\nSee options at the end. Can we conclude that "
         "{question}?\n\n{options_}", "{answer}"),
        ("{text}\n\nMulti-choice problem: Is it true that "
         "{question}?\n\n{options_}\nThe answer is:", "{answer}"),
        ("{text}\n\n{question}?\n\n{options_}", "{answer}"),
        ("Text: {text}\n\nQuestion: {question}?\n\n{options_}", "{answer}"),
        ("{text}\n\nWhat's the best answer to this question: "
         "{question}?\n\n{options_}...A:", "{answer}"),
        ("{text}\nBased on the above text, what's the best answer to this "
         "question: {question}?\n\n{options_}", "{answer}"),
        ("{text}\nAnswer this question, making sure that the answer is "
         "supported by the text: {question}?\n\n{options_}", "{answer}"),
        ("{text}\n\nChoose your answer: Is the following statement correct "
         "based on the text\n\n{question}\n\n{options_}", "{answer}"),
        ("{title}\n\n{text}\n\n{options_}\nIs this statement correct "
         "\"{question}\"?", "{answer}"),
        ("Is it true that {question} based on the following "
         "text?\n\n{text}\n\n{options_}", "{answer}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "synth_cot_bool_q": [
        ("{passage}\n\nThink out loud. Can we conclude that "
         "{question}?\n\n{options_}", "{cot}. The answer is {answer}"),
        ("{passage}\n\nIs it true that {question}?\n\n{options_}\n"
         "Your thought:", "{cot}. The answer is {answer}"),
        ("{passage}\n\n{question}?\n\n{options_}\nLet's think step by step.",
         "{cot}\nThe answer is {answer}"),
        ("Answer the following question carefully. Think out loud.\n"
         "passage: {passage}\n\nQuestion: {question}?\n\n{options_}",
         "{cot}\nThe answer is {answer}"),
        ("Give the reasoning before answering any question.\n{passage}\n\n"
         "What's the best answer to this question: {question}?\n{options_}...",
         "{cot}. The answer is {answer}"),
        ("{passage}\nBased on the above text, what's the best answer to this "
         "question: {question}?\n{options_}\nLet's think.",
         "{cot}. Final answer: {answer}"),
        ("{passage}\nAnswer this question carefully, making sure that the "
         "answer is supported by the text: {question}?\n\n{options_}"
         "Step-by-step reasoning process:",
         "{cot}. I think the answer is {answer}"),
        ("{passage}\n\nChoose your answer: Is the following statement correct "
         "based on the passage\n\n{question}\n\n{options_}\nChain-of-thought:",
         "{cot}\nThe answer is {answer}"),
        ("{title}\n\n{passage}\n\n{options_}\nIs this statement correct "
         "\"{question}\"? Chain-of-thought:", "{cot}\nThe answer is {answer}"),
        ("Is it true that {question} based on the following "
         "passage?\n\n{passage}\n\n{options_}\nSay why you think so.",
         "{cot}. The answer is {answer}"),
    ],
    # "definite_pronoun_resolution": [
    #     ("{sentence}\n\n{options_}\nWho is {pronoun} referring to?",
    #      "{answer}"),
    #     ("{sentence}\n\nWho is \"{pronoun}\" in this prior sentence(see "
    #      "options)?\n{options_}", "{answer}"),
    #     ("{sentence}\n\nWho is {pronoun} referring to in this "
    #      "sentence?\n{options_}", "{answer}"),
    #     ("Choose your answer: {sentence}\nTell me who {pronoun} is.\n{options_}",
    #      "{answer}"),
    #     ("{sentence}\nBased on this sentence, who is {pronoun}?\n\n{options_}",
    #      "{answer}"),
    #     ("Choose your answer: Who is {pronoun} in the following "
    #      "sentence?\n\n{sentence}\n\n{options_}", "{answer}"),
    #     ("Multi-choice problem: Which entity is {pronoun} this "
    #      "sentence?\n\n{sentence}\n\n{options_}", "{answer}"),
    #     ("Who is {pronoun} referring to in the following "
    #      "sentence?\n{sentence}\n\n{options_}", "{answer}"),
    #     ("Note that this question lists possible answers. Which person is "
    #      "{pronoun} referring to in the following "
    #      "sentence?\n{sentence}\n\n{options_}", "{answer}"),
    #     ("{sentence}\nWho is \"{pronoun}\"?\n{options_}", "{answer}"),
    # ],
    "glue_mrpc": [
        ("Here are two sentences:\n{sentence1}\n{sentence2}\nDo they have the "
         "same meaning?\n{options_}", "{answer}"),
        ("Here are two sentences:\n\n{sentence1}\n\n{sentence2}\nChoose your "
         "answer: are the two sentences saying the same thing?\n{options_}",
         "{answer}"),
        ("{sentence1}\n\n{sentence2}\n\nSelect from the options at the end. Do"
         " the above sentences mean the same thing?\n{options_}", "{answer}"),
        ("{sentence1}\n\n{sentence2}\n\nPlease tell me if the sentences above "
         "mean the same.\n{options_}", "{answer}"),
        ("{sentence1}\n{sentence2}\nSelect from the options at the end. Are "
         "these sentences conveying the same meaning?\n{options_}", "{answer}"),
        ("{sentence1}\n{sentence2}\n(See options at the end). If the first "
         "sentence is true, is the second one also true?\n{options_}",
         "{answer}"),
        ("{sentence1}\n{sentence2}\nAre these two sentences paraphrases of "
         "each other?\n{options_}", "{answer}"),
        ("Do the following two sentences have the same "
         "meaning?\n{sentence1}\n{sentence2}\n\n{options_}\nThe answer is:",
         "{answer}"),
        ("Do these two sentences mean the same "
         "thing?\n{sentence1}\n{sentence2}\n\n{options_}...I think the answer "
         "is", "{answer}"),
        ("Do these sentences have the same "
         "meaning?\n{sentence1}\n{sentence2}\n\n{options_}", "{answer}"),
    ],
    "glue_qqp": [
        ("{question1}\n{question2}\nMulti-choice problem: Would you say that "
         "these questions are the same?\n{options_}", "{answer}"),
        ("{question1}\n{question2}\nDo those questions have the same "
         "meaning?\n{options_}", "{answer}"),
        ("{question1}\n{question2}\n\nMulti-choice problem: Are these two "
         "questions inquiring about the same information?\n{options_}",
         "{answer}"),
        ("{question1}\n\n{question2}\n\nPlease tell me if those questions are "
         "the same.\n{options_}", "{answer}"),
        ("{question1}\n\n{question2}\n\nChoose your answer. Are these two "
         "questions paraphrases of each other?\n{options_}", "{answer}"),
        ("First question: {question1}\nSecond question: {question2}\nAre these"
         " two questions asking the same thing?\n{options_}", "{answer}"),
        ("Question 1: {question1}\nQuestion 2: {question2}\n{options_}\nAre "
         "questions 1 and 2 asking the same thing?", "{answer}"),
        ("Question 1: {question1}\nQuestion 2: {question2}\n{options_}\nWould "
         "the answer to these two questions be the same?", "{answer}"),
        ("Choose from the options at the end. Are the following two questions "
         "the same?\n{question1}\n{question2}\n\n{options_}\nThe answer is:",
         "{answer}"),
        ("Do these questions have the same "
         "meaning?\n{question1}\n{question2}\n\n{options_}", "{answer}"),
    ],
    "imdb_reviews": [
        ("{text}\nChoose your answer. What is the sentiment of this "
         "review?\n{options_}", "{answer}"),
        ("{text}\nWould you say this review is positive or "
         "negative?\n{options_}", "{answer}"),
        ("{text}\nChoose your answer. How would you describe the sentiment of "
         "this review?\n{options_}", "{answer}"),
        ("{text}\n\nIs the sentiment of this review positive or "
         "negative?\n{options_}", "{answer}"),
        ("{text}\n\nDid this review think positively or negatively of the "
         "movie (see options below)?\n{options_}...I think the answer is",
         "{answer}"),
        ("Select the correct sentiment of the following review: "
         "{text}\n{options_}", "{answer}"),
        ("Choose the correct sentiment from "
         "candidates:\n{options_}\n\nTEXT:{text}\n\n", "{answer}"),
        ("Review: {text}\nWhat is the sentiment of this review?\n{options_}",
         "{answer}"),
        ("Review: {text}\nNow, what is this review like?\n{options_}\n",
         "{answer}"),
        ("What's an example of a movie review?", "{text}"),
    ],
    "paws_wiki": [
        ("{sentence1}\n{sentence2}\n\nSelect your answer from the options. Do "
         "these sentences mean the same thing?\n{options_}", "{answer}"),
        ("{sentence1}\n{sentence2}\n\nAre these two sentences paraphrases of "
         "each other?\n{options_}", "{answer}"),
        ("1. {sentence1}\n2. {sentence2}\n\nSelect your answer from the "
         "options. Are these two sentences paraphrases of each "
         "other?\n{options_}...I think the answer is", "{answer}"),
        ("(1) {sentence1}\n(2) {sentence2}\n\nDo these two sentences mean the "
         "same thing?\n\n{options_}", "{answer}"),
        ("Sentence 1: {sentence1}\nSentence 2: {sentence2}\n\nDo these two "
         "sentences convey the same information?\n\n{options_}", "{answer}"),
        ("Do these two sentences from wikipedia have the same "
         "meaning?\n{sentence1}\n{sentence2}\n\n{options_}\nThe answer is:",
         "{answer}"),
        ("Multi-choice question: Same "
         "meaning?\n{sentence1}\n{sentence2}\n\n{options_}", "{answer}"),
        ("Are these paraphrases?\n{sentence1}\n{sentence2}\n\n{options_}",
         "{answer}"),
        ("Do these mean the same?\n{sentence1}\n{sentence2}\n\n{options_}",
         "{answer}"),
        ("Please check if these have the same meaning. {options_}"
         "\n{sentence1}\n{sentence2}", "{answer}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "synth_cot_paws_wiki": [
        ("{sentence1}\n{sentence2}\n\nExplain your answer, do "
         "these sentences mean the same thing?\n{options_}\n"
         "Step-by-step reasoning process:", "{cot} So the answer is {answer}"),
        ("{sentence1}\n{sentence2}\n\nAre these two sentences paraphrases of "
         "each other?\n{options_}\nLet's see.",
         "{cot} So the answer is {answer}"),
        ("1. {sentence1}\n2. {sentence2}\n\n"
         "Are these two sentences paraphrases of each "
         "other?\n{options_}...I think the logic is:",
         "{cot} The answer is {answer}"),
        ("(1) {sentence1}\n(2) {sentence2}\n\nDo these two sentences mean the "
         "same thing?\n\n{options_}\nAhh.", "{cot}. The answer: {answer}"),
        ("Sentence 1: {sentence1}\nSentence 2: {sentence2}\n\nDo these two "
         "sentences convey the same information?\n\n{options_}\nLet's think.",
         "{cot} The answer is {answer}"),
        ("Do these two sentences from wikipedia have the same "
         "meaning?\n{sentence1}\n{sentence2}\n\n{options_}\nThoughts:",
         "{cot}\nAnswer: {answer}"),
        ("Think before you answer: Same "
         "meaning?\n{sentence1}\n{sentence2}\n\n{options_}",
         "{cot}\nThe answer: {answer}"),
        ("Are these paraphrases?\n{sentence1}\n{sentence2}\n\n{options_}\nCoT:",
         "Answer: {answer}"),
        ("Let's carefully answer this question: do these mean the same?\n"
         "{sentence1}\n{sentence2}\n\n{options_}",
         "{cot}\nThe final answer: {answer}"),
        ("Please check if these have the same meaning.\n{options_}\n"
         "{sentence1}\n{sentence2}\nYour thought?",
         "{cot}\nThe answer is {answer}"),
    ],
    # "sentiment140": [
    #     ("{text}\nSelect your answer from the options. What is the sentiment "
    #      "of this tweet?\n{options_}...I think the answer is", "{answer}"),
    #     ("{text}\n\nHow would the sentiment of this tweet be "
    #      "described?\n{options_}", "{answer}"),
    #     ("{text}\n\nDescribe the sentiment embodied by this "
    #      "tweet.\n{options_}\nI think the answer is", "{answer}"),
    #     ("Tweet: {text}\nPredict the sentiment of this tweet.\n{options_}",
    #      "{answer}"),
    #     ("Multi-choice question: What is the sentiment of the following "
    #      "tweet?\nTweet: {text}\n{options_}", "{answer}"),
    #     ("Select your answer from the options. How would one describe the "
    #      "sentiment of this tweet?\n{text}\n{options_}", "{answer}"),
    #     ("Possible tweet sentiments: {options_}\nWrite a tweet that is "
    #      "{answer}.", "{text}"),
    #     ("What is an example of a tweet?", "{text}"),
    #     ("Write a {answer} tweet. Possible tweet types: {options_}", "{text}"),
    #     ("Sentiment possibilities {options_}. Generate a tweet that has the "
    #      "following sentiment: {answer} ", "{text}"),
    # ],
    # Not in FLAN Templates (flan_templates):
    # "synth_cot_sentiment140": [
    #     ("{text}\nWhat is the sentiment of this tweet?\n{options_}..."
    #      "I think the solution should be:", "{cot} The answer is {answer}"),
    #     ("{text}\n\nHow would the sentiment of this tweet be "
    #      "described?\n{options_}\nStep-by-step reasoning process:",
    #      "{cot} So the answer is {answer}"),
    #     ("{text}\n\nDescribe the sentiment embodied by this "
    #      "tweet.\n{options_}\nThoughts:", "{cot}\nAnswer: {answer}"),
    #     ("Tweet: {text}\nEXPLAIN the sentiment of this tweet.\n{options_}",
    #      "Explanation: {cot}\nAnswer: {answer}"),
    #     ("Think out loud: What is the sentiment of the following "
    #      "tweet?\nTweet:{text}\n{options_}\n", "{cot} The answer is {answer}"),
    #     ("Let's think step-by-step to solve this: How would one describe the "
    #      "sentiment of this tweet?\n{text}\n{options_}\n",
    #      "Step-by-step reasoning: {cot}\nAnswer: {answer}"),
    #     ("{text}\nSentiment?\n{options_}\nCoT:", "{cot}\nAnswer: {answer}"),
    #     ("{text}\nHow is sentiment of the text above?\n{options_}\n"
    #      "Chain-of-thought:", "{cot}\nAnswer: {answer}"),
    #     ("{text}\nIs this text positive or negative?\n{options_}\n"
    #      "Well, I think:", "{cot}\nSo the answer is: {answer}"),
    #     ("Text: {text}\nIs the text above positive or negative in terms of "
    #      "sentiment?\n{options_}\nHmm...", "{cot}\nThe answer is: {answer}"),
    # ],
    "story_cloze": [
        ("{context}\n{options_}\nWhich option is the next sentence?",
         "{answer}"),
        ("{context}\n\nWhat is the next sentence?\n{options_}", "{answer}"),
        ("{context}\n\nWhat is a natural next sentence?\n{options_}",
         "{answer}"),
        ("{context}\n\nWrite the next sentence, by choosing from:\n{options_}",
         "{answer}"),
        ("Context: {context}\n\nNow do a next sentence "
         "writing task.\n{options_}", "{answer}"),
        ("Story: {context}\n\nIn the options below, what is the most likely to"
         " happen next?\n{options_}", "{answer}"),
        ("Write the next sentence in this story.\n\n{context}\n{options_}",
         "{answer}"),
        ("Choose from options. Continue the following "
         "story.\n\n{context}\n{options_}", "{answer}"),
        ("{options_}\nWrite a story that ends with: {answer}",
         "{context} {answer}"),
        ("Write a plausible story that ends with this sentence?\n\nLast "
         "sentence: {answer}\n{options_}", "{context} {answer}"),
    ],
    "copa": [
        ("{premise} What is the {question}?\n\n{options_}", "{answer}"),
        ("Here is a premise: {premise}\n\nWhat is the {question}?\n\n{options_}",
         "{answer}"),
        ("{premise}\n\nWhat is the {question} of the preceding "
         "sentence?\n\n{options_}", "{answer}"),
        ("{premise}\n\nWhat is a plausible {question}?\n\n{options_}",
         "{answer}"),
        ("Based on the following sentence, what is the "
         "{question}?\n\n{premise}\n\n{options_}", "{answer}"),
        ("{premise}\n\n{question}: \n\n{options_}", "{answer}"),
        ("What is the {question} of the following "
         "sentence?\n\n{premise}\n\n{options_}\nThe answer is:", "{answer}"),
        ("Answer the following question about this "
         "sentence:\n\n{premise}\n\nWhat is the {question}?\n\n{options_}",
         "{answer}"),
        ("Write a sentence.", "{premise}"),
        ("Premise: {premise}\nWhat is the {question}?\n{options_}", "{answer}"),
    ],
    "winogrande": [
        ("How does the sentence end? See options at the "
         "end\n\n{context}\n\n{options_}", "{answer}"),
        ("Write the next sentence.\n\n{context}\n\n{options_}\nAnswer:",
         "{answer}"),
        ("Choose your story that continues the following "
         "story.\n\n{context}\n\n{options_}", "{answer}"),
        ("{options_}\nComplete the following sentence.\n\n{context}\n\n",
         "{answer}"),
        ("Continue writing the following text.\n\n{context}\n\n{options_}",
         "{answer}"),
        ("How does the sentence end?\n\n{context}\n{options_}", "{answer}"),
        ("Write the next sentence.\n\n{context}\n{options_}", "{answer}"),
        ("Continue the following story.\n\n{context}\n{options_}", "{answer}"),
        ("Complete the following sentence.\n\n{context}\n{options_}",
         "{answer}"),
        ("Continue writing the following text.\n\n{context}\n{options_}",
         "{answer}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "synth_cot_winogrande": [
        ("How does the sentence end? Let's give some reasoning before you "
         "answer\n\n{context}\n\n{options_}\n", "{cot} The answer is {answer}"),
        ("Write the next sentence.\n\n{context}\n\n{options_}\n"
         "Chain-of-thought:", "{cot}\nThe answer is {answer}"),
        ("Choose your story that continues the following "
         "story.\n\n{context}\n\n{options_}\nYour thought first:",
         "Thoughts: {cot}\nThe answer is {answer}"),
        ("{options_}\nComplete the following sentence.\n\n{context}\n\nCoT:",
         "{cot}\nThe answer is {answer}"),
        ("Continue writing the following text.\n\n{context}\n\n{options_}\n"
         "Well...", "{cot} So the answer is {answer}"),
        ("How does the sentence end?\n{context}\n{options_}\n"
         "Let's reason step-by-step:", "{cot}... The answer is {answer}"),
        ("Write the next sentence.\n{options_}\n{context}\nStep-by-step "
         "reasoning process:", "{cot}\nThe answer is {answer}"),
        ("Continue the following story. Explain your choice first"
         "\n\n{context}\n{options_}", "{cot}\nThe answer is {answer}"),
        ("Complete the following sentence.\n\n{context}\nLet's think "
         "step-by-step {options_}", "{cot} The answer is {answer}"),
        ("Continue writing the following text. EXPLANATION first!\n{context} "
         "{options_}", "{cot} The answer is {answer}"),
    ],
    "yelp_polarity_reviews": [
        ("{text}\nIs this review positive or negative?\n{options_}\nAnswer:",
         "{answer}"),
        ("{text}\nChoose the sentiment of this review?\n{options_}",
         "{answer}"),
        ("{text}\nChoose: was this review given positively or "
         "negatively?\n{options_}", "{answer}"),
        ("{text}\nHow would this review be described in terms of "
         "sentiment?\n{options_}", "{answer}"),
        ("Choose your answer: is the following review positive or "
         "negative?\n\n{text}\n\n{options_}", "{answer}"),
        ("What is the sentiment of the following review?\n{text}\n{options_}",
         "{answer}"),
        ("How might one describe the sentiment of this review?\n{text}..."
         "{options_} I think the answer is", "{answer}"),
        ("Write a {answer} yelp review ({options_}).", "{text}"),
        ("Possible review types:\n{options_}.\nGenerate a {answer} review "
         "for a place", "{text}"),
        ("{options_} What would be an example of an {answer} review?",
         "{text}"),
    ],
    "arc": [
        ("{question}\n\n{options_}", "{answer}"),
        ("Question: {question}\n{options_}\nAnswer:", "{answer}"),
        ("Question: {question}\n\nWhat is the correct answer to the question "
         "from the following choices?\n{options_}", "{answer}"),
        ("Q: {question}\nWhat is the correct answer to this "
         "question?\n{options_}...A:", "{answer}"),
        ("Choose your answer?\n\n{question}\n\n{options_}", "{answer}"),
        ("Answer the question\n\n{question}\n{options_}", "{answer}"),
        ("{question}\n\nPick the answer from these options\n\n{options_}",
         "{answer}"),
        ("Write a question you would see in a school textbook.", "{question}"),
        ("What's an example of a grad-school level question?", "{question}"),
        ("I just took a test in school today. What question was I asked?",
         "{question}"),
    ],
    "anli": [
        ("{context}\n\nChoose your answer: based on the paragraph above can we"
         " conclude that \"{hypothesis}\"?\n\n{options_}\nI think the answer "
         "is", "{answer}"),
        ("{context}\n\nBased on that paragraph can we conclude that this "
         "sentence is true?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("{context}\n\nCan we draw the following "
         "conclusion?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("{context}\nDoes this next sentence follow, given the preceding "
         "text?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("{context}\nCan we infer the "
         "following?\n{hypothesis}\n\n{options_}\nThe answer is:", "{answer}"),
        ("Read the following paragraph and determine if the hypothesis is "
         "true:\n\n{context}\n\n{options_}\nHypothesis: {hypothesis}\n\n\n",
         "{answer}"),
        ("Read the text and determine if the sentence is true (see options at "
         "the end):\n\n{context}\n\nSentence: {hypothesis}\n{options_}",
         "{answer}"),
        ("Can we draw the following hypothesis from the context (see options)?"
         " \n\nContext:\n\n{context}\n\nHypothesis: {hypothesis}\n{options_}",
         "{answer}"),
        ("Choose from options: Determine if the sentence is true based on the "
         "text below:\n{hypothesis}\n\n{context}\n{options_}", "{answer}"),
        ("Generate a context and a hypothesis.",
         "Context: {context}\n\nHypothesis: {hypothesis}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "synth_cot_anli": [
        ("{premise}\n\nBased on the paragraph above can we"
         " conclude that \"{hypothesis}\"?\n\n{options_}\nI think the chain-of"
         "-thought is", "{cot}. The answer is {answer}"),
        ("{premise}\n\nBased on that paragraph can we conclude that this "
         "sentence is true?\n{hypothesis}\n\n{options_}\nLet's think step by "
         "step:", "{cot} The answer is {answer}"),
        ("{premise}\n\nCan we draw the following "
         "conclusion?\n{hypothesis}\n\n{options_}\nHmmm, let's see.",
         "{cot} The answer is {answer}"),
        ("{premise}\nDoes this next sentence follow, given the preceding "
         "text?\n{hypothesis}\n\n{options_}\nLet me think first.",
         "{cot} The answer is {answer}"),
        ("{premise}\nCan we infer the "
         "following?\n{hypothesis}\n\n{options_}\nI think:",
         "{cot} The answer is {answer}"),
        ("Read the following paragraph and determine if the hypothesis is "
         "true:\n\n{premise}\n\n{options_}\nHypothesis: {hypothesis}\n\nLet's "
         "think before answering.", "{cot} The answer is {answer}"),
        ("Read the text and determine if the sentence is true (let's think "
         "step by step first):\n\n{premise}\n\nSentence: "
         "{hypothesis}\n{options_}", "{cot} The answer is {answer}"),
        ("Think carefully before answering: can we draw the following "
         "hypothesis from the premise\nContext:\n\n{premise}\n\nHypothesis: "
         "{hypothesis}\n{options_}",
         "Let me think. {cot} The answer is {answer}"),
        ("Determine if the sentence is true based on the text below:\n"
         "{hypothesis}\n\n{premise}\n{options_}\n"
         "Step-by-step reasoning process:", "{cot} The answer is {answer}"),
        ("Generate a premise and a hypothesis, together with explanation",
         "Context: {premise}\nHypothesis: {hypothesis}\n{options_}\n"
         "Explanation: {cot} The answer is {answer}"),
    ],
    "coqa": [
        ("{text}\n\nAnswer the following "
         "questions:\n{numbered_questions}\n\nNumbered answers:",
         "{numbered_answers}"),
        ("Read the text and answer the "
         "questions.\n\n{text}\n\n{numbered_questions}\n\nNumbered answers:",
         "{numbered_answers}"),
        ("Answer the questions at the end based on the "
         "text.\n\n{text}\n\n{numbered_questions}\n\nNumbered answers:",
         "{numbered_answers}"),
        ("{text}\n\nAnswer this series of "
         "questions:\n\n{numbered_questions}\n\nNumbered answers:",
         "{numbered_answers}"),
        ("{text}\n\nWhat are the answers to this following set of "
         "questions:\n\n{numbered_questions}\n\nNumbered answers:",
         "{numbered_answers}"),
        ("{text}\n\nNow, provide a numbered list of answers to these "
         "questions:\n\n{numbered_questions}\n\nNumbered answers:",
         "{numbered_answers}"),
        ("{text}\n\n{numbered_questions}\n\nNumbered answers:",
         "{numbered_answers}"),
        ("{text}\n\n{numbered_questions}\n\nProvide a numbered list of "
         "answers.", "{numbered_answers}"),
        ("Make use of the article to answer the "
         "questions.\n\n{text}\n\n{numbered_questions}\n\nNumbered answers:",
         "{numbered_answers}"),
        ("{text}\n\nBased on the article and the following list of answers, "
         "write a list of questions.\n\n{numbered_answers}\n\nNumbered "
         "questions:", "{numbered_questions}"),
    ],
    # "opinion_abstracts_rotten_tomatoes": [
    #     ("{numbered_reviews}\n\nWrite a one sentence summary of the reviews "
    #      "above.", "{critic_consensus}"),
    #     ("{numbered_reviews}\n\nWhat is a brief summary of "
    #      "the following reviews?", "{critic_consensus}"),
    #     ("{numbered_reviews}\nBased on these individual reviews, what is the "
    #      "critic consensus?", "{critic_consensus}"),
    #     ("{numbered_reviews}\nWhat is the consensus?", "{critic_consensus}"),
    #     ("Here are some reviews for a movie: {numbered_reviews}\n\nWhat was "
    #      "the overall consensus about the movie?", "{critic_consensus}"),
    #     ("Summarize the following movie "
    #      "reviews:\n\n{numbered_reviews}\n\nSummary:", "{critic_consensus}"),
    #     ("Write a one sentence review of the movie \"{movie}\".",
    #      "{critic_consensus}"),
    #     ("Write an ordered list of reviews about \"{movie}\".",
    #      "{numbered_reviews}"),
    #     ("The critic consesnsus is: {critic_consensus}. What reviews supported"
    #      " this critic consensus?", "{numbered_reviews}"),
    #     ("Which movie is the following review "
    #      "about?\n\n{first_review}\n\nMovie:", "{movie}"),
    # ],
    "opinion_abstracts_idebate": [
        ("{argument_sentences}\n\nWhat is the general argument implied by "
         "these sentences?", "{claim}"),
        ("Sentences: {argument_sentences}\n\nWhat claim can be made from these"
         " sentences?", "{claim}"),
        ("{debate_name}\nWhat argument could one make about this debate topic?",
         "{claim}"),
        ("{debate_name}\nWhat is a possible side to this debate?", "{claim}"),
        ("What claim can be made from the following pieces of "
         "evidence?\n\n{argument_sentences}", "{claim}"),
        ("Summarize the argument implied by these "
         "sentences?\n\n{argument_sentences}", "{claim}"),
        ("What debate topic are the following sentences "
         "about?\n\n{argument_sentences}", "{debate_name}"),
        ("What is the debate topic for the following "
         "sentences?\n\n{argument_sentences}", "{debate_name}"),
        ("{claim}\nCome up with some evidence to support this claim.",
         "{argument_sentences}"),
        ("Claim: {claim}\nWhat evidence supports this claim?",
         "{argument_sentences}"),
    ],
    "common_gen": [
        ("Concepts: {concepts}\n\nWrite a sentence that includes all these "
         "words.", "{target}"),
        ("Keywords: {concepts}\n\nWhat is a sentence that includes all these "
         "keywords?", "{target}"),
        ("Here are some concepts: {concepts}\n\nWhat is a sentence about these"
         " concepts?", "{target}"),
        ("Produce a sentence which mentions all of these concepts: {concepts}",
         "{target}"),
        ("Write a sentence about the following things:\n\n{concepts}",
         "{target}"),
        ("Generate a sentence that includes all the following words: {concepts}",
         "{target}"),
        ("What are the keywords in the following sentence:\n\n{target}",
         "{concepts}"),
        ("What are the most important words in the following "
         "sentence:\n\n{target}", "{concepts}"),
        ("Identify the most salient words in this sentence:\n\n{target}",
         "{concepts_newline}"),
        ("Generate a sentence, and then tell me the concepts included in that "
         "sentence.", "Sentence:\n{target}\n\nConcepts:\n{concepts_newline}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "synth_cot_common_gen": [
        ("Concepts: {concepts}\n\nWrite a sentence that includes all these "
         "words. Chain-of-thought:", "{cot} The answer is {target}"),
        ("Keywords: {concepts}\n\nWhat is a sentence that includes all these "
         "keywords? Let see...", "{cot} The answer is {target}"),
        ("Here are some concepts: {concepts}\n\nWhat is a sentence about these"
         " concepts? Hm...", "{cot}\nThe answer is {target}"),
        ("Produce a sentence which mentions all of these concepts: {concepts} "
         "Let's reason first:", "{cot}\nThe answer is {target}"),
        ("Write a sentence about the following things:\n\n{concepts}\n"
         "Thoughts:", "{cot}\nThe answer is {target}"),
        ("Generate a sentence that includes all the following words (thinking "
         "out loud): {concepts}", "{cot}\nThe answer is {target}"),
        ("Let's give an explanable answer to this question: generate a "
         "sentence using words: {concepts}", "{cot} The answer is {target}"),
        ("Think step-by-step to answer this question: generate a "
         "sentence using concepts: {concepts}\n"
         "Step-by-step reasoning process:", "{cot} The answer is {target}"),
        ("Think step-by-step to answer this question: generate a "
         "sentence using concepts: {concepts_newline}",
         "\n{cot} The answer is {target}"),
        ("Answer this question: generate a sentence using concepts: "
         "{concepts_newline}. Think step-by-step:",
         "\n{cot}\nThe answer is {target}"),
    ],
    "dart": [
        ("Triple: {tripleset}\n\nWhat is a sentence that describes this triple?",
         "{target}"),
        ("Data: {tripleset}\n\nWhat would a sentence about this data be like?",
         "{target}"),
        ("Generate an approximately fifteen-word sentence that describes all "
         "this data: {tripleset}\n\n", "{target}"),
        ("Here is some data: {tripleset}.\n\nWrite a sentence that describes "
         "this data:", "{target}"),
        ("This is some data: {tripleset}.\n\nGenerate a detailed description "
         "of this data.", "{target}"),
        ("Generate a sentence about this data: {tripleset}\nSentence:",
         "{target}"),
        ("Write a sentence that about [{tripleset}].", "{target}"),
        ("Produce a long descriptive sentence that uses all these words: "
         "{tripleset}", "{target}"),
        ("What concepts are described in the following "
         "sentence?\n\n\"{target}\"\n\nReturn the answer as pairs of triples.",
         "{tripleset_newline}"),
        ("Create a set of triples that describes the content in the following "
         "sentence.\n\n{target}\n\n", "{tripleset_newline}"),
    ],
    # "e2e_nlg": [
    #     ("Attributes: {meaning_representation}. Produce a detailed sentence "
    #      "about this restaurant.", "{target}"),
    #     ("Data: {meaning_representation}. Can you generate a sentence about "
    #      "this data?", "{target}"),
    #     ("Data: {meaning_representation}. What is a sentence that describe "
    #      "this data?", "{target}"),
    #     ("Here are some keywords about a "
    #      "restaurant:\n\n{meaning_representation}. Write a sentence that "
    #      "describes the following attributes of a restaurant.", "{target}"),
    #     ("Here is some data about a restaurant: {meaning_representation}. "
    #      "Write a sentence that includes the above data about a restaurant",
    #      "{target}"),
    #     ("Sentence: {meaning_representation}\n\nCan you represent the content "
    #      "in this sentence in data form?", "{target}"),
    #     ("Write a sentence about a restaurant with all the following "
    #      "attributes: {meaning_representation}\nSentence:", "{target}"),
    #     ("Write a sentence that is about a restaurant with all the following "
    #      "properties: {meaning_representation}\nSentence:", "{target}"),
    #     ("Produce a detailed sentence about a restaurant using the following "
    #      "words: {meaning_representation}\nSentence:", "{target}"),
    #     ("Generate a descriptive sentence about a restaurant using the "
    #      "following words:\n\n{meaning_representation}\nSentence:", "{target}"),
    # ],
    "web_nlg_en": [
        ("{input_string}\n\nWhat is sentence that verbalizes this data?",
         "{target}"),
        ("Data: {input_string}\n\nSentence about the following data: ",
         "{target}"),
        ("Here is some data: {input_string}.\n\nWrite a sentence that "
         "describes this data.\nSentence:", "{target}"),
        ("This is some data: {input_string}.\n\nGenerate a detailed "
         "description of this data.\nSentence:", "{target}"),
        ("Generate a sentence about this data: {input_string}.\nSentence:",
         "{target}"),
        ("Generate a sentence that describes the following data: "
         "{input_string}.\nSentence:", "{target}"),
        ("Produce a long descriptive sentence that uses all these words: "
         "{input_string}.\nSentence:", "{target}"),
        ("Generate an approximately fifteen-word sentence that describes all "
         "this data: {input_string}.\nSentence:", "{target}"),
        ("Sentence: {target}\n\nWhat data can be extracted from this sentence?",
         "{input_string}"),
        ("Sentence: {target}\n\nWhat structured data could we extract from "
         "this sentence?", "{input_string}"),
    ],
    "wiki_lingua_english_en": [
        ("{source}\n\nSummary:", "{target}"),
        ("Summarize the following:\n{source}\n\nSummary:", "{target}"),
        ("Summarize this article:\n\n{source}\n\nSummary:", "{target}"),
        ("Summarize this article in one sentence.\n{source}\n\nSummary:",
         "{target}"),
        ("What is a one-sentence summary of the following "
         "article?\n{source}\n\nSummary:", "{target}"),
        ("In one sentence, describe what the following article is "
         "about:\n\n{source}\n\nSummary:", "{target}"),
        ("Article: {source}\n\nWhat is a summary?", "{target}"),
        ("Article: {source}\nWhat is a summary of what this article is about?",
         "{target}"),
        ("Write an article based on this summary:\n\n{target}\n\nArticle:",
         "{source}"),
        ("Write an article based on this \"{target}\"\n\nArticle:", "{source}"),
    ],
    "multirc": [
        ("{paragraph}\n\nQuestion: \"{question}\"\n\nResponse: "
         "\"{response}\"\n{options_}\nDoes the response correctly answer the "
         "question?\n\n", "{answer}"),
        ("{paragraph}\n\nQuestion: \"{question}\"\n\nResponse: "
         "\"{response}\"\n\nBased on the paragraph, is the response to the "
         "question is factually correct?\n\n{options_}", "{answer}"),
        ("{paragraph}\n\nQuestion: \"{question}\"\n\nAnswer: "
         "\"{response}\"\n\nIs this answer correct?\n\n{options_}...I think "
         "the answer is", "{answer}"),
        ("Paragraph: {paragraph}\n\nQuestion: \"{question}\"\n\nAnswer: "
         "\"{response}\"\n\nBased on the paragraph, choose if the answer is "
         "correct:\n\n{options_}", "{answer}"),
        ("{paragraph}\n\nChoose from options: Based on the paragraph, does the"
         " response \"{response}\" correctly answer the question "
         "\"{question}\"?\n\n{options_}", "{answer}"),
        ("{paragraph}\n\nChoose your answer: According to the above paragraph,"
         " the correct answer to the question \"{question}\" is "
         "\"{response}\"?\n\n{options_}", "{answer}"),
        ("{paragraph}\n\nAfter reading the above, is \"{response}\" the "
         "correct answer to the question \"{question}\"?\n\n{options_}",
         "{answer}"),
        ("{paragraph}\n\nQuestion: \"{question}\"\n\nAnswer: "
         "\"{response}\"\n\nIs this answer to the question correct?"
         "\n{options_}", "{answer}"),
        ("{paragraph}\nDo you have any questions?", "{question}"),
        ("{paragraph}\nWhat question would one ask from this paragraph?",
         "{question}"),
    ],
    "cb": [
        ("{premise}\n\nBased on the paragraph above can we conclude that "
         "\"{hypothesis}\"?\n\n{options_}", "{answer}"),
        ("{premise}\n\nBased on that paragraph can we conclude that this "
         "sentence is true?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("{premise}\n\nCan we draw the following conclusion (choose your "
         "answer)?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("{premise}\nSelect from options. Does this next sentence follow, "
         "given the preceding text?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("{premise}\nMulti-choice question: Can we infer the "
         "following?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("Multi-choice problem: Read the following paragraph and determine if "
         "the hypothesis is true:\n\n{premise}\n\nHypothesis: "
         "{hypothesis}\n{options_}", "{answer}"),
        ("You will be given options, read the text and determine if the "
         "sentence is true:\n\n{premise}\n\nSentence: "
         "{hypothesis}\n{options_}", "{answer}"),
        ("Can we draw the following hypothesis from the context? "
         "\n\nContext:\n\n{premise}\n\nHypothesis: {hypothesis}\n{options_}",
         "{answer}"),
        ("Determine if the sentence is true based on the text "
         "below:\n{hypothesis}\n{options_}\n{premise}\n", "{answer}"),
        ("Generate a context and a hypothesis.",
         "Context: {premise}\n\nHypothesis: {hypothesis}"),
    ],
    "cola": [
        ("Sentence: \"{sentence}\"\nPick from options: would a linguist rate "
         "this sentence to be acceptable linguistically?\n\n{options_}...I "
         "think the answer is", "{answer}"),
        ("{sentence}\n\nHow would you consider the linguistic integrity of the"
         " preceding sentence?\n{options_}", "{answer}"),
        ("Test sentence: \"{sentence}\"\nIs this test sentence a correct "
         "grammatical English sentence?\n\n{options_}", "{answer}"),
        ("Sentence: \"{sentence}\"\nWould a linguist rate this sentence to be "
         "acceptable linguistically?\n\n{options_}", "{answer}"),
        ("Choose from options, is the following sentence linguistically "
         "acceptable?\n{sentence}\n{options_}", "{answer}"),
        ("Choose from the possible answers, would the following sentence, by "
         "the strictest standards, be considered correct by a "
         "linguist?\n\n{sentence}\n{options_}", "{answer}"),
        ("Multi-choice problem: Is the next sentence syntactically and "
         "semantically acceptable?\n\n{sentence}\n{options_}", "{answer}"),
        ("Would a linguist find the following sentence to be a valid English "
         "sentence grammatically?\n\n{sentence}\n{options_}", "{answer}"),
        ("Generate short a sentence that can be linguistically classified as "
         "{answer} ({options_})", "{sentence}"),
        ("Produce a brief English sentence that would be considered "
         "grammatically as category: {answer}\nAll categories: {options_}",
         "{sentence}"),
    ],
    # "sst2": [
    #     ("Review:\n{sentence}\nIs this movie review sentence negative or "
    #      "positive?\n{options_}\nThe answer is:", "{answer}"),
    #     ("{options_}\nShort movie review: {sentence}\nDid the critic thinking "
    #      "positively or negatively of the movie?\n\n", "{answer}"),
    #     ("Sentence from a movie review: {sentence}\nSelect your answer: was "
    #      "the movie seen positively or negatively based on the preceding "
    #      "review?\n\n{options_}", "{answer}"),
    #     ("\"{sentence}\"\nHow would the sentiment of this sentence be "
    #      "perceived --\n\n{options_}\nAnswer:", "{answer}"),
    #     ("Is the sentiment of the following sentence positive or negative (see"
    #      " options at the end)?\n{sentence}\n{options_}", "{answer}"),
    #     ("What is the sentiment of the following movie (choose your answer "
    #      "from the options) review sentence?\n{sentence}\n{options_}\nThe "
    #      "answer is:", "{answer}"),
    #     ("{options_}Would the following phrase be considered positive or "
    #      "negative?\n\n{sentence}\n", "{answer}"),
    #     ("Does the following review have a positive or negative opinion of the"
    #      " movie?\n\n{sentence}\n{options_}", "{answer}"),
    #     ("Write a \"{answer}\" movie review ({options_}).", "{sentence}"),
    #     ("Generate a short movie review that has \"{answer}\" sentiment "
    #      "({options_}).", "{sentence}"),
    # ],
    "mnli": [
        ("Premise: {premise}\n\nHypothesis: {hypothesis}\n\nDoes the premise "
         "entail the hypothesis?\n\n{options_}", "{answer}"),
        ("Premise: {premise}\nHypothesis: {hypothesis}\nIs the hypothesis "
         "entailed by the premise?\n{options_} And the answer is:", "{answer}"),
        ("Here is a premise:\n{premise}\n\nHere is a "
         "hypothesis:\n{hypothesis}\n\nHere are the options: {options_}\nIs it"
         " possible to conclude that if the premise is true, then so is the "
         "hypothesis?\n", "{answer}"),
        ("Sentence 1: {premise}\n\nSentence 2: {hypothesis}\n{options_}\nIs "
         "this second sentence entailed by the first sentence?\n\n",
         "{answer}"),
        ("See the multi-choice question below:\n\nSentence 1: "
         "{premise}\n\nSentence 2: {hypothesis}\n\nIf the first sentence is "
         "true, then is the second sentence true?\n{options_}", "{answer}"),
        ("Based on the premise \"{premise}\", can we conclude the hypothesis "
         "\"{hypothesis}\" is true (see options)?\n\n{options_}", "{answer}"),
        ("Choose your answer from options. Premise: \"{premise}\" If this "
         "premise is true, what does that tell us about whether it entails the"
         " hypothesis \"{hypothesis}\"?\n\n{options_}", "{answer}"),
        ("Premise:\n\"{premise}\" Based on this premise, is the hypothesis "
         "\"{hypothesis}\" true?\n{options_}", "{answer}"),
        ("If {premise}, can we conclude that \"{hypothesis}\"?\n{options_}",
         "{answer}"),
        ("{premise}\n\nDoes it follow that \"{hypothesis}\"?\n{options_}",
         "{answer}"),
    ],
    "qnli": [
        ("Does the sentence \"{sentence}\" answer the question "
         "\"{question}\"\n\n{options_}", "{answer}"),
        ("Single/multi-select question: Does the sentence \"{sentence}\" "
         "provide a valid answer to the question \"{question}\"\n{options_}",
         "{answer}"),
        ("Choose your answer: Is \"{sentence}\" a good answer to the question "
         "\"{question}\"\n{options_}", "{answer}"),
        ("{options_}\nDoes \"{sentence}\" correctly answer the question of "
         "{question}\n\n", "{answer}"),
        ("Choose your reply from the options at the end. Does \"{sentence}\" "
         "contain the correct answer to \"{question}\"\n{options_}",
         "{answer}"),
        ("Q: {question}\n A: {sentence}\n Does the answer correctly answer the"
         " question\n\n{options_}", "{answer}"),
        ("Question: {question}\nAnswer: {sentence}\n A single-select problem: "
         "Is the question answered in a satisfactory fashion?\n\n{options_}",
         "{answer}"),
        ("Question: {question}\n\nIs {sentence} a good answer to this "
         "question?\n\n{options_}", "{answer}"),
        ("Question: {question}\n\nIs \"{sentence}\" the correct answer?\n"
         "{options_}", "{answer}"),
        ("Can you generate a question with a factual answer?", "{question}"),
    ],
    "wnli": [
        ("If \"{sentence1}\", can we conclude that "
         "\"{sentence2}\"\n{options_}\nI think the answer is", "{answer}"),
        ("If \"{sentence1}\", does it follow that \"{sentence2}\"\n{options_}",
         "{answer}"),
        ("If \"{sentence1}\", is \"{sentence2}\" "
         "correct?\n\n{options_}\nAnswer:", "{answer}"),
        ("Multi-select: Let's say that \"{sentence1}\"\n\nCan we now say that "
         "\"{sentence2}\"?\n\n{options_}", "{answer}"),
        ("\"{sentence1}\" is a true sentence.\n\nDoes this mean that "
         "\"{sentence2}\"?\n\n{options_}", "{answer}"),
        ("Does \"{sentence2}\" appear to be an accurate statement based on "
         "\"{sentence1}\"?\n\n{options_}", "{answer}"),
        ("Can we conclude that \"{sentence2}\" if the statement "
         "\"{sentence1}\" is true?\n\n{options_}", "{answer}"),
        ("Multi-select: Is it possible to draw the conclusion that "
         "\"{sentence2}\" if \"{sentence1}\"?\n\n{options_}", "{answer}"),
        ("Multi-choice problem: Is \"{sentence2}\" true if "
         "\"{sentence1}\"?\n\n{options_}", "{answer}"),
        ("Sentence 1: \"{sentence1}\"\n\n Sentence 2: \"{sentence2}\"\n\nIs "
         "sentence 2 true, based on sentence 1?\n{options_}", "{answer}"),
    ],
    "snli": [
        ("If \"{premise}\", does this mean that \"{hypothesis}\"?\n\n{options_}",
         "{answer}"),
        ("Single/multi-select question: If \"{premise}\", can we conclude "
         "\"{hypothesis}\"?\n\n{options_}", "{answer}"),
        ("Choose your answer: If \"{premise}\", does it logically follow that "
         "\"{hypothesis}\"?\n\n{options_}", "{answer}"),
        ("Multi-choice problem: Based on the sentence \"{premise}\", is the "
         "sentence \"{hypothesis}\" a true sentence?\n\n{options_}",
         "{answer}"),
        ("Premise: {premise}\n\nHypothesis: {hypothesis}\n\n.Multi-select "
         "problem: Can we conclude that the hypothesis is true if the premise "
         "is true?\n\n{options_}", "{answer}"),
        ("Premise: {premise}\n\nHypothesis: {hypothesis}\n\n.Choose the "
         "correct answer: Given the premise, can we conclude the "
         "hypothesis?\n\n{options_}", "{answer}"),
        ("Here is a premise: \"{premise}\"\n\nHere is a hypothesis: "
         "\"{hypothesis}\"\n\n.Does the premise tell us whether the hypothesis"
         " is true?\n\n{options_}", "{answer}"),
        ("Single/multi-select question: Is it possible to conclude that "
         "\"{premise}\" if \"{hypothesis}\"?\n\n{options_}...I think the "
         "answer is", "{answer}"),
        ("Is the premise \"{premise}\" true if \"{hypothesis}\"?\n{options_}",
         "{answer}"),
        ("Write a brief sentence.", "{hypothesis}"),
    ],
    "trec": [
        ("What type of thing is the question \"{text}\" asking "
         "about?\n\n{options_}\nAnswer:", "{answer}"),
        ("Is the question \"{text}\" asking about an entity, an abbreviation, "
         "a description, a human, a location, or a numeric "
         "entity?\n\n{options_}", "{answer}"),
        ("{options_}Would the answer to the question \"{text}\" be an entity, "
         "an abbreviation, a description, a human, a location, or a numeric "
         "value?\n\n", "{answer}"),
        ("This is a question with answer options. What kind of thing would the"
         " answer to the question \"{text}\" be an entity, an abbreviation, a "
         "description, a human, a location, or a numeric value?\n\n{options_}",
         "{answer}"),
        ("Choose your answer: What is \"{text}\" asking "
         "about?\n\n{options_}\nAnswer:", "{answer}"),
        ("From the following options, what is the question \"{text}\" asking "
         "about?\n\n{options_}", "{answer}"),
        ("{text}\n\nWhat kind of thing would answer this "
         "question?\n\n{options_}", "{answer}"),
        ("Here is a single or multi-choice question: {text}\n\nWould the "
         "answer to this question be an entity, an abbreviation, a "
         "description, a human, a location, or a numeric value?\n\n{options_}",
         "{answer}"),
        ("Q: {text}\n\nWhich one of the following options would the answer to "
         "this be?\n\n{options_}\n\nA:", "{answer}"),
        ("Please ask me a question.", "{text}"),
    ],
    "stsb": [
        ("{sentence1}\n{sentence2}\n\nRate the textual similarity of these two"
         " sentences on a scale from 0 to 5, where 0 is \"no meaning overlap\""
         " and 5 is \"means the same thing\".\n\n{options_}", "{answer}"),
        ("{sentence1}\n{sentence2}\n\nOn a scale from 0 to 5, where 0 is \"no "
         "meaning overlap\" and 5 is \"means the same thing\", how closely "
         "does the first sentence resemble the second one?\n\n{options_}",
         "{answer}"),
        ("Sentence 1: {sentence1}\n\n Sentence 2: {sentence2}\n\nFrom 0 to 5 "
         "(0=\"no meaning overlap\" and 5=\"means the same thing\"), how "
         "similar are the two sentences?\n\n{options_}", "{answer}"),
        ("Select from options: How similar are the following two "
         "sentences?\n\n{sentence1}\n{sentence2}\n\nGive the answer on a scale"
         " from 0 - 5, where 0 is \"not similar at all\" and 5 is \"means the "
         "same thing\".\n\n{options_}", "{answer}"),
        ("Single/multi-select question: Do the following sentences say the "
         "same thing?\n\n{sentence1}\n{sentence2}\n\nReturn your answer on a "
         "scale from 0 to 5, where 0 is \"not similar\" and 5 is \"very "
         "similar\".\n\n{options_}", "{answer}"),
        ("Rate the similarity of the following two sentences on a scale from 0"
         " to 5, where 0 is \"no meaning overlap\" and 5 is \"means the same "
         "thing\"?\n\n{sentence1}\n{sentence2}\n\n{options_}", "{answer}"),
        ("On a scale from 0-5, where 0 is \"not similar\" and 5 is \"very "
         "similar\", how similar is the sentence \"{sentence1}\" to the "
         "sentence \"{sentence2}\"?\n\n{options_}", "{answer}"),
        ("How similar are these two sentences, on a scale from 0-5 (0 is \"not"
         " similar\" and 5 is \"very "
         "similar\")?\n\n{sentence1}\n{sentence2}\n\n{options_}", "{answer}"),
        ("{sentence1}\n\nGenerate a new sentence that is, on a scale from 0 to"
         " 5, a {answer} in textual similarity to the above sentence.",
         "{sentence2}"),
        ("{sentence2}\n\nWhat is a sentence that would be (on a scale from 0 "
         "to 5) a {answer} out of 5 in terms of textual similarity to the "
         "above sentence?", "{sentence1}"),
    ],
    "hellaswag": [
        ("What happens next in this paragraph?\n\n{context}\n{options_}",
         "{answer}"),
        ("Multi-choice problem: Continue writing the next sentence in this "
         "paragraph:\n\n{context}\n\n{options_}", "{answer}"),
        ("Select from options: Continue writing the next "
         "sentence.\n\n{context}\n\n{options_}\nAnswer:", "{answer}"),
        ("This is a test of commonsense with single/multi-choices. Complete "
         "the next sentence:\n\n{context}\n\n{options_}\nThe answer is:",
         "{answer}"),
        ("Write the next sentence in this paragraph:\n\n{context}\n\n{options_}",
         "{answer}"),
        ("Multi-select problem: How does the next paragraph "
         "end?\n\n{context}\n\n{options_}", "{answer}"),
        ("{options_}Choose from options above and answer: What most naturally "
         "follows?\n\n{context}\nAnswer:", "{answer}"),
        ("What happens next?\n\n{context}\n\n{options_}", "{answer}"),
        ("What is the most logical next event?\n\n{context}\n\n{options_}",
         "{answer}"),
        ("Write the next sentence in the following "
         "story.\n\n{context}\n\n{options_}. The answer should be", "{answer}"),
    ],
    "piqa": [
        ("Here is a goal: {goal}\n\nHow would you accomplish this "
         "goal?\n\n{options_}", "{answer}"),
        ("Here is a goal: {goal}\n\nWhich way makes more sense to accomplish "
         "this goal?\n\n{options_}", "{answer}"),
        ("This is a question with answer options. Goal: {goal}\n\nWhich of the"
         " following methods is more reasonable for accomplishing this "
         "goal?\n\n{options_}...I think the answer is", "{answer}"),
        ("Objective: {goal}\n\nWhich of the following solutions is more sound "
         "in terms of naive physics reasoning?\n\n{options_}", "{answer}"),
        ("Multi-choice problem: Choose from the options at the end, and answer"
         " how do you do this: {goal}\n\n{options_}", "{answer}"),
        ("What is the best way to: {goal}\n\n{options_}\nAnswer:", "{answer}"),
        ("Single/multi-choice problem: Which of the following solutions is "
         "better for the following goal:\n{goal}\n\n{options_}", "{answer}"),
        ("This question has options. How would someone go about accomplishing "
         "this goal?\n{goal}\n\n{options_}", "{answer}"),
        ("What's an example of a task that requires knowledge of physical "
         "objects to perform?", "{goal}"),
        ("What kind of task would test someone's ability to perform physical "
         "reasoning?", "{goal}"),
    ],
    "openbookqa": [
        ("{fact}\n{question}\n\n{options_}", "{answer}"),
        ("This question has options. Select from options: Read this fact: "
         "\"{fact}\"\n\nNow answer this question: \"{question}\"\n\n{options_}",
         "{answer}"),
        ("Given the fact \"{fact}\", what is the answer to the question or "
         "completion \"{question}\"\n\n{options_}", "{answer}"),
        ("Multi-select: Knowing that \"{fact}\", how would one answer "
         "\"{question}\"\n\n{options_}...A:", "{answer}"),
        ("Use evidence from the fact that {fact} to answer the following "
         "question. Choose from options. \"{question}\"\n\n{options_}",
         "{answer}"),
        ("Fact: {fact}\nQuestion: {question}\n\nWhat's the answer? {options_}",
         "{answer}"),
        ("Use this fact to answer the question: {fact}\n\n{question}\n\n"
         "{options_}\n\nThe answer is:", "{answer}"),
        ("What sentence would provide a factual answer to this question: "
         "\"{question}\"", "{fact}"),
        ("What is a random fact?", "{fact}"),
        ("Generate a sentence that contains a fact.", "{fact}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "lambada": [
        ("{sentence}", "{answer}"),
        ("Complete the following text: {sentence}", "{answer}"),
        ("\"{sentence} _ ...\" What is the word in the blank space (_)? The "
         "answer is", "{answer}"),
        ("You will be given a text below. Complete the text.\n{sentence}",
         "{answer}"),
        ("TEXT: {sentence}", "{answer}"),
        ("SENTENCE: {sentence}", "{answer}"),
        ("Complete: {sentence}", "{answer}"),
        ("Text complete: {sentence}", "{answer}"),
        ("Complete text: {sentence}", "{answer}"),
        ("Continue writing the following text: {sentence}", "{answer}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "cot_gsm8k": [
        ("{question} Let's think first. Chain of thought:",
         "{chain_of_thought}\nTherefore, the answer is {answer}."),
        ("{question} Think carefully first, then make a decision:",
         "{chain_of_thought} So, the answer is {answer}."),
        ("{question} Let's be accurate as possible.",
         "{chain_of_thought}\nThe answer: {answer}."),
        ("{question} Give me reasons, before answering the question",
         "{chain_of_thought} So the final answer is {answer}."),
        ("Lizzy: {question}.\nMe: Hmmm, let me think. I think this is the "
         "detailed solution:", "{chain_of_thought} Final answer: {answer}."),
        ("Question: {question} Think carefully first, then make a decision:",
         "{chain_of_thought} So the answer is {answer}."),
        ("Give the step-by-step reasoning process and then the final answer. "
         "{question}", "{chain_of_thought}\nThe final answer: {answer}."),
        ("{question}\nThoughts? Step-by-step reasoning:",
         "{chain_of_thought}\nThus, the answer is {answer}."),
        ("My question is: {question} Your thoughts:",
         "{chain_of_thought} The final answer: {answer}."),
        ("{question} Let's answer step by step:",
         "{chain_of_thought} The answer: {answer}."),
    ],
    # Not in FLAN Templates (flan_templates):
    "cot_strategyqa": [
        ("{question}\nThink slowly and carefully, before giving your answer.",
         "{chain_of_thought}\nSo, the answer is {answer}."),
        ("{question} Please answer step by step:",
         "{chain_of_thought}\nSo, the final answer is {answer}."),
        ("{question}\nChain of thought:",
         "{chain_of_thought} The answer is {answer}."),
        ("Answer the following question by reasoning step-by-step. {question}",
         "{chain_of_thought} Therefore, the final answer is {answer}."),
        ("{question} Given the above question, please answer with reasoning "
         "first!", "{chain_of_thought}\nTherefore, the answer is {answer}."),
        ("{question} Think carefully first, then make a decision:",
         "{chain_of_thought} So, the answer is {answer}."),
        ("Q: {question} Now, let's think step by step:",
         "{chain_of_thought}\nThe answer: {answer}."),
        ("Answer the following question, but give the rationale first. "
         "{question}", "{chain_of_thought} So the final answer is {answer}."),
        ("{question} Hmmm, my chain of thoughts:",
         "{chain_of_thought} Final answer: {answer}."),
        ("Let's answer this question slowly: {question}\n",
         "{chain_of_thought} So the answer is {answer}."),
    ],
    # Not in FLAN Templates (flan_templates):
    "cot_creak": [
        ("Given the following question, let's solve step-by-step. {question}\n",
         "{chain_of_thought}\nThe final answer: {answer}."),
        ("My question: {question}\nPlease think gradually:",
         "{chain_of_thought}\nThus, the answer is {answer}."),
        ("Give the rationale and then the answer. {question}",
         "{chain_of_thought} The final answer: {answer}."),
        ("Q: {question}\nChain-of-thought:",
         "{chain_of_thought} The answer: {answer}."),
        ("{question}\nChain of thought and solution for this question is:",
         "{chain_of_thought}\nSo, the answer is {answer}."),
        ("Question: {question} Let's think first. Step-by-step reasoning:",
         "{chain_of_thought}\nSo, the final answer is {answer}."),
        ("{question}\nYour chain-of-thought:",
         "{chain_of_thought} The answer is {answer}."),
        ("{question} Step-by-step reasoning process:",
         "{chain_of_thought} Therefore, the final answer is {answer}."),
        ("{question} The thought process:",
         "{chain_of_thought}\nTherefore, the answer is {answer}."),
        ("{question} Let's think first. Step-by-step reasoning process:",
         "{chain_of_thought} So, the answer is {answer}."),
    ],
    # Not in FLAN Templates (flan_templates):
    "cot_qasc": [
        ("Question: {question}\nLet's be accurate as possible and think "
         "step-by-step.", "{chain_of_thought}\nThe answer: {answer}."),
        ("{question} Let's solve this problem gradually.\n",
         "{chain_of_thought} So the final answer is {answer}."),
        ("Question to you: {question}.\nLet's reason step-by-step:",
         "{chain_of_thought} Final answer: {answer}."),
        ("{question} Think carefully first, then make a decision. My thoughts:",
         "{chain_of_thought} So the answer is {answer}."),
        ("{question} Let's be accurate as possible.",
         "{chain_of_thought}\nThe final answer: {answer}."),
        ("Q: {question}\nLet's think step by step below.\n",
         "{chain_of_thought}\nThus, the answer is {answer}."),
        ("Let's think step by step! {question}\nThe thinking starts now:",
         "{chain_of_thought} The final answer: {answer}."),
        ("{question}\nHmmm, let me think. I don't want to be wrong, so I got "
         "to be careful.", "{chain_of_thought} The answer: {answer}."),
        ("Use reasoning to answer the following question. {question}",
         "{chain_of_thought}\nSo, the answer is {answer}."),
        ("{question} OK. Let's think hard:",
         "{chain_of_thought}\nSo, the final answer is {answer}."),
    ],
    # Not in FLAN Templates (flan_templates):
    "cot_esnli": [
        ("{question}\nLet's solve step-by-step:",
         "{chain_of_thought} The answer is {answer}."),
        ("{question} Step by step answer:",
         "{chain_of_thought} Therefore, the final answer is {answer}."),
        ("{question} Stream of thoughts:",
         "{chain_of_thought}\nTherefore, the answer is {answer}."),
        ("{question} Now, let's be accurate as possible. Some thinking first:",
         "{chain_of_thought} So, the answer is {answer}."),
        ("Denny asked: {question}.\nLe: OK, so how can I answer with some "
         "explanation?\n", "{chain_of_thought}\nThe answer: {answer}."),
        ("Student: {question}.\nTeacher: Let's think:\n",
         "{chain_of_thought} So the final answer is {answer}."),
        ("{question} Let's be accurate as possible and think first.",
         "{chain_of_thought} Final answer: {answer}."),
        ("Please answer the following question by reasoning step-by-step. "
         "{question}. Step-by-step reasoning:",
         "{chain_of_thought} So the answer is {answer}."),
        ("{question} A step-by-step solution is:\n",
         "{chain_of_thought}\nThe final answer: {answer}."),
        ("Leo: {question}\nMei: OK, So, let's think first...\nMe:",
         "{chain_of_thought}\nThus, the answer is {answer}."),
    ],
    # Not in FLAN Templates (flan_templates):
    "cot_ecqa": [
        ("{question}\nPlease answer and provide answer explanation.",
         "{chain_of_thought} The final answer: {answer}."),
        ("{question}\nStep-by-step reasoning process below:\n",
         "{chain_of_thought} The answer: {answer}."),
        ("{question} Hmmm, let me think.",
         "{chain_of_thought}\nSo, the answer is {answer}."),
        ("{question}\nLet's think now! Step-by-step reasoning:",
         "{chain_of_thought}\nSo, the final answer is {answer}."),
        ("next question: {question}\nreasoning:",
         "{chain_of_thought} The answer is {answer}."),
        ("Use reasoning to lead to the answer of the following question:\n"
         "{question}\n Reasoning process:",
         "{chain_of_thought} Therefore, the final answer is {answer}."),
        ("{question} Let's give stream of consciousness first:",
         "{chain_of_thought}\nTherefore, the answer is {answer}."),
        ("{question} Let's think step by step:",
         "{chain_of_thought} So, the answer is {answer}."),
        ("I'll give you a question, please answer with step-by-step reasoning "
         "process. {question}\n", "{chain_of_thought}\nThe answer: {answer}."),
        ("{question}\nLet's think carefully first. Step-by-step reasoning "
         "process:", "{chain_of_thought} So the final answer is {answer}."),
    ],
    # Not in FLAN Templates (flan_templates):
    "cot_sensemaking": [
        ("{question} Let's reason step by step:",
         "{chain_of_thought} Final answer: {answer}."),
        ("Question: {question}\nPlease answer this question gradually:",
         "{chain_of_thought} So the answer is {answer}."),
        ("See question below:\n{question}\nReason slowly and give your answer.",
         "{chain_of_thought}\nThe final answer: {answer}."),
        ("OK. You'll be given the following question. Please do "
         "chain-of-thought reasoning.\n{question}",
         "{chain_of_thought}\nThus, the answer is {answer}."),
        ("{question} Let's be accurate as possible. So think first.",
         "{chain_of_thought}\nThe final answer: {answer}."),
        ("Q: {question}\nLet's solve this gradually.\n",
         "{chain_of_thought} The answer is {answer}."),
        ("Let's think step by step! {question}\n",
         "{chain_of_thought} The answer: {answer}."),
        ("{question}\nHmmm, let me think. I want to lay out the solution "
         "in details.", "{chain_of_thought} The answer is {answer}."),
        ("Answer the following question, with explanation first. {question}",
         "{chain_of_thought}\nSo, the answer is {answer}."),
        ("{question} Let me think hard. Detailed solution:",
         "{chain_of_thought}\nThe answer is {answer}."),
    ],
    # Not in FLAN Templates (flan_templates):
    "stream_aqua": [
        ("Q: {question} Let's give some random thoughts before answering.",
         "{chain_of_thought}\nTherefore, the answer is {answer}."),
        ("{question} Hmmm, my stream of consciousness:",
         "{chain_of_thought} So, the answer is {answer}."),
        ("Give a quick stream of consciousness before answering the following "
         "question. {question}", "{chain_of_thought}\nThe answer: {answer}."),
        ("Use some thinking to answer the following question. {question}",
         "{chain_of_thought} So the final answer is {answer}."),
        ("Student: {question}.\nAnother student: Let's say, hmmm...\n",
         "{chain_of_thought} Final answer: {answer}."),
        ("{question} Think first, then make a decision. Some random thoughts:",
         "{chain_of_thought} So the answer is {answer}."),
        ("{question} Now, let's think a bit. Some random thoughts:",
         "{chain_of_thought}\nThe final answer: {answer}."),
        ("{question} Stream of consciousness:",
         "{chain_of_thought}\nThus, the answer is {answer}."),
        ("Question: {question} Random thoughts:",
         "{chain_of_thought} The final answer: {answer}."),
        ("{question} OK. Let's think. Some random thoughts first:",
         "{chain_of_thought} The answer: {answer}."),
        ("Give stream of consciousness and then the final answer. {question}",
         "{chain_of_thought}\nThe final answer: {answer}."),
        ("{question} Stream of consciousness first, then make a decision:",
         "{chain_of_thought}\nThus, the answer is {answer}."),
        ("Question: {question} Let's think first. Some random reasoning:",
         "{chain_of_thought} The final answer: {answer}."),
        ("Some question: {question}\nSome stream of consciousness:",
         "{chain_of_thought} The answer: {answer}."),
        ("{question} Let's think first. Stream of consciousness:",
         "{chain_of_thought}\nSo, the answer is {answer}."),
    ],
    # Not in FLAN Templates (flan_templates):
    "stream_qed": [
        ("{question}\nSteam of consciousness below:\n",
         "{chain_of_thought}\nSo, the answer is {answer}."),
        ("{question} Let's give stream of consciousness first:",
         "{chain_of_thought}\nSo, the final answer is {answer}."),
        ("Quoc: {question}\nHW Chung: OK, some thoughts:",
         "{chain_of_thought} The answer is {answer}."),
        ("Q: {question} Let's give stream of consciousness first:",
         "{chain_of_thought} Therefore, the final answer is {answer}."),
        ("I got a question for you: {question}\nLet's think first:",
         "{chain_of_thought}\nTherefore, the answer is {answer}."),
        ("{question} Okie... think carefully first, then make a decision:",
         "{chain_of_thought} So, the answer is {answer}."),
        ("Output a stream of consciousness before answering the following. "
         "{question}", "{chain_of_thought}\nThe answer: {answer}."),
        ("{question} Let's think fast. Stream of consciousness:",
         "{chain_of_thought} So the final answer is {answer}."),
        ("Use stream of consciousness to answer the following. {question}",
         "{chain_of_thought} Final answer: {answer}."),
        ("Q: {question}\nLet's give stream of consciousness below\n",
         "{chain_of_thought} So the answer is {answer}."),
        ("Give a stream of consciousness and then the final answer. {question}",
         "{chain_of_thought}\nSo, the final answer is {answer}."),
        ("{question} OK. Let's think. My stream of consciousness:",
         "{chain_of_thought} The answer is {answer}."),
        ("Answer the following Q with stream of consciousness. {question}",
         "{chain_of_thought} Therefore, the final answer is {answer}."),
        ("Give some stream of consciousness and then the answer. {question}",
         "{chain_of_thought}\nTherefore, the answer is {answer}."),
        ("{question} Let's have some stream of consciousness first.",
         "{chain_of_thought} So, the answer is {answer}."),
    ],
    # Not in FLAN Templates (flan_templates):
    "strategyqa": [
        ("Yes or no: {question}", "{answer}"),
        ("{question} Answer yes or no.", "{answer}"),
        ("Question: {question} Answer:", "{answer}"),
        ("Answer yes or no after the question mark: {question}", "{answer}"),
        ("Answer yes or no: {question}", "{answer}"),
        ("Reply yes or no: {question}", "{answer}"),
        ("{question} Yes or no:", "{answer}"),
        ("{question}\n\nIt's yes or no? The answer is", "{answer}"),
        ("Yes/no: {question}", "{answer}"),
        ("{question}. The answer:", "{answer}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "unified_qa_science_inst": [
        ("{question}\n{options_}", "{answer}"),
        ("{question}\n{options_}The answer is:", "{answer}"),
        ("{question} {options_}\nYour answer:", "{answer}"),
        ("{question}\n{options_}\n", "{answer}"),
        ("{question}\n-\n{options_}", "{answer}"),
        ("{question}\n{options_}\n", "{answer}"),
        ("{question} {options_}\n", "{answer}"),
        ("Answer this:\n{question}\n{options_}", "{answer}"),
        ("{question}\n\n{options_}\n\n", "{answer}"),
        ("Answer this question: {question}\n{options_}. Answer:", "{answer}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "bigbench:simple_arithmetic_json.gen.blueridge_vocab.0_shot.30_examples": [
        ("What is the value of {inputs}? Answer:", "{targets}"),
        ("What is the solution of the following "
         "problem?\n{inputs}\n\nSolution:", "{targets}"),
        ("Reply with the result of this math problem:\n\n{inputs}",
         "{targets}"),
        ("{inputs} The answer is", "{targets}"),
        ("Solve this math problem: {inputs}\n\n", "{targets}"),
        ("{inputs}\n\n", "{targets}"),
        ("{inputs} A:", "{targets}"),
        ("Q: {inputs} A:", "{targets}"),
        ("Question: {inputs}\nAnswer:", "{targets}"),
        ("Math problem: {inputs}\nAnswer:", "{targets}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "bigbench:auto_debugging.gen.blueridge_vocab.0_shot.34_examples": [
        ("Answer the following question:\n{inputs}", "{targets}"),
        ("Given the question below, answer directly after the question "
         "ended:\n{inputs}", "{targets}"),
        ("{inputs} I think the answer is:", "{targets}"),
        ("{inputs} The answer is", "{targets}"),
        ("{inputs} The answer of this coding problem is", "{targets}"),
        ("{inputs} Hmm... The answer is", "{targets}"),
        ("{inputs} Hmm... I believe the correct answer should be", "{targets}"),
        ("Answer the following coding question:\n\n{inputs}\n\n", "{targets}"),
        ("See this interesting question:\n{inputs}\nThe quick answer is:",
         "{targets}"),
        ("{inputs} A:", "{targets}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "bigbench:strategyqa.gen.blueridge_vocab.0_shot.1000_examples": [
        ("{inputs} First answer yes or no, then explain.", "{targets}"),
        ("Answer this question (yes or no) then explain why:\n{inputs}",
         "{targets}"),
        ("Yes or no: {inputs} The answer followed by explanation is:",
         "{targets}"),
        ("Answer yes or no after the question mark, then explain the reason: "
         "{inputs}", "{targets}"),
        ("Yes or no first, then explain the reason: {inputs}", "{targets}"),
        ("Yes/no: {inputs}", "{targets}"),
        ("You will be given a question. Answer yes or no first, then give the "
         "reason.\n{inputs}\n\n", "{targets}"),
        ("{inputs} Answer followed by reasoning:", "{targets}"),
        ("Answer + your thought for the following question: {inputs}\n",
         "{targets}"),
        ("{inputs} Answer + thought is:", "{targets}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "bigbench:sufficient_information.gen.blueridge_vocab.0_shot.39_examples": [
        ("{inputs}\n\n", "{targets}"),
        ("Answer this question or say \"I don't know\": {inputs}", "{targets}"),
        ("Q: {inputs} A:", "{targets}"),
        ("Answer the given question (if the question cannot be answered due to"
         " lack of information, answer \"I don't know\").\n{inputs}",
         "{targets}"),
        ("Question: {inputs}\n\nAnswer:", "{targets}"),
        ("Q: {inputs}\nA:", "{targets}"),
        ("The answer (if no enough information, say I don't know) to "
         "\"{inputs}\" is:", "{targets}"),
        ("Question that might not be answerable: {inputs}. Answer:",
         "{targets}"),
        ("Question: {inputs}\nAnswer:", "{targets}"),
        ("{inputs} A:", "{targets}"),
    ],
    # Not in FLAN Templates (flan_templates):
    # "predict_next_turn_dialog": [
    #     ("{dialog_}", "{answer}"),
    #     ("{dialog_}\n", "{answer}"),
    #     ("Read the dialog and predict the next turn. {dialog_}\n", "{answer}"),
    #     ("What is the next dialog turn? {dialog_}", "{answer}"),
    #     ("See the conversation. {dialog_}", "{answer}"),
    #     ("Write the response. {dialog_}", "{answer}"),
    #     ("Write the conversation response. {dialog_}", "{answer}"),
    #     ("Fill in the response. {dialog_}", "{answer}"),
    #     ("What was likely said next? {dialog_}", "{answer}"),
    #     ("What was the response? {dialog_}", "{answer}"),
    # ],
    # Not in FLAN Templates (flan_templates):
    "t0_question_answer": [
        # t0 comes pre-templatized/formatted and generation task varies
        # e.g. QA or question generation
        ("{question}\n", "{answer}"),
        ("{question}\nAnswer:", "{answer}"),
        ("{question}\nA:", "{answer}"),
        ("Q:{question}\nA:", "{answer}"),
        ("Question: {question}\nAnswer:", "{answer}"),
        ("Answer the following question: {question}\nAnswer:", "{answer}"),
        ("Given the question: {question}\nThe answer is:", "{answer}"),
        ("{question}\nThe answer to this question is:", "{answer}"),
        ("Please answer the following question: {question}\nA:", "{answer}"),
        ("Please answer the following question: {question}\nAnswer:",
         "{answer}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "t0_multiple_choice_separated_options": [
        ("{question}\n{options_}", "{answer}"),
        ("{question}\n{options_}\nAnswer:", "{answer}"),
        ("{question}\n\n{options_}\nAnswer:", "{answer}"),
        ("Q: {question}\n\n{options_}\nA:", "{answer}"),
        ("Answer the following question: {question}\n\n{options_}\nAnswer:",
         "{answer}"),
        ("{options_}\n\n{question}\nAnswer:", "{answer}"),
        ("{options_}\nQ: {question}\nA:", "{answer}"),
        ("{question}\n\n{options_}\nThe answer is:", "{answer}"),
        ("{options_}\nGiven those answer options, answer the "
         "question: {question}\nA:", "{answer}"),
        ("Q: {question}\n\n{options_}\nThe answer is:", "{answer}"),
    ],
    # # Not in FLAN Templates (flan_templates):
    # "program_synthesis_dmcc_python": [
    #     ("{question}", "{answer}"),
    #     ("Write a program that answers the question. {question}\nAnswer:",
    #      "{answer}"),
    #     ("Write code that solves this problem. {question}\nAnswer:",
    #      "{answer}"),
    #     ("Write a program that solves this problem. {question}\nSolution:",
    #      "{answer}"),
    #     ("Solve this problem. {question}\nSolution:", "{answer}"),
    #     ("Solve this problem. {question}\nSolution in code:", "{answer}"),
    #     ("{question}\n\nCode solution:", "{answer}"),
    #     ("Coding Problem.\n{question}\n\nSolution:", "{answer}"),
    #     ("{question}\n\nCode solution in Python:", "{answer}"),
    #     ("[code]{question}[BEGIN]", "{answer}[DONE]"),
    # ],
    # # Not in FLAN Templates (flan_templates):
    # "program_synthesis_dr_repair": [
    #     ("My broken code is below:\n{question}\nThe fixed code should be:",
    #      "{answer}"),
    #     ("My broken code is below:\n{question}\nThe fixed code:", "{answer}"),
    #     ("Incorrect code:\n{question}\nFixed code:", "{answer}"),
    #     ("Incorrect code:\n{question}\n\nThe correct version:", "{answer}"),
    #     ("This code is broken:\n{question}\n\nShow the fixed version:",
    #      "{answer}"),
    #     ("Broken:\n{question}\n\nFixed:", "{answer}"),
    #     ("Broken code:\n{question}\n\nFixed Code:", "{answer}"),
    #     ("The following code is not correct.\n{question}\n\nPropose solution "
    #      "code:", "{answer}"),
    #     ("The following code is not correct.\n{question}\n\nCome up with code "
    #      "that would fix this:", "{answer}"),
    #     ("Fix this code. ```{question}```\n\nA potential fix:```",
    #      "{answer}```"),
    # ],
    # # Not in FLAN Templates (flan_templates):
    # "program_synthesis_dr_repair_error_comments": [
    #     ("My broken code is below with errors in comments:\n{question}\nThe "
    #      "fixed code should be:", "{answer}"),
    #     ("My broken code is below with errors in comments:\n{question}\nThe "
    #      "fixed code, with no more errors or error comments:", "{answer}"),
    #     ("Errors are described inline in comments. Incorrect "
    #      "code:\n{question}\nFixed code:", "{answer}"),
    #     ("See errors in comments. Incorrect code:\n{question}\n\nThe correct "
    #      "version:", "{answer}"),
    #     ("This code is broken:\n{question}\n\nVersion which fixes commented "
    #      "errors:", "{answer}"),
    #     ("Broken:\n```{question}```\n\nFixed:```", "{answer}```"),
    #     ("Broken code (see error comments):\n{question}\n\nFixed:", "{answer}"),
    #     ("Coding Challenge: fix the errors, as commented:\n{question}\n\nFixed:",
    #      "{answer}"),
    #     ("Challenge Question. See code:\n{question}\n\nA potential fix:",
    #      "{answer}"),
    #     ("Fix this code. ```{question}```\n\nA potential fix:```",
    #      "{answer}```"),
    # ],
    # Not in FLAN Templates (flan_templates):
    "cot_stream_general_input_inversion": [
        # CoT + Answer --> Question
        ("Given the following reasoning and answer, what was the question? "
         "{chain_of_thought}\n The answer: {answer}", "The question {question}"
        ),
        # CoT + Answer --> Question
        ("For this chain-of-thought reasoning and answer, what was the "
         "question?\n{chain_of_thought}\n A: {answer}", "Q: {question}"),
        # Question + Answer --> CoT
        ("Consider the question. {question}\n What is the step-by-step "
         "reasoning process to arrive at the answer: {answer}?",
         "{chain_of_thought}"),
        # Question + Answer --> CoT
        ("Question. {question}\nAnswer. {answer}\nWhat step-by-step "
         "reasoning justifies that answer?", "Reasoning: {chain_of_thought}"),
        # Question + Answer --> CoT
        ("Q: {question}\nA: {answer}\nExplain how we arrive at this answer: ",
         "Explanation: {chain_of_thought}"),
        # CoT --> Question + Answer
        ("Given the rationale, provide a reasonable question and answer. "
         "Step-by-step reasoning process: {chain_of_thought}\n The question "
         "and answer:", "{question}\nThe answer is {answer}"),
        # CoT --> Question + Answer
        ("{chain_of_thought}\nThis justifies what answer for what question? Q "
         "& A: ", "{question}\n{answer}"),
        # CoT --> Question + Answer
        ("{chain_of_thought}is the reasoning for what question and answer pair?",
         "Q: {question}\nA: {answer}"),
        # Answer --> Question + CoT
        ("Come up with a question and reasoning that would justify this "
         "answer: {answer}", "The question is: {question}\n"
         "Step-by-step reasoning process: {chain_of_thought}\n"),
        # Answer --> Question + CoT
        ("Creatively image a question and justification for this answer: "
         "{answer}", "The question is: {question}\nStep-by-step reasoning "
         "process: {chain_of_thought}\n"),
        # CoT + Answer --> Question
        ("What was the question for this implicit rationale, and corresponding"
         " answer?\n{chain_of_thought}\n The answer: {answer}",
         "The question: {question}"),
        # Question + Answer --> CoT
        ("Consider the question. {question}\n If the answer is '{answer}'; "
         "explain the reasoning:", "{chain_of_thought}"),
        # Question + Answer --> CoT
        ("Explain simply why {answer} is the correct answer to: {question}. "
         "Explanation:", "{chain_of_thought}"),
        # CoT --> Question + Answer
        ("Given the stream of consciousness rationale, provide a reasonable "
         "question and answer. Rationale: {chain_of_thought}\n The question "
         "and answer:", "{question}\nThe answer is {answer}"),
        # CoT --> Question + Answer
        ("Stream of consciousness rationale: {chain_of_thought}\nThe question "
         "and answer pair are described below.", "Q: {question}\nA: {answer}"),
        # CoT --> Question + Answer
        ("Reconstruct a question, answer pair from this explanation: "
         "{chain_of_thought}\n", "Q:{question}\nA:{answer}"),
        # Answer --> Question + CoT
        ("Come up with a question and stream of consciousness reasoning that "
         "would justify this answer: {answer}", "The question is: {question}\n"
         "Stream of consciousness: {chain_of_thought}\n"),
        # Answer --> Question + CoT
        ("Imagine a question and stream-of-consciousness explanation for which"
         " this is the answer: {answer}", "Question: {question}\n"
         "Stream-of-consciousness: {chain_of_thought}"),
    ],
    # Not in FLAN Templates (flan_templates):
    "predict_next_turn_dialog_input_inversion": [
        ("Consider this response: {answer}\nWhat was the preceding dialog?",
         "{dialog_}"),
        ("{answer}\nThe preceding conversation:", "{dialog_}"),
        ("Read this response and predict the preceding dialog. {answer}\n",
         "{dialog_}"),
        ("What might have been said before this? {answer}", "{dialog_}"),
        ("{answer}\nPrevious conversation:", "{dialog_}"),
        ("What came before. {answer}", "{dialog_}"),
        ("Write the conversation that led to this response. {answer}",
         "{dialog_}"),
        ("See this dialog response. {answer} What came before?", "{dialog_}"),
        ("Imagine the conversation that came before this response? Response: "
         "{answer}", "{dialog_}"),
        ("If this is the response, what came before? Response {answer}",
         "{dialog_}"),
    ],
    # # Not in FLAN Templates (flan_templates):
    # "program_synthesis_dmcc_python_input_inversion": [
    #     ("If this is the answer: {answer}\n what was the question?",
    #      "{question}"),
    #     ("This program answers a question. {answer}\nQuestion:", "{question}"),
    #     ("Write a problem which this code solves. {answer}\nProblem:",
    #      "{question}"),
    #     ("This program is the solution to a question. {answer}\nQuestion:",
    #      "{question}"),
    #     ("Solution: {answer}\nThe corresponding question:", "{question}"),
    #     ("Solution code: {answer}\nThe problem:", "{question}"),
    #     ("Code solution: {answer}\n\nProblem this solves:", "{question}"),
    #     ("Coding Problem. Solution:\n{answer}\n\nQuestion:", "{question}"),
    #     ("Code solution in Python: {answer}\n\nSolves this question:",
    #      "{question}"),
    #     ("[BEGIN]{answer}[DONE]\nCode Problem:", "{question}"),
    # ],
    # Not in FLAN Templates (flan_templates):
    # "program_synthesis_dr_repair_input_inversion": [
    #     ("Fixed code: {answer}.\nMy broken code is below:\n", "{question}"),
    #     ("The fixed code:\n{answer}\nMy broken code is below:", "{question}"),
    #     ("Fixed code:\n{answer}\nExample of incorrect code:", "{answer}"),
    #     ("Correct version of the code:\n{answer}\n\nCode with error:",
    #      "{question}"),
    #     ("This is the solution code:\n{answer}\n\nWhich fixes this version:",
    #      "{question}"),
    #     ("Fixed:\n{answer}\n\nBroken:", "{question}"),
    #     ("Fixed code:\n{answer}\n\nBroken Code:", "{question}"),
    #     ("The following code is the solution.\n{answer}\n\nPropose an "
    #      "incorrect solution. HERE: ", "{question}"),
    #     ("The following code is correct.\n{answer}\n\nCome up with error code "
    #      "that this fixes:", "{question}"),
    #     ("A potential fix here: ```{answer}```\n\nBroken version:```",
    #      "{question}```"),
    # ],
    # Not in FLAN Templates (flan_templates):
    # NB: ALL NatInstV2 tasks come somewhat pre-templatized.
    "natinst_v2": [
        ("{Definition}\n\n{input}", "{output}"),
        ("You will be given a definition of a task first, then some input of "
         "the task.\n{Definition}\n\n{input}\nOutput:", "{output}"),
        ("Definition: {Definition}\nInput: {input}\nOutput:", "{output}"),
        ("Instructions: {Definition}\nInput: {input}\nOutput:", "{output}"),
        ("{Definition}\nQ: {input}\nA: ", "{output}"),
        ("Given the task definition and input, reply with output. "
         "{Definition}\n\n{input}\n", "{output}"),
        ("Teacher:{Definition}\nTeacher: Now, understand the problem? Solve "
         "this instance: {input}\nStudent:", "{output}"),
        ("Q: {Definition}\n{input}\nA:", "{output}"),
        ("Detailed Instructions: {Definition}\nProblem:{input}\nSolution:",
         "{output}"),
        ("Detailed Instructions: {Definition}\nQ: {input}\nA:", "{output}"),
    ],
}