import streamlit as st
from pytube import extract
from youtube_transcript_api import YouTubeTranscriptApi
from spacy_streamlit import visualize_ner, load_model
from collections import Counter
import random

# Load spaCy model using spacy-streamlit
nlp = load_model("en_core_web_sm")

def get_transcript(video_url):
    video_id = extract.video_id(video_url)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join([elem["text"] for elem in transcript])

def create_mcqs(input_text, question_count=5):
    processed_doc = nlp(input_text)
    sentence_list = [sentence.text for sentence in processed_doc.sents]
    selected_sentences = random.sample(sentence_list, min(question_count, len(sentence_list)))
    mcq_list = []
    for sentence in selected_sentences:
        sentence_doc = nlp(sentence)
        noun_list = [token.text for token in sentence_doc if token.pos_ == "NOUN"]
        if len(noun_list) < 2:
            continue
        noun_frequency = Counter(noun_list)
        if noun_frequency:
            main_noun = noun_frequency.most_common(1)[0][0]
            question_format = sentence.replace(main_noun, "__________")
            choices = [main_noun]
            for _ in range(3):
                distractor = random.choice(list(set(noun_list) - set([main_noun])))
                choices.append(distractor)

            random.shuffle(choices)

            correct_option = chr(64 + choices.index(main_noun) + 1)
            mcq_list.append((question_format, choices, correct_option))

    return mcq_list

# Streamlit app
st.title("YouTube Transcript MCQ Generator")

# Input field for the YouTube video URL
video_url = st.text_input("Enter YouTube Video URL")

if st.button("Generate MCQs"):
    if video_url:
        # Extract transcript from the YouTube video
        try:
            transcript_text = get_transcript(video_url)
            st.success("Transcript extracted successfully!")
            
            # Generate MCQs
            mcqs = create_mcqs(transcript_text, question_count=5)

            # Display the generated MCQs
            if mcqs:
                for i, mcq in enumerate(mcqs, start=1):
                    question_stem, answer_choices, correct_answer = mcq
                    st.write(f"Q{i}: {question_stem}?")
                    for j, choice in enumerate(answer_choices, start=1):
                        st.write(f"{chr(64+j)}: {choice}")
                    st.write(f"Correct Answer: {correct_answer}\n")
            else:
                st.warning("No suitable sentences found to create MCQs.")

            # Optional: Visualize named entities in the transcript
            st.subheader("Named Entities in Transcript")
            visualize_ner(transcript_text, model=nlp)

        except Exception as e:
            st.error(f"Error extracting transcript: {str(e)}")
    else:
        st.warning("Please enter a valid YouTube URL.")
