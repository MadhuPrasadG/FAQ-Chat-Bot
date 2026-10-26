import re
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- STEP 1: Your FAQ Data ---
faq_data = {
    "What are the library timings?": "The library is open from 8:00 AM to 10:00 PM on weekdays, and 9:00 AM to 5:00 PM on weekends.",
    "How can I apply for a hostel room?": "You can apply for a hostel room through the student portal under the 'Accommodation' tab.",
    "When do the final exams start?": "Final exams for the current semester begin on November 15th. Please check your department notice board for the exact schedule.",
    "Where is the computer lab located?": "The main computer lab is located on the second floor of the Science and Technology building, Room 204.",
    "Who do I contact for IT support?": "For IT support, you can email itsupport@university.edu or visit the helpdesk on the ground floor.",
    "who is the head of the computer science department?":"The head of the Computer Science Department is Dr.Ambili , You can reach her at Staff room Located in the 3ed floor of the Science and Technology building.",
    "where is the Data Science final year project exhibition?": "The Data Science final year project exhibition will be held in the main auditorium on the 1st floor of the Science and Technology building on December 10th.",
    "Where is the Classrooms for Msc Data Science?": "The classrooms for MSc Data Science are located in the Science and Technology building, Rooms 301 to 305.",
    "What is the minimum attendance requirement?": "Students must maintain a minimum of 75% attendance in all subjects to be eligible for the final semester exams.",
    "How can I pay my semester fees?": "Semester fees can be paid online through the university payment portal using a credit card, debit card, or net banking.",
    "Where is the cafeteria located?": "The main campus cafeteria is located on the ground floor of the Student Center, next to the sports complex.",
    "What is the process for getting a parking permit?": "You can apply for a student parking permit at the campus security office by providing your student ID and vehicle registration details.",
    "How do I access the campus Wi-Fi?": "Connect to the 'Campus-Secure' network and log in using your student ID and the password provided during orientation.",
    "Are there any internship opportunities for Data Science students?": "Yes, the placement cell regularly shares internship opportunities. Please register on the career portal and upload your updated resume.",
    "Who is the academic advisor for the Data Science program?": "Prof. Sharma is the academic advisor for MSc Data Science. You can meet him during his office hours on Tuesdays and Thursdays from 2 PM to 4 PM.",
    "How do I join the coding club?": "You can join the coding club by attending their introductory session on the first Friday of the month or by registering on the student council website.",
    "When is the deadline for submitting the final year project report?": "The final year project report must be submitted to your respective project guide on or before November 30th.",
    "How can I request an official transcript?": "Official transcripts can be requested by filling out the transcript request form at the registrar's office. It usually takes 5 to 7 business days to process.",
    "Can I pay my tuition with pizza?": "While the accounting department loves pizza, they strictly accept legal tender. Nice try, though.",
    "Will this chatbot do my Data Science project for me?": "Nice try! I am just a simple FAQ chatbot. You still have to write your own Python code.",
    "Is it okay to attend morning classes in my pajamas?": "Technically, there is no dress code against it, but your professors will definitely silently judge you.",
    "What do I do if a campus monkey steals my laptop?": "Do not negotiate with the monkeys. Report the theft to campus security immediately and consider carrying decoy bananas.",
    "Can I bring my pet iguana to the computer lab?": "Only if it knows how to code in Python. Otherwise, pets are strictly prohibited in the academic buildings.",
    "Does Dr. Ambili accept bribes in the form of coffee?": "Her love for coffee is legendary, but she is incorruptible. You still have to study for your exams.",
    "Where can I hide to avoid my final exams?": "There is no escape from finals. However, the library basement is very quiet if you just want to panic in peace.",
    "How many energy drinks is it safe to consume before a coding deadline?": "The health center recommends sleeping instead. If you can see sounds or hear colors, you have had too many.",
    "Is the Wi-Fi password really 'password123'?": "No, this is a university, not a local cafe. Check your orientation packet for the secure login details.",
    "Can I use ChatGPT to write my apologies for arriving late?": "Yes, but your professor will probably use ChatGPT to write your detention slip."
}

questions = list(faq_data.keys())
answers = list(faq_data.values())

# --- STEP 2 & 3: Preprocess and Vectorize ---
def preprocess_text(text):
    text = text.lower() 
    text = re.sub(r'[^\w\s]', '', text) 
    return text.strip() 

cleaned_questions = [preprocess_text(q) for q in questions]
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(cleaned_questions)

# --- STEP 4 & 5: Calculate similarity ---
# Notice we added 'history' here because Gradio automatically passes chat history!
def get_response(user_query, history):
    cleaned_query = preprocess_text(user_query)
    query_vector = vectorizer.transform([cleaned_query])
    similarities = cosine_similarity(query_vector, question_vectors)
    
    best_match_index = similarities.argmax()
    highest_score = similarities[0, best_match_index]
    
    if highest_score < 0.3:
        return "I'm sorry, I don't understand your question. Could you rephrase it?"
        
    return answers[best_match_index]

# --- STEP 6: CREATE THE WEB INTERFACE ---
# This creates a beautiful chat UI automatically!
demo = gr.ChatInterface(
    fn=get_response, 
    title="🤖 University FAQ Chatbot",
    description="Ask me anything about campus facilities, Data Science classes, or... campus monkeys."
)

if __name__ == "__main__":
    demo.launch()