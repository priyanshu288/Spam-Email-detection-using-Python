import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Load data
df = pd.read_csv('mail_data.csv')

# Preprocess data
data = df.where((pd.notnull(df)), '')
data.loc[data['Category'] == 'spam', 'Category',] = 0
data.loc[data['Category'] == 'ham', 'Category',] = 1

# Train-test split
X = data['Message']
Y = data['Category']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Feature extraction
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# Train the model
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Spam Emails Section
def spam_emails_section():
    st.header("Introduction to Spam Emails")

    st.subheader("What are Spam Emails?")
    st.write("Spam emails, also known as junk emails, are unwanted messages sent to a large number of recipients.")
    
    st.subheader("Characteristics of Spam Emails")
    st.write("- Unsolicited: Users did not request the information.")
    st.write("- Often contain misleading information or offers.")
    st.write("- May contain phishing attempts to steal sensitive information.")
    st.write("- Use various techniques to bypass spam filters.")

# Developer 1 Section
def developer1_section():
    st.header("Priyanshu Tandon(06113302720)")
    st.write("Priyanshu is a skilled full-stack developer specializing in creating interactive web applications.")
    st.write("GitHub: [Developer 1](https://github.com/priyanshu28)")

# Developer 2 Section
def developer2_section():
    st.header("Ankit Kumar(00513302720)")
    st.write("Ankit is a passionate web developer skilled with latest tech stack and experience.")
    st.write("GitHub: [Developer 2](https://github.com/ankitkumar020591)")

# Developer 3 Section
def developer3_section():
    st.header("Ayush(01713302720)")
    st.write("Ayush is a skilled full-stack developer specializing in creating interactive web applications.")
    st.write("GitHub: [Developer 3](https://github.com/ayush007)")

# Spam Classifier Section
def spam_classifier_section():
    st.header("Spam Classifier App")

    # User input
    user_input = st.text_area("Enter an email message:", "Hello, you have won 1000$, click the link to claim")

    if st.button("Classify"):
        # Make prediction
        input_data_features = feature_extraction.transform([user_input])
        prediction = model.predict(input_data_features)

        # Display result
        if prediction[0] == 1:
            st.success("Not spam")
        else:
            st.error("Spam mail")

def main():
    st.title("Spam or Not Spam Classifier")

    # Navigation
    page = st.sidebar.selectbox("Select Page", ["Introduction", "Spam Emails", "Developer 1", "Developer 2", "Developer 3", "Spam Classifier"])

    if page == "Introduction":
        st.header("Introduction to the Spam Classifier App")
        st.write("This app helps you classify whether an email is spam or not.")

    elif page == "Spam Emails":
        spam_emails_section()

    elif page == "Developer 1":
        developer1_section()

    elif page == "Developer 2":
        developer2_section()

    elif page == "Developer 3":
        developer3_section()

    elif page == "Spam Classifier":
        spam_classifier_section()

if __name__ == "__main__":
    main()
