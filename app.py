import streamlit as st
import pickle
import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# Loading models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Resume cleaning function
def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

# User-friendly interface with minimal words and clean design
def main():
    st.title("🎯 Resume Screening App")
    st.markdown("**Upload your resume to see the predicted job category!**")
    
     # Instructions and layout improvement
    st.markdown("""
    **Instructions**:
    1. Click on **Browse files** to upload a resume in `.txt` or `.pdf` format.
    2. The app will clean the text and predict the relevant job category.
    """)

    uploaded_file = st.file_uploader("", type=['txt', 'pdf'])

    if uploaded_file is not None:
        resume_bytes = uploaded_file.read()
        try:
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = clean_resume(resume_text)

        # Display progress
        with st.spinner('Analyzing...'):
            input_features = tfidf.transform([cleaned_resume])
            prediction_id = clf.predict(input_features)[0]

        # Mapping prediction to job category
        category_mapping = {
            15: "Java Developer", 23: "Testing", 8: "DevOps Engineer", 20: "Python Developer",
            24: "Web Designing", 12: "HR", 13: "Hadoop", 3: "Blockchain", 10: "ETL Developer",
            18: "Operations Manager", 6: "Data Science", 22: "Sales", 16: "Mechanical Engineer",
            1: "Arts", 7: "Database", 11: "Electrical Engineering", 14: "Health and fitness",
            19: "PMO", 4: "Business Analyst", 9: "DotNet Developer", 2: "Automation Testing",
            17: "Network Security Engineer", 21: "SAP Developer", 5: "Civil Engineer",
            0: "Advocate"
        }

        category_name = category_mapping.get(prediction_id, "Unknown")
        
        # Display result
        st.success(f"🔍 Job Category: **{category_name}**")

    else:
        st.info("Drag and drop your resume above ⬆️")

if __name__ == "__main__":
    main()
