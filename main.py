import streamlit as st
import analysis


def front() :
    st.title("Sentiment Analysis")

    text = st.text_input("Enter your Data")
    btn = st.button("Press me")
    if btn:
        if text :
            st.write("OK got your text")
        else : st.write("Enter something")
        st.write("Your Text is: ",analysis.predict(text))

if __name__ == "__main__":
    front()