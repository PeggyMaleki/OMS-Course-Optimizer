import datetime
import streamlit as st 
import requests
import json
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np 
import re
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import openai

openai.api_key = 'sk-proj-uNGyg7PNZrnM8VY5OTMkT3BlbkFJzSgQirrK7Q843NDZiYYb'

#I will first scrape OMSCentral data and put it into a data frame. 
def scrape_oms_central():   
    url = "https://www.omscentral.com" #accessing the URL

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200). Got help from chatGPT here
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        course_tags = soup.find_all(string=re.compile(r'"courses":'))
        script_content = ''.join(str(tag) for tag in course_tags)
        try:
            script_json = json.loads(script_content) #chatGPT wrote this
            # Access the "courses" list
            courses_list = script_json['props']['pageProps']['courses'] #chatGPT wrote this

            # Initialize lists to store data. I will use these for my dataframe 
            course_id_list = []
            course_name_list = []
            avg_difficulty_list = []
            avg_hours_peer_week_list = []
            rating_list = []
            foundational_requirement_list = []
            description_list = []
            credit_hours_list = []

            # Extract data from each course dictionary in courses_list
            for course in courses_list:
                course_id_list.append(course['codes'][0])
                course_name_list.append(course['name'])
                avg_difficulty_list.append(course.get('difficulty', None)) 
                avg_hours_peer_week_list.append(course.get('workload', None))
                rating_list.append(course.get('rating', None))
                foundational_requirement_list.append(course['isFoundational'])
                description_list.append(course.get('description'))
                credit_hours_list.append(course.get('creditHours'))

                #After initially printing out the list, some entries were inputted as "NaN" [not a number]. This little code block replace NaN with None. chatGPT wrote this
                #avg_difficulty_list = [difficulty if isinstance(difficulty, (int, float)) else None for difficulty in avg_difficulty_list]
                #avg_hours_peer_week_list = [hours if isinstance(hours, (int, float)) else None for hours in avg_hours_peer_week_list]
            
            #I wanted to display the full dataframe. chatGPT wrote this
            pd.set_option('display.max_rows', None)  # Show all rows
            pd.set_option('display.max_columns', None)  # Show all columns

            # Create DataFrame
            df = pd.DataFrame({
                'Course ID': course_id_list,
                'Course Name': course_name_list,
                'Difficulty': [round(difficulty, 2) if isinstance(difficulty, (int, float)) else None for difficulty in avg_difficulty_list],
                'OMSC Hours': [round(difficulty, 2) if isinstance(difficulty, (int, float)) else None for difficulty in avg_hours_peer_week_list],
                'Rating': [round(difficulty, 2) if isinstance(difficulty, (int, float)) else None for difficulty in rating_list],
                'Meets Foundational Requirement': foundational_requirement_list,
                'Description': description_list,
                'Credit Hours': credit_hours_list
            })

            # Replace NaN with None in the DataFrame. Got help with code from chatGPT on this
            return df
            
        except json.JSONDecodeError as e: #chatGPT wrote this exception block
            print(f"Error decoding JSON: {e}")
    else:
            print(f"Failed to retrieve the page. Status code: {response.status_code}")

#displaying the scraped data frame. 
def display_scraped_data():
    df = scrape_oms_central()
    if df is not None:
        st.subheader("List of Available OMS Courses")
        st.write("this data has been scraped from OMSCentral.")
        st.table(df)

#mkaing sure entries are valid. chatGPT wrote this function
def validate_positive_integer(integer_str):
    try:
        number = float(integer_str)
        if number < 0:
            raise ValueError
    except ValueError:
        st.warning("Please enter a valid positive integer.")
        return False
    return True

#ChatGPT wrote this. Making sure entries are valid.
def custom_number_input(label, default_value=0):
    input_value = st.text_input(label, value=str(default_value))
    while not validate_positive_integer(input_value):
        input_value = st.text_input(label, value=str(default_value), key='text_input_invalid', help="Invalid Input")
    return float(input_value)

#collecting user constraints
def user_constraints():
    constraints = {} #holding constraints in a dict

    st.write("The following questions will ask you for your study constraints to best gauge the amount of hours you have to study and dedicate to a class.\n")

    #getting user constraints through streamlit
    default_sleeptime = datetime.time(23, 0)
    default_waketime = datetime.time(7, 0)
    constraints['sleep_start'] = st.time_input("At what time do you usually go to sleep?", value=default_sleeptime)
    constraints['sleep_end'] = st.time_input("At what time do you usually wake up?", value=default_waketime)
    constraints['work_hours'] = custom_number_input("How many hours do you work per week?") 
    constraints['leisure_hours'] = custom_number_input("How many hours of leisure time do you want per week?")
    constraints['hobbies'] = custom_number_input("How many hours do you spend on hobbies per week?")
    constraints['workout_hours'] = custom_number_input("How many hours (if any) do you spend on health and fitness per week?")
    constraints['chores_hours'] = custom_number_input("How many hours do you spend on chores and other obligations outside of work per week?")
    constraints['family_hours'] = custom_number_input("How many hours do you spend with family per week?")
    constraints['dinner_duration'] = custom_number_input("How long does dinner usually take? (Enter in minutes)")

    return constraints

def user_learning_prefs():
    learning_prefs = {}  # Holding preferences in a dict

    st.write("Please provide information about your learning goals and skill levels. We will use these to help calculate the amount of time you need to spend on studying.")

    skill_level_mapping = {'Average': 'average', 'Above Average': 'above average', 'Below Average': 'below average'}
    skill_level = st.selectbox("What are your skill levels in coding?", options=['Average', 'Above Average', 'Below Average'])
    learning_goal = st.selectbox("What grade are you striving for in your classes?", options=['A', 'B', 'C'])
    
    learning_prefs['skill_level'] = skill_level_mapping[skill_level]
    learning_prefs['learning_goal'] = learning_goal
    
    return learning_prefs

#collecting user learning preferences
def calculate_study_hours(learning_prefs, constraints):
    # Constants for calculation
    leisure_hours_per_week = constraints['leisure_hours']
    chores_hours_per_week = constraints['chores_hours']
    workout_hours_per_week = constraints['workout_hours']
    hobbies_hours_per_week = constraints['hobbies']
    family_hours_per_week = constraints['family_hours']
    work_hours_per_week = constraints['work_hours']
    dinner_hours_per_week = constraints['dinner_duration'] * 7 / 60  # Convert dinner duration from minutes to hours

    # Calculate total hours available for studying per week. Got help from ChatGPT for datetime.
    sleep_time = datetime.datetime.combine(datetime.datetime.now().today(), constraints['sleep_start'])
    wake_time = datetime.datetime.combine(datetime.datetime.now().today(), constraints['sleep_end'])

    if sleep_time > wake_time:
        # Adjust sleep time to be on the previous day
        sleep_time -= datetime.timedelta(days=1)
        hours_sleeping_per_week = wake_time - sleep_time
    # Calculate the difference in hours
    if sleep_time < wake_time:
        # If sleep time is before wake time on the same day
        hours_sleeping_per_week = wake_time - sleep_time
    
    hours_sleeping_per_week = hours_sleeping_per_week.total_seconds() * 7 / 3600  # Convert seconds to hours

    total_hours_available_per_week = (24 * 7) - (leisure_hours_per_week + chores_hours_per_week + workout_hours_per_week + family_hours_per_week + work_hours_per_week + dinner_hours_per_week + hours_sleeping_per_week + hobbies_hours_per_week)
    total_hours_available_per_week = round(total_hours_available_per_week, 2)
    
    return total_hours_available_per_week

def calculate_true_hours(raw_data, skill_level, learning_goal):
    skill_multipliers = {'below average': 1.25, 'average': 1.0, 'above average': 0.75}
    goal_multipliers = {'A': 1.25, 'B': 1.0, 'C': 0.75}
    
    multiplier_skill = skill_multipliers.get(skill_level, 1.0)
    multiplier_goal = goal_multipliers.get(learning_goal, 1.0)

    true_hours_df = raw_data.copy()  # Create a copy of the original DataFrame
    # Calculate estimated hours with multiplier effects
    true_hours_df['Estimated Hours'] = round(true_hours_df['OMSC Hours'] * multiplier_skill * multiplier_goal, 2)
    
    return true_hours_df

def calculate_similarity(embedding1, embedding2):
    embedding1_vector = embedding1['last_hidden_state']
    embedding2_vector = embedding2['last_hidden_state']
    
    return cosine_similarity([embedding1_vector], [embedding2_vector])[0][0]


# Streamlit app. Using the Streamlit documentation: https://docs.streamlit.io/library/api-reference/data
def main():
    #title of websitee 
    st.title("OMS Course and Study Optimizer")

    #display either home screen with scraped data or go into side pages
  
    st.header("Schedule Optimizer")
    st.write("The questions below will ask you to input your time constraints and learning preferences in order to gauge the actual amount of study time you have available.")

    constraints = user_constraints()
    learning_prefs =  user_learning_prefs()
        
    # Calculate study hours per week
    study_hours_per_week = calculate_study_hours(learning_prefs, constraints)

    st.write("Based on your inputs, the recommended study hours per week are:", study_hours_per_week)
                
    classes_df = calculate_true_hours(scrape_oms_central(), learning_prefs['skill_level'], learning_prefs['learning_goal'])
    filtered_df = classes_df[classes_df['Estimated Hours'] < study_hours_per_week]
        
    column_order = ['Course ID', 'Course Name', 'Estimated Hours', 'OMSC Hours', 'Difficulty', 'Rating', 'Description', 'Credit Hours', 'Meets Foundational Requirement']
    filtered_df = filtered_df.reindex(columns=column_order)
    filtered_df = filtered_df.dropna(subset=['Description'])
    
    st.write('''Below, you'll find a comprehensive list of OMS courses tailored to fit your schedule. 
                Estimated Hours provides an approximate duration for each class based on your skill level 
                and desired grade. For personalized recommendations aligned with specific learning goals 
                and time constraints, please answer the next optional question.''')
    
    st.dataframe(filtered_df, use_container_width=True)
    
    user_input = st.text_input("What do you want to learn? (Optional)")
        
    if user_input:
        context = ""
        for _, row in filtered_df.iterrows():
            # Converting all filtered courses into context for OpenAI
            context += f"{row['Course Name']} ({row['Course ID']}): {row['Description']}\n"

        prompt = '''From the following course list and course descriptions, recommend up to 5 classes for a student
                    who has stated the following learning goal:''' + user_input
        
        prompt = prompt + '''. \n\n Only pick courses that directly relate to the student's learning goal.
                                Order the recommendations based on how closely the course matches the student's specific
                                learning objectives, with the most relevant courses being the first recommendation. 
                                For each course, summarize what that course teaches and why it would be a good course for the student
                                based on the student's learning objective. Structure the response directly towards the student. Refer to them as you.'''

        prompt = prompt + '''\n\n The following are the courses to choose from: ''' + context

        # Call OpenAI API to generate response using GPT-3.5 Turbo engine
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=350)
        
        st.write(response['choices'][0]['text'])

if __name__ == "__main__":
    main()

