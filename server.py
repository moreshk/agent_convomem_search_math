from flask import Flask, render_template, request, redirect, url_for, session
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# The rest of your imports here
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import initialize_agent
from langchain import LLMMathChain

# Setting up the chatbot
llm = OpenAI(temperature=0)
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

search = GoogleSearchAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
    ),
]

memory = ConversationBufferMemory(memory_key="chat_history")
llm=OpenAI(temperature=0)
agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

app = Flask(__name__)
app.secret_key = '1234'  # It's better to use a random value

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if 'chat_history' not in session:
        session['chat_history'] = []

    if request.method == 'POST':
        user_input = request.form.get('message')
        bot_response = agent_chain.run(user_input) # Using the chatbot here
        
        session['chat_history'].append({'user': user_input, 'bot': bot_response})
        session.modified = True
        
    return render_template('chat.html', chat_history=session['chat_history'])
        # return render_template('chat.html', user_input=user_input, bot_response=bot_response)
    # return render_template('chat.html')

if __name__ == '__main__':
    app.run(debug=True)
