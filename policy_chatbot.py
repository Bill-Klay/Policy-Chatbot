import os
from openai import OpenAI
import docx2txt
import PyPDF2
from flask import Flask, Response, request
from waitress import serve

app = Flask(__name__)

# Load configuration from .config file
app.config.from_pyfile('.config')

# Access the configuration values
API_KEY = app.config['OPENAI_API_KEY']
DOCUMENTS_FOLDER_PATH = app.config['DOCUMENTS_FOLDER_PATH']

def read_text_from_file(file_path):
    """
    Reads text from a file with supported formats (.docx, .pdf, .txt).

    Parameters:
    file_path (str): The path to the file to be read.

    Returns:
    str: The text extracted from the file.
    """
    if file_path.endswith('.docx'):
        text = docx2txt.process(file_path)
    elif file_path.endswith('.pdf'):
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ''
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as txt_file:
            text = txt_file.read()

    return text

def chat_with_bot(user_query):
    """
    Initiates a chat with the OpenAI bot using the provided user query.

    Parameters:
    user_query (str): The user's query to send to the OpenAI bot.

    Returns:
    openai.Completion: The response from the OpenAI bot.
    """
    history_openai_format = []
    folder = os.listdir(DOCUMENTS_FOLDER_PATH)
    for filename in folder:
        file = os.path.join(DOCUMENTS_FOLDER_PATH, filename)
        file_contents = read_text_from_file(file.strip())
        history_openai_format.append({"role": "user", "content": file_contents})
    history_openai_format.append({"role": "user", "content": user_query})
    client = OpenAI(api_key=API_KEY)
    response = client.chat.completions.create(
        messages=history_openai_format,
        model="gpt-4-1106-preview",
        stream=True
    )
    return response

def process_chatbot(prompt):
    """
    Processes the chatbot response for a given prompt and yields content chunks.

    Parameters:
    prompt (str): The prompt to send to the chatbot.

    Yields:
    str: Content chunks from the chatbot response.
    """
    response = chat_with_bot(prompt)
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

@app.route('/prompt', methods=['GET'])
def execute_chatbot():
    """
    Flask route to execute the chatbot and stream the response.

    Returns:
    Response: A Flask Response object that streams the chatbot's response.
    """
    prompt = request.args.get('prompt')
    return Response(process_chatbot(prompt), mimetype='text')

@app.route('/', methods=['GET'])
def index():
    """
    Flask route to display a welcome message and confirm the server is running.

    Returns:
    str: A welcome message.
    """
    return "Hello Friend! Yes, the server is running 🏃"

if __name__ == "__main__":
    HOST = os.environ.get('SERVER_HOST', '10.0.100.176')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '6969'))
    except ValueError:
        PORT = 5555

    print("Server running at ", HOST, " @ ", PORT)
    serve(app.app, host=HOST, port=PORT)
