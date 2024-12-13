# pip install langchain openai requests
# pip install beautifulsoup4 requests

from bs4 import BeautifulSoup
import requests
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI

# Step 1: Scrape the web page
url = "https://en.wikipedia.org/wiki/Wikipedia"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")
text = soup.get_text()

# Step 2: Prepare the text for LangChain
documents = [{"text": text}]

# Step 3: Initialize OpenAI LLM
llm = ChatOpenAI(temperature=0, openai_api_key="your_openai_api_key")

# Step 4: Summarize the content
chain = load_summarize_chain(llm, chain_type="map_reduce")
summary = chain.run(documents)

print("Summary of the Article:")
print(summary)