import os
import logging
from flask import Flask, request, render_template
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

app = Flask(__name__, template_folder="templates")

def authenticate_client():
    endpoint = os.environ["AZURE_LANGUAGE_ENDPOINT"]
    key = os.environ["AZURE_LANGUAGE_KEY"]
    return TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))

def get_extractive_summary(client, text):
    try:
        poller = client.begin_extractive_summarization([text], language="en", sentences_count=3)
        result = poller.result()
        return "\n".join(
            sentence.text for doc in result if doc.kind == "ExtractiveSummarization" for sentence in doc.sentences
        )
    except Exception as e:
        logging.error(f"Extractive error: {e}")
        return "Error during extractive summarization."

def get_abstractive_summary(client, text):
    try:
        poller = client.begin_abstractive_summarization([text], language="en")
        result = poller.result()
        return "\n".join(
            summary.text for doc in result if doc.kind == "AbstractiveSummarization" for summary in doc.summaries
        )
    except Exception as e:
        logging.error(f"Abstractive error: {e}")
        return "Error during abstractive summarization."

@app.route("/", methods=["GET", "POST"])
def index():
    extractive_summary = ""
    abstractive_summary = ""
    input_text = ""

    if request.method == "POST":
        input_text = request.form.get("project_data")
        if input_text:
            client = authenticate_client()
            extractive_summary = get_extractive_summary(client, input_text)
            abstractive_summary = get_abstractive_summary(client, input_text)

    return render_template(
        "index.html",
        input_text=input_text,
        extractive_summary=extractive_summary,
        abstractive_summary=abstractive_summary
    )

if __name__ == "__main__":
    app.run(debug=True)
