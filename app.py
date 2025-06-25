import os
import time
import logging
import requests
from flask import Flask, request, render_template
from azure.ai.textanalytics import TextAnalyticsClient, ExtractiveSummaryAction, AbstractiveSummaryAction
from azure.core.credentials import AzureKeyCredential

app = Flask(__name__, template_folder="templates")

# ------------------ SDK Client ------------------
def authenticate_client():
    endpoint = os.environ["AZURE_LANGUAGE_ENDPOINT"]
    key = os.environ["AZURE_LANGUAGE_KEY"]
    return TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))


# ------------------ Extractive Summary ------------------
def get_extractive_summary(client, text):
    try:
        poller = client.begin_analyze_actions(
            [text],
            actions=[ExtractiveSummaryAction(max_sentence_count=3)],
            language="en"
        )
        result = poller.result()
        for doc in result:
            for action_result in doc:
                if action_result.is_error:
                    return f"Extractive Error: {action_result.error.message}"
                return "\n".join(sentence.text for sentence in action_result.sentences)
    except Exception as e:
        logging.exception("Extractive summarization error:")
        return f"Error during extractive summarization: {str(e)}"


# ------------------ Abstractive Summary ------------------
def get_abstractive_summary(client, text):
    try:
        poller = client.begin_analyze_actions(
            [text],
            actions=[AbstractiveSummaryAction()],
            language="en"
        )
        result = poller.result()
        for doc in result:
            for action_result in doc:
                if action_result.is_error:
                    return f"Abstractive Error: {action_result.error.message}"
                return "\n".join(summary.text for summary in action_result.summaries)
    except Exception as e:
        logging.exception("Abstractive summarization error:")
        return f"Error during abstractive summarization: {str(e)}"


# ------------------ Web Route ------------------
@app.route("/", methods=["GET", "POST"])
def index():
    input_text = ""
    extractive_summary = ""
    abstractive_summary = ""

    if request.method == "POST":
        input_text = request.form.get("project_data")
        if input_text and len(input_text.strip()) > 50:
            client = authenticate_client()
            extractive_summary = get_extractive_summary(client, input_text)
            abstractive_summary = get_abstractive_summary(client, input_text)
        else:
            extractive_summary = "Please enter at least 50 characters of text."
            abstractive_summary = ""

    return render_template(
        "index.html",
        input_text=input_text,
        extractive_summary=extractive_summary,
        abstractive_summary=abstractive_summary
    )

if __name__ == "__main__":
    app.run(debug=True)
