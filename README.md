# social-media-rag

> A simple social media RAG (retrieval-augmented generation) application using Beeper Matrix.

## Features

- Aggregating up to 10 social media (i.e., Discord, Slack, WhatsApp, etc.) privately using the Beeper Matrix protocol.
- Aggregated data is used by RAG (retrieval augmented generation) to answer user queries.
- RAG implementation variant can be by Graphlit (SaaS) or LangChain (Milvus, Cohere, GoogleAI).
- The application supports processing image and text data.

## Tutorial

### 1. Install the dependencies.

```bash
uv pip install -e . -U
```

### 2. Run the app variant.

```bash
streamlit run app.py
```

### 3. Sidebar configurations.

- Input API Credentials.
- Input Beeper Matrix Credentials.
- Input Room ID(s).
- Input settings.
- Click on the "Sync" button.
- Optionally, click "Reset" button to clear the data.

### 5. Question and Answer.

- Input your query in the text box.
- Click on the "Submit" button.
- Click citation button to view citation details.

## Demonstration

![demo-1.png](assets/demo-1.png)