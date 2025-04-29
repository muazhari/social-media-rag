# social-media-rag

> A simple social media RAG (retrieval-augmented generation) application using Graphlit and Beeper Matrix.

## Features

- Aggregating up to 10 social media (i.e., Discord, Slack, WhatsApp, etc.) privately using the Beeper Matrix protocol.
- Aggregated data is used by Graphlit RAG (retrieval augmented generation) to answer user queries.
- The application supports processing image and text data.

## Tutorial

### 1. Install the dependencies.

```bash
uv pip install -e . -U
```

### 2. Run the app.

```bash
streamlit run main.py
```

### 3. Sidebar configurations.

- Input Graphlit Credentials.
- Input Beeper Matrix Credentials.
- Input Room ID(s).
- Input settings.
- Click on the "Sync" button.

### 5. Question and Answer.

- Input your query in the text box.
- Click on the "Submit" button.
- Click citation button to view citation details.

## Demonstration

![demo-1.png](assets/demo-1.png)