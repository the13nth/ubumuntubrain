# Document Embedding and Search System

A RAG (Retrieval-Augmented Generation) system that allows you to upload documents, generate embeddings, and perform semantic search with Gemini AI responses.

## Features

- Document upload (PDF, TXT)
- Semantic search using embeddings
- RAG with Google's Gemini AI
- 3D visualization of document embeddings
- External API access

## Setup

1. Clone the repository
2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your API keys:

   ```
   GOOGLE_API_KEY=your_gemini_api_key_here
   EXTERNAL_API_KEY=your_chosen_api_key_here
   ```

5. Run the application:

   ```bash
   python app.py
   ```

## External API Documentation

### Endpoint Details

- **URL**: `http://localhost:5000/api/external/query`
- **Method**: POST
- **Authentication**: API Key (via X-API-Key header)

### Request Format

#### Headers

```
Content-Type: application/json
X-API-Key: your_external_api_key_here
```

#### Body

```json
{
    "query": "your search query here"
}
```

### Response Format

```json
{
    "answer": "Generated answer from Gemini AI",
    "query_embedding_visualization": {
        "x": 0.123,
        "y": 0.456,
        "z": 0.789
    }
}
```

### Example Calls

#### cURL

```bash
curl -X POST http://localhost:5000/api/external/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_external_api_key_here" \
  -d '{"query": "your search query here"}'
```

#### Python

```python
import requests
import json

url = "http://localhost:5000/api/external/query"
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "your_external_api_key_here"
}
data = {
    "query": "your search query here"
}

response = requests.post(url, headers=headers, json=data)
result = response.json()
print(result)
```

#### JavaScript

```javascript
const response = await fetch('http://localhost:5000/api/external/query', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'X-API-Key': 'your_external_api_key_here'
    },
    body: JSON.stringify({
        query: 'your search query here'
    })
});
const result = await response.json();
console.log(result);
```

### Error Responses

#### Invalid API Key (401)

```json
{
    "error": "Invalid API key"
}
```

#### Missing Query (400)

```json
{
    "error": "Query is required"
}
```

#### Server Error (500)

```json
{
    "error": "Error message details"
}
```

### Notes

- The API key must match the one set in your `.env` file
- The query should be a text string
- The response includes both the AI-generated answer and 3D coordinates for visualization
- The web interface will show a notification when the API is called
- All API calls are logged in the web interface with the option to re-run queries

## UI Features

- Real-time API call notifications
- Visual indicator for API status
- Query history display
- 3D embedding visualization
- Document management interface
