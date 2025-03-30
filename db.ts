// RAG API client

export async function sendQueryToRAG(query: string) {
  try {
    // Use the proxy server running on port 3001
    const response = await fetch('http://localhost:3001/api/proxy/query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
        // No need to send API key - proxy will add it
      },
      body: JSON.stringify({ query })
    });

    if (!response.ok) {
      const errorData = await response.json();
      console.error('API error:', response.status, errorData);
      throw new Error(`API error: ${response.status} - ${errorData.error || 'Unknown error'}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error calling RAG API:', error);
    throw error;
  }
} 