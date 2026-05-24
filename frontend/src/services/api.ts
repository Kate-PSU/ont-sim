// frontend/src/services/api.ts
// API функции для взаимодействия с бэкендом

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');

export interface WikipediaSimilarityResponse {
  domains: string[];
  matrix: number[][];
  top_pairs: Array<{ d1: string; d2: string; score: number }>;
}

export const getWikipediaSimilarity = async (): Promise<WikipediaSimilarityResponse> => {
  const response = await fetch(`${API_BASE_URL}api/v1/similarity/wikipedia`);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return response.json();
};

export interface EnsembleResult {
  similarity: number;
  sbert_score: number;
  tfidf_score: number;
  weights: { sbert: number; tfidf: number };
}

export const getEnsembleSimilarity = async (
  domain1: string,
  domain2: string,
  weights?: { sbert: number; tfidf: number }
): Promise<EnsembleResult> => {
  const params = new URLSearchParams();
  if (weights) {
    params.set('sbert_weight', weights.sbert.toString());
    params.set('tfidf_weight', weights.tfidf.toString());
  }
  const response = await fetch(`${API_BASE_URL}/api/v1/similarity/ensemble/${encodeURIComponent(domain1)}/${encodeURIComponent(domain2)}?${params}`);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return response.json();
};

export const getDomains = async (): Promise<string[]> => {
  const response = await fetch(`${API_BASE_URL}/api/v1/domains`)
  if (!response.ok) return []
  const data = await response.json()
  return data.domains || []
}
